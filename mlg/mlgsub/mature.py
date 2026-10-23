"""Maturation: after the furnace picks a family, the MLG grows one brain for
hours (manifesto LXIV: the size Q says, not the size the body allows).

The furnace judges genomes on short budgets, which favours small brains.
Before a long run the MLG sizes the brain to the budget it now has: its
self-model gives the compute the lab can deliver in that time, a brain of
N weights needs about BYTES_PER_WEIGHT bytes of text per weight to use it
(ASSUMED, recorded as such), and the field body caps what can live there.
The run checkpoints to disk every few minutes and resumes from there when
the lab dies, as Colab sessions do.
"""
import math
import os
import time

import torch

from . import brain as B

BYTES_PER_WEIGHT = 40  # ASSUMED: bytes of text a byte-level weight needs (~10 BPE tokens)
CONTEXT = 1024          # room for a passage, a question and the answer


def plan(furnace, hours, log=print):
    """Choose the maturation genome; returns (genome, inheritance or None)."""
    sub, lab, home = furnace.sub, furnace.lab, furnace.home
    champ = furnace.champion
    base = dict(champ["g"]) if champ else dict(B.ANCESTOR)
    dtype = ("bf16" if lab.device.type == "cuda" and "bf16" in lab.dtypes
             and not lab.dtypes["bf16"].get("unsupported") else "fp32")
    flops = hours * 3600 * lab.tflops(dtype if dtype in lab.dtypes else "fp32", 4096) * 1e12 * lab.efficiency
    target = math.sqrt(flops / (6 * BYTES_PER_WEIGHT))
    best = None
    for d in B.GENES["d_model"]:
        for L in B.GENES["n_layers"]:
            heads = max([h for h in B.GENES["n_heads"] if d % h == 0 and d // h >= 32] or [0])
            if not heads:
                continue
            g = dict(base, d_model=d, n_layers=L, n_heads=heads, ctx=CONTEXT, dtype=dtype,
                     batch=64 if d <= 512 else 32, lr=1e-3 if d <= 384 else 6e-4, dropout=0.0)
            if not B.valid(g):
                continue
            p = B.predict(g, lab, furnace.deploy)
            if not p["fits_lab"] or (furnace.deploy is not None and not p["fits_deploy"]):
                continue
            dist = abs(math.log(p["active_params"] / target))
            if best is None or dist < best[0]:
                best = (dist, g, p)
    if best is None:
        raise RuntimeError("no brain fits both the lab and the field body")
    _, g, p = best
    inh = None
    if champ and champ["genome"] in furnace.states:
        weights, mode = B.inherit(g, champ["g"], furnace.states[champ["genome"]][1])
        if weights is not None:
            inh = (weights, mode, champ["brain"], champ["genome"])
    gid = B.genome_id(g)
    sub.obj("Genome", gid, origin="mature.plan", epistemic="EXPERIMENTAL", generation=-1, home=home.id,
            **g, **{f"pred_{k}": v for k, v in p.items()})
    if champ:
        sub.link(champ["genome"], "PARENT_OF", gid, mutated=["maturation"])
    did = f"decision:{home.id}:mature-{gid[7:13]}"
    sub.obj("Decision", did, origin="mature.plan", epistemic="INFERRED", verdict="MATURE", stage="PLAN",
            body=home.id, hours=hours, compute_flops=flops, target_params=target,
            bytes_per_weight_assumed=BYTES_PER_WEIGHT,
            reason=f"{hours:.1f} h of this lab at self-model efficiency {lab.efficiency:.2f} can feed about "
                   f"{target / 1e6:.0f}M weights; the closest brain that lives in {home.name or home.id} "
                   f"has {p['active_params'] / 1e6:.1f}M active")
    sub.link(did, "CHOSE", gid)
    for t in ("self_model_efficiency", "usable_memory"):
        if t in lab.traits:
            sub.link(did, "JUSTIFIED_BY", lab.traits[t])
    if furnace.deploy is not None:
        for t in ("latency_budget", "prefill_budget", "dram_bandwidth"):
            if t in furnace.deploy.traits:
                sub.link(did, "JUSTIFIED_BY", furnace.deploy.traits[t])
    sub.event("MATURE_PLAN", gid, target_params=target, params=p["params"], hours=hours)
    log(f"[maturation] {hours:.1f} h can feed ~{target / 1e6:.0f}M weights. Growing d{g['d_model']} "
        f"L{g['n_layers']} E{g['n_experts']}k{g['top_k']} ctx{g['ctx']} {g['dtype']}: "
        f"{p['params'] / 1e6:.1f}M params, {p.get('deploy_latency_ms', 0):.1f} ms/byte predicted on the field body"
        + (f", inheriting the champion ({inh[1]})" if inh else ", from scratch"))
    return g, inh


def mature(furnace, g, hours, ckpt_dir, inheritance=None, checkpoint_every=600, log=print):
    """Train g for `hours` (resuming from ckpt_dir), then put it through the
    furnace's ATTACK and COMPARE like any other brain. Returns the furnace row."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"mature-{B.genome_id(g)[7:19]}.pt")
    resume = None
    if os.path.exists(path):
        resume = torch.load(path, map_location="cpu", weights_only=True)
        log(f"[maturation] resuming at step {resume['steps']} ({resume['elapsed'] / 3600:.2f} h done)")
        furnace.sub.event("MATURE_RESUME", B.genome_id(g), steps=resume["steps"], elapsed=resume["elapsed"])

    def save(state):
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)  # a dying lab never leaves half a checkpoint
        log(f"[maturation] checkpoint at step {state['steps']}, {state['elapsed'] / 3600:.2f} h")

    t = time.time()
    furnace.seconds_override = hours * 3600
    row = furnace.run(g, generation=-1, parents=(furnace.champion["genome"],) if furnace.champion else (),
                      mutated=("maturation",), inheritance=inheritance,
                      train_kw=dict(resume=resume, checkpoint=save, checkpoint_every=checkpoint_every))
    furnace.seconds_override = None
    log(f"[maturation] done in {(time.time() - t) / 3600:.2f} h this session: {row['status']}"
        + (f", dev {row['dev_bpb']:.3f} bpb" if "dev_bpb" in row else ""))
    return row
