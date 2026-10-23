"""Brain genomes: the MLG designs its own neural core (manifesto XXIV, LIV).

A genome is a compact description of a byte-level mixture-of-experts
Transformer, the same family as the Resident Core in core/tf.c. Before a
genome is trained, its cost is PREDICTED from the body's measured traits
(XXII); a genome that cannot fit is never run. After training, prediction is
compared with reality and the error corrects the body's self-model (XXI).

Inheritance is Lamarckian: a child keeps what its parent learned whenever the
shapes allow it. A deeper child gets its parent's layers plus new layers that
start as the identity (zeroed output projections), so growing never forgets;
a child with more experts gets copies of its parent's experts (knowledge
phase transitions, manifesto XXXI).
"""
import hashlib
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .body import DTYPES, EnergyMeter, _sync

GENES = {
    "d_model": [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536],
    "n_layers": [2, 3, 4, 6, 8, 12, 16, 24],
    "n_heads": [2, 4, 8, 12, 16],
    "n_experts": [1, 2, 4, 8, 16],
    "top_k": [1, 2, 4],
    "ffn_mult": [2, 3, 4],
    "ctx": [128, 256, 512, 1024],
    "dtype": ["fp32", "tf32", "bf16", "fp16"],
    "lr": [1e-4, 3e-4, 6e-4, 1e-3, 2e-3],
    "batch": [16, 32, 64, 128, 256],
    "dropout": [0.0, 0.1, 0.2],
}
# Genes a child can change and still inherit its parent's weights.
INHERITABLE = ["n_layers", "n_experts", "top_k", "n_heads", "ctx", "dtype", "lr", "batch",
               "dropout"]
GROWTH = ("n_layers", "n_experts")

# The architecture of the current Resident Core (README: 4 layers, 4 experts,
# one active per token, context 256), re-expressed at byte level so it can
# compete on equal terms. It is the baseline every child must beat.
ANCESTOR = {"d_model": 192, "n_layers": 4, "n_heads": 4, "n_experts": 4, "top_k": 1,
            "ffn_mult": 4, "ctx": 256, "dtype": "fp32", "lr": 1e-3, "batch": 32,
            "dropout": 0.1}


def valid(g):
    return (g["d_model"] % g["n_heads"] == 0 and g["d_model"] // g["n_heads"] >= 16
            and g["top_k"] <= g["n_experts"])


def genome_id(g):
    return "genome:" + hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()[:12]


def random_genome(rng, dtypes):
    while True:
        g = {k: rng.choice(v) for k, v in GENES.items()}
        g["dtype"] = rng.choice(dtypes)
        if valid(g):
            return g


def mutate(g, rng, dtypes, n=None, genes=None, grow=False):
    """Mostly small steps along a gene's axis; returns (child, genes changed).
    `genes` limits which genes may change; `grow` biases depth and experts up."""
    pool = list(genes or GENES)
    for _ in range(100):
        child, changed = dict(g), []
        for gene in rng.sample(pool, min(len(pool), n or rng.choice([1, 1, 2, 3]))):
            options = dtypes if gene == "dtype" else GENES[gene]
            if gene != "dtype" and g[gene] in options and rng.random() < 0.8:
                up = grow and gene in GROWTH and rng.random() < 0.75
                i = options.index(g[gene]) + (1 if up else rng.choice([-1, 1]))
                value = options[max(0, min(len(options) - 1, i))]
            else:
                value = rng.choice(options)
            if value != g[gene]:
                child[gene] = value
                changed.append(gene)
        if changed and valid(child):
            return child, changed
    return random_genome(rng, dtypes), list(GENES)


def crossover(a, b, rng):
    for _ in range(100):
        child = {k: (a if rng.random() < 0.5 else b)[k] for k in GENES}
        if valid(child):
            return child
    return dict(a)


def params(g):
    """(total, active per byte), counted exactly like the module below."""
    d, L, E, k, f = g["d_model"], g["n_layers"], g["n_experts"], g["top_k"], g["ffn_mult"]
    emb = 256 * d + g["ctx"] * d
    gate = d * E if E > 1 else 0
    per = 4 * d * d + gate + 4 * d
    total = emb + L * (per + E * 2 * f * d * d) + 2 * d
    active = emb + L * (per + k * 2 * f * d * d) + 2 * d
    return total, active


def predict(g, lab, deploy=None):
    """Cost of a genome before it exists (PREDICTED). `deploy` is the body the
    brain must live in when that is not the lab it trains in."""
    total, active = params(g)
    B, T, d, L = g["batch"], g["ctx"], g["d_model"], g["n_layers"]
    act_bytes = 2 if g["dtype"] in ("bf16", "fp16") else 4
    # weights + grads + Adam moments, plus stored activations; attention
    # scores are assumed not stored (fused kernels) until the peak says otherwise
    train_mem = (16 * total + B * T * L * d * (18 + 4 * g["ffn_mult"] * g["top_k"]) * act_bytes) \
        * lab.mem_factor
    flops = 6 * active * B * T + 12 * L * B * T * T * d
    n_eff = (B * T * d * g["ffn_mult"] * d) ** (1 / 3)
    dtype = g["dtype"] if g["dtype"] in lab.dtypes else "fp32"
    step_s = flops / (lab.tflops(dtype, n_eff) * 1e12 * lab.efficiency)
    out = {"params": total, "active_params": active, "train_bytes": int(train_mem),
           "step_seconds": step_s, "fits_lab": train_mem < 0.85 * lab.usable_bytes}
    if deploy is not None:
        infer_bytes = 4 * total
        flop_s = deploy.tflops("fp32", 256) * 1e12
        lat_ms = 1000 * max(2 * active / flop_s, 4 * total / (deploy.dram_gbs * 1e9))
        out.update(deploy_bytes=infer_bytes, deploy_latency_ms=lat_ms,
                   fits_deploy=infer_bytes <= deploy.deploy_bytes and lat_ms <= deploy.latency_ms)
    return out


# ---- the neural core ------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.up = nn.Linear(d, f * d, bias=False)
        self.down = nn.Linear(f * d, d, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class MoE(nn.Module):
    def __init__(self, d, f, n, k):
        super().__init__()
        self.n, self.k = n, k
        self.experts = nn.ModuleList(Expert(d, f) for _ in range(n))
        self.gate = nn.Linear(d, n, bias=False) if n > 1 else None
        self.aux = torch.zeros(())

    def forward(self, x):
        B, T, C = x.shape
        h = x.reshape(-1, C)
        if self.gate is None:
            return self.experts[0](h).view(B, T, C)
        probs = self.gate(h).float().softmax(-1)
        w, idx = probs.topk(self.k, -1)
        # top-k weights sum to one. With k = 1 that weight is the constant 1 and the
        # router would get no gradient from the loss; dividing by its detached value
        # keeps the forward pass identical while the gradient still reaches the gate.
        w = w / (w.detach() if self.k == 1 else w.sum(-1, keepdim=True))
        out = torch.zeros_like(h)
        for e in range(self.n):
            hit = idx == e
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel():
                we = (w * hit).sum(-1)[rows].unsqueeze(-1).to(h.dtype)
                out.index_add_(0, rows, (self.experts[e](h[rows]) * we).to(out.dtype))
        # load balance: fraction routed x mean gate probability (Switch-style)
        frac = F.one_hot(idx[:, 0], self.n).float().mean(0)
        self.aux = self.n * (frac * probs.mean(0)).sum()
        return out.view(B, T, C)


class Block(nn.Module):
    def __init__(self, g):
        super().__init__()
        d = g["d_model"]
        self.h = g["n_heads"]
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.moe = MoE(d, g["ffn_mult"], g["n_experts"], g["top_k"])
        self.drop = g["dropout"]

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.drop if self.training else 0.0)
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(B, T, C)), self.drop, self.training)
        return x + F.dropout(self.moe(self.ln2(x)), self.drop, self.training)


class Brain(nn.Module):
    def __init__(self, g):
        super().__init__()
        d = g["d_model"]
        self.ctx = g["ctx"]
        self.tok = nn.Embedding(256, d)
        self.pos = nn.Embedding(g["ctx"], d)
        self.blocks = nn.ModuleList(Block(g) for _ in range(g["n_layers"]))
        self.ln = nn.LayerNorm(d)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.ln(x) @ self.tok.weight.t()  # output tied to input bytes
        aux = sum(b.moe.aux for b in self.blocks if b.moe.gate is not None)
        return logits, aux


def snapshot(model):
    """A brain's learned weights, kept on the host so children can inherit them."""
    return {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}


def inherit(child, parent, pstate):
    """Map a parent's weights onto a child's shape. Returns (state, mode) or
    (None, None) when the shapes share nothing (width or FFN size changed).
    mode: clone (same architecture), grow (deeper or more experts), prune
    (shallower or fewer experts), grow+prune (both), reshape (heads, top-k or
    context changed). Growing deeper preserves the parent's function exactly;
    splitting experts preserves it only up to the symmetry-breaking noise, and
    only with top-1 routing (with top-k > 1 the copies of the best expert take
    the slots its runners-up held)."""
    if pstate is None or child["d_model"] != parent["d_model"] or child["ffn_mult"] != parent["ffn_mult"]:
        return None, None
    Lp, Lc, Ep, Ec = parent["n_layers"], child["n_layers"], parent["n_experts"], child["n_experts"]
    out = {}
    for k, v in Brain(child).state_dict().items():
        if k.startswith("blocks."):
            parts = k.split(".")
            if int(parts[1]) >= Lp:  # a new layer begins as the identity
                out[k] = torch.zeros_like(v) if k.endswith(("proj.weight", "down.weight")) else v
            elif ".moe.experts." in k:
                e = int(parts[4])
                w = pstate[k.replace(f"experts.{e}.", f"experts.{e % Ep}.")].clone()
                if e >= Ep:  # split an expert; noise breaks the symmetry
                    w += 1e-3 * w.std() * torch.randn_like(w)
                out[k] = w
            elif k.endswith("moe.gate.weight"):
                if Ep > 1:
                    gate = pstate[k]
                    out[k] = torch.stack([gate[e % Ep] + (1e-3 * torch.randn_like(gate[0]) if e >= Ep else 0)
                                          for e in range(Ec)])
                else:  # one expert becomes many identical ones: uniform routing
                    out[k] = torch.zeros_like(v)
            else:
                out[k] = pstate[k].clone()
        elif k == "pos.weight":
            n = min(v.shape[0], pstate[k].shape[0])
            v = v.clone()
            v[:n] = pstate[k][:n]
            out[k] = v
        else:
            out[k] = pstate[k].clone()
    grew, shrank = Lc > Lp or Ec > Ep, Lc < Lp or Ec < Ep
    if grew and shrank:
        mode = "grow+prune"
    elif grew:
        mode = "grow"
    elif shrank:
        mode = "prune"
    elif all(child[x] == parent[x] for x in ("n_heads", "top_k", "ctx")):
        mode = "clone"
    else:
        mode = "reshape"
    return out, mode


# ---- data: train / dev / frozen Q root ------------------------------------
class Corpus:
    """Train on the MLG corpus, select on dev, and keep the Q root sealed.
    Any Q root or dev line that also appears in training is removed from the
    training text before a single step runs (no evaluation leaks)."""

    TRAIN = ["core/data/corpus.txt", "core/data/corpus-mlg.txt", "core/data/corpus-dialog.txt"]
    DEV = ["core/data/valid.txt", "core/data/valid-mlg.txt", "core/data/valid-dialog.txt"]
    QROOT = "trust/qroot/language.txt"
    GATE = "trust/gate/dialog.txt"  # held-out conversation, carved by tools/carve_gate.py

    def __init__(self, root, device, max_train_bytes=None, extra=()):
        """`extra`: more training text files (for example a Wikipedia dump),
        cleaned of evaluation lines like the MLG corpus."""
        read = lambda p: open(os.path.join(root, p), "rb").read()

        def sealed(path, lock):
            blob = read(path)
            sha = hashlib.sha256(blob).hexdigest()
            words = open(os.path.join(root, lock)).read().split()
            return blob, sha, dict(zip(words[1::2], words[0::2])).get(os.path.basename(path)) == sha

        self.qroot_bytes, self.qroot_hash, self.qroot_sealed = sealed(self.QROOT, "trust/qroot.lock")
        self.gate_bytes, self.gate_hash, self.gate_sealed = sealed(self.GATE, "trust/gate.lock")
        dev = b"\n".join(read(p) for p in self.DEV)
        held = {l.strip() for l in b"\n".join([self.qroot_bytes, self.gate_bytes, dev]).splitlines()
                if len(l.strip()) > 8}
        kept, removed, self.extra, own = [], 0, [], 0
        for p in list(self.TRAIN) + [os.path.abspath(x) for x in extra]:
            blob = read(p)
            if p not in self.TRAIN:
                self.extra.append((p, len(blob), hashlib.sha256(blob).hexdigest()))
            for line in blob.splitlines(keepends=True):
                if line.strip() in held:
                    removed += 1
                else:
                    kept.append(line)
                    if p in self.TRAIN:
                        own += len(line)
        train = b"".join(kept)
        del kept
        # the MLG's own text comes first; batches draw half their rows from it
        # so a large extra corpus adds knowledge without drowning the MLG's voice
        self.own_bytes = min(own, len(train))
        if max_train_bytes:
            train = train[:max_train_bytes]
        self.removed_lines = removed
        to = lambda b: torch.frombuffer(bytearray(b), dtype=torch.uint8).to(device)
        self.train, self.dev = to(train), to(dev)
        self.qroot, self.gate = to(self.qroot_bytes), to(self.gate_bytes)
        self.train_hash = hashlib.sha256(train).hexdigest()

    def batch(self, B, T):
        dev, n = self.train.device, len(self.train)
        if n - self.own_bytes > 4 * T and self.own_bytes > T + 1:
            k = B // 2
            ix = torch.cat([torch.randint(self.own_bytes - T - 1, (k,), device=dev),
                            torch.randint(self.own_bytes, n - T - 1, (B - k,), device=dev)])
        else:
            ix = torch.randint(n - T - 1, (B,), device=dev)
        w = self.train[ix[:, None] + torch.arange(T + 1, device=ix.device)].long()
        return w[:, :-1], w[:, 1:]

    def shuffled_dev(self, seed=0):
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(self.dev), generator=gen).to(self.dev.device)
        return self.dev[perm]


def autocast(g, device):
    if g["dtype"] in ("bf16", "fp16"):
        return torch.autocast(device.type, dtype=DTYPES[g["dtype"]])
    return torch.autocast(device.type, enabled=False)


@torch.no_grad()
def byte_nats(model, g, data, device, max_bytes=1 << 20, batch=32):
    """Nats spent on every byte of `data` after the first. Windows slide by half
    a context and each byte is scored once, with as much history as the brain
    can hold, so brains of any context length are scored on the same bytes."""
    model.eval()
    data = data[:max_bytes].long()
    N = len(data)
    L = min(g["ctx"], N - 1)
    if L < 1:
        model.train()
        return torch.empty(0)
    stride = max(1, L // 2)
    starts = list(range(0, N - 1 - L, stride)) + [N - 1 - L]
    out = torch.empty(N - 1, device=data.device)
    span = torch.arange(L + 1, device=data.device)
    done = 0  # bytes 1..done are scored
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        w = data[torch.tensor(chunk, device=data.device)[:, None] + span]
        with autocast(g, device):
            logits, _ = model(w[:, :-1])
        nats = F.cross_entropy(logits.float().transpose(1, 2), w[:, 1:], reduction="none")
        for row, s in zip(nats, chunk):
            out[done:s + L] = row[done - s:]
            done = s + L
    model.train()
    return out


def bits_per_byte(model, g, data, device, **kw):
    nats = byte_nats(model, g, data, device, **kw)
    return nats.mean().item() / math.log(2) if len(nats) else float("nan")


def paired_gain(champion_nats, challenger_nats, block=64, samples=2000, seed=0):
    """Bits per byte the challenger saves over the champion on the same bytes,
    with a 95% interval from a block bootstrap (neighbouring bytes are not
    independent, so whole blocks are resampled)."""
    d = (champion_nats - challenger_nats).double().cpu() / math.log(2)
    n = len(d) // block
    if n < 2:
        m = d.mean().item()
        return m, m, m
    blocks = d[: n * block].view(n, block).mean(1)
    gen = torch.Generator().manual_seed(seed)
    boot = blocks[torch.randint(n, (samples, n), generator=gen)].mean(1)
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=boot.dtype)).tolist()
    return d.mean().item(), lo, hi


def q_language(bpb):
    """The Q root's own formula (trust/qroot/policy.json)."""
    return max(0.0, 1.0 - bpb / 8.0)


@torch.no_grad()
def generate(model, g, prompt, device, n=160, temperature=0.7, seed=0):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    idx = torch.tensor([list(prompt.encode())], device=device)
    for _ in range(n):
        with autocast(g, device):
            logits, _ = model(idx[:, -g["ctx"]:])
        p = (logits[0, -1].float() / temperature).softmax(-1)
        nxt = torch.multinomial(p, 1, generator=gen)
        idx = torch.cat([idx, nxt.view(1, 1)], 1)
        if nxt.item() == 10 and idx.shape[1] > len(prompt.encode()) + 8:
            break
    model.train()
    return bytes(idx[0].tolist()).decode("utf-8", "replace")


def load_brain(path, device="cpu"):
    """A committed brain from its checkpoint: (model, genome)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = Brain(ck["genome"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck["genome"]


def reply(model, g, text, device, n=240, temperature=0.6, seed=None):
    """One dialogue turn in the MLG corpus format (U: question, M: answer)."""
    device = torch.device(device)
    prompt = f"U: {text.strip()}\nM:"
    out = generate(model, g, prompt, device, n=n, temperature=temperature,
                   seed=seed if seed is not None else random.randrange(1 << 30))
    return out[len(prompt):].split("\nU:")[0].strip()


@torch.no_grad()
def latency_ms(model, g, device, reps=8):
    """Measured ms per generated byte at full context."""
    model.eval()
    x = torch.randint(0, 256, (1, g["ctx"]), device=device)
    for _ in range(2):
        with autocast(g, device):
            model(x)
    _sync(device)
    t = time.perf_counter()
    for _ in range(reps):
        with autocast(g, device):
            model(x)
    _sync(device)
    model.train()
    return 1000 * (time.perf_counter() - t) / reps


def train(g, lab, corpus, seconds, seed=0, init_state=None, lr_scale=1.0, resume=None,
          checkpoint=None, checkpoint_every=600.0):
    """Train for a fixed wall-clock budget on the lab body, from scratch or
    from inherited weights. Returns the model and what actually happened, to
    be compared with predict().

    `lr_scale` lowers the peak rate for a brain that inherited a trained mind,
    so its first steps refine what it knows instead of shaking it apart.
    `checkpoint(state)` is called every `checkpoint_every` seconds and at the
    end; passing such a state back as `resume` continues an interrupted run
    with the same weights, optimizer moments, step count and schedule."""
    torch.manual_seed(seed)
    dev = lab.device
    model = Brain(g)
    if resume is not None:
        model.load_state_dict(resume["model"])
    elif init_state is not None:
        model.load_state_dict(init_state)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=g["lr"], betas=(0.9, 0.95), weight_decay=0.1,
                            fused=dev.type == "cuda")
    scaler = torch.amp.GradScaler(dev.type, enabled=g["dtype"] == "fp16")
    steps, before = 0, 0.0
    if resume is not None:
        opt.load_state_dict(resume["opt"])
        scaler.load_state_dict(resume["scaler"])
        steps, before = resume["steps"], resume["elapsed"]
    peak = g["lr"] * lr_scale
    warm = 50 if init_state is None and resume is None else 10
    meter = EnergyMeter(dev)
    loss_v, diverged = float("nan"), False

    def state(elapsed):
        return {"model": snapshot(model), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "steps": steps, "elapsed": elapsed, "genome": g}

    old_tf32 = None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = g["dtype"] == "tf32"
    try:
        with meter:
            t0 = last_ck = time.perf_counter()
            while True:
                now = time.perf_counter()
                frac = (before + now - t0) / seconds
                if frac >= 1:
                    break
                if checkpoint and now - last_ck >= checkpoint_every:
                    checkpoint(state(before + now - t0))
                    last_ck = now
                lr = peak * min(1.0, (steps + 1) / warm) * 0.5 * (1 + math.cos(math.pi * frac))
                for pg in opt.param_groups:
                    pg["lr"] = lr
                x, y = corpus.batch(g["batch"], g["ctx"])
                with autocast(g, dev):
                    logits, aux = model(x)
                    loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1)) + 0.01 * aux
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                steps += 1
                if steps % 25 == 0 or steps == 1:
                    loss_v = loss.item()
                    if not math.isfinite(loss_v):
                        diverged = True
                        break
            _sync(dev)
    finally:
        if old_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
    if checkpoint and not diverged:
        checkpoint(state(before + meter.seconds))
    done = steps - (resume["steps"] if resume else 0)
    real = {
        "steps": steps,
        "seconds": meter.seconds,
        "total_seconds": before + meter.seconds,
        "step_seconds": meter.seconds / max(done, 1),
        "bytes_seen": done * g["batch"] * g["ctx"],
        "train_loss_bpb": loss_v / math.log(2) if math.isfinite(loss_v) else None,
        "joules": meter.joules if meter.available else None,
        "peak_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
        "diverged": diverged,
    }
    return model, real
