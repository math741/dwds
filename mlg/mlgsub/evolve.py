"""The Candidate Furnace for brains (manifesto XI-XV, XXIII-XXVII, LXIV-LXVIII).

Every genome goes SANDBOX -> TRAIN -> ATTACK -> BENCHMARK -> COMPARE and ends
COMMITTED, ROLLED BACK or KILLED, with a Decision node that names the
evidence it rests on. Selection reads dev; the frozen Q root is only the
promotion gate, so the search never grades its own homework (XIV).
"""
import gc
import math
import os
import random
import time

import torch

from . import brain as B

QROOT_TOLERANCE = 0.002  # trust/qroot/policy.json: language_tolerance
TOLERANCE_BPB = 8 * QROOT_TOLERANCE  # the same tolerance in bits per byte (Q = 1 - bpb / 8)
INHERITED_LR = 0.5  # peak rate of a brain that inherited trained weights, relative to its gene


def _pareto(rows, keys):
    front = []
    for r in rows:
        if not any(all(o[k] <= r[k] for k in keys) and any(o[k] < r[k] for k in keys)
                   for o in rows if o is not r):
            front.append(r)
    return front


class Furnace:
    def __init__(self, sub, lab, corpus, deploy=None, seconds=60, log=print):
        self.sub, self.lab, self.corpus, self.log = sub, lab, corpus, log
        self.deploy = deploy  # body the brain must live in, if not the lab
        self.home = deploy or lab
        self.seconds = seconds
        self.n_decisions = 0
        self.champion = None  # dict of the committed brain for self.home
        self.results = []
        self.states = {}  # genome -> (dev bpb, weights): the best brains, for inheritance
        self.keep = 4
        self.lineage = {}  # genome -> (seconds and bytes trained by it and its ancestors, inherited generations)
        self.save_dir = os.environ.get("MLG_CHAMPIONS", "champions")  # every COMMIT is saved here
        self.on_step = None    # train() hook: the live link answers the headset between steps
        self.on_commit = None  # called with the new champion's row
        self.seconds_override = None  # maturation trains one brain for hours
        self._suites()

    # ---- frozen rulers -------------------------------------------------
    def _suites(self):
        s, c = self.sub, self.corpus
        s.obj("Suite", "suite:dev", origin="core/data/valid*.txt", epistemic="KNOWN",
              role="selection", bytes=len(c.dev))
        s.obj("Suite", "suite:qroot-language", origin="trust/qroot/language.txt",
              epistemic="KNOWN", role="frozen promotion gate", bytes=len(c.qroot),
              sha256=c.qroot_hash, sealed=c.qroot_sealed,
              formula="max(0, 1 - bits_per_byte / 8)")
        s.obj("Suite", "suite:gate-dialog", origin="trust/gate/dialog.txt", epistemic="KNOWN",
              role="sealed promotion gate: held-out conversation", bytes=len(c.gate),
              sha256=c.gate_hash, sealed=c.gate_sealed)
        s.obj("Dataset", "dataset:mlg-corpus", origin="core/data/corpus*.txt",
              epistemic="KNOWN", bytes=len(c.train) - sum(n for _, n, _ in c.extra),
              training_mix_sha256=c.train_hash, removed_eval_lines=c.removed_lines)
        self.datasets = ["dataset:mlg-corpus"]
        for path, n, sha in c.extra:
            did = "dataset:" + path.replace("\\", "/").rsplit("/", 1)[-1]
            s.obj("Dataset", did, origin=path, epistemic="KNOWN", bytes=n, sha256=sha)
            self.datasets.append(did)
        if not c.qroot_sealed:
            s.event("TRUST_FAIL", "suite:qroot-language", reason="hash differs from qroot.lock")
            raise RuntimeError("Q root does not match trust/qroot.lock; refusing to evaluate")
        if not c.gate_sealed:
            s.event("TRUST_FAIL", "suite:gate-dialog", reason="hash differs from gate.lock")
            raise RuntimeError("dialog gate does not match trust/gate.lock; refusing to evaluate")
        s.event("QROOT_SEALED", "suite:qroot-language", sha256=c.qroot_hash,
                removed_train_lines=c.removed_lines)

    def _decide(self, verdict, stage, brain_or_genome, reason, evidence=(), **props):
        self.n_decisions += 1
        did = f"decision:{self.home.id}:{self.n_decisions:04d}"
        self.sub.obj("Decision", did, origin="furnace", epistemic="OBSERVED",
                     verdict=verdict, stage=stage, reason=reason, body=self.home.id, **props)
        self.sub.link(did, "CHOSE" if verdict in ("COMMIT", "RESTORE") else "KILLED", brain_or_genome)
        for ev in evidence:
            self.sub.link(did, "JUSTIFIED_BY", ev)
        self.sub.event(verdict, brain_or_genome, stage=stage, reason=reason, decision=did)
        return did

    def _evaluate(self, bid, suite, bpb, **props):
        eid = f"eval:{bid}:{suite.split(':', 1)[1]}"
        self.sub.obj("Evaluation", eid, origin="furnace", epistemic="MEASURED",
                     bits_per_byte=bpb, q_language=B.q_language(bpb), **props)
        self.sub.link(bid, "EVALUATED_BY", eid)
        self.sub.link(eid, "ON_SUITE", suite)
        return eid

    # ---- one candidate through the furnace -----------------------------
    def run(self, g, generation, parents=(), mutated=(), seed=0, inheritance=None, train_kw=None):
        """`inheritance`: (weights, mode, parent brain id, parent genome id);
        `train_kw`: extra arguments for brain.train (maturation resumes with them)."""
        s, lab = self.sub, self.lab
        gid = B.genome_id(g)
        pred = B.predict(g, lab, self.deploy)
        s.obj("Genome", gid, origin="evolve", epistemic="EXPERIMENTAL", generation=generation,
              home=self.home.id, inherited=inheritance[1] if inheritance else None,
              **g, **{f"pred_{k}": v for k, v in pred.items()})
        for p in parents:
            s.link(p, "PARENT_OF", gid, mutated=list(mutated))
        if inheritance:
            s.link(gid, "INHERITED_FROM", inheritance[2], mode=inheritance[1])
        tag = (f"g{generation} {gid[7:13]} d{g['d_model']} L{g['n_layers']} E{g['n_experts']}k{g['top_k']} "
               f"{g['dtype']}" + (f" <{inheritance[1]}>" if inheritance else ""))
        row = {"genome": gid, "g": g, "generation": generation, "params": pred["params"]}

        # SANDBOX: never execute what prediction says cannot work (XXII)
        if not pred["fits_lab"]:
            self._decide("KILL", "SANDBOX", gid, "predicted training memory exceeds the lab body",
                         [lab.traits["usable_memory"]], predicted_bytes=pred["train_bytes"])
            self.log(f"  x {tag}: killed before running, needs {pred['train_bytes'] / 2**30:.1f} GiB")
            return {**row, "status": "KILLED", "stage": "SANDBOX"}
        if self.deploy is not None and not pred["fits_deploy"]:
            why = (f"{pred['deploy_bytes'] / 2**20:.0f} MiB, {pred['deploy_latency_ms']:.1f} ms/byte, "
                   f"first byte after {pred['deploy_prefill_s']:.1f} s")
            self._decide("KILL", "SANDBOX", gid, f"would not live in {self.deploy.id}: {why}",
                         [self.deploy.traits[t] for t in ("brain_memory_budget", "latency_budget",
                                                          "prefill_budget") if t in self.deploy.traits])
            self.log(f"  x {tag}: killed, cannot live in {self.deploy.name} ({why})")
            return {**row, "status": "KILLED", "stage": "SANDBOX"}

        # TRAIN in the lab. Running out of memory means the self-model lied:
        # the candidate dies and the memory model becomes more pessimistic.
        oom = False
        try:
            model, real = B.train(g, lab, self.corpus, self.seconds_override or self.seconds, seed=seed,
                                  init_state=inheritance[0] if inheritance else None,
                                  lr_scale=INHERITED_LR if inheritance else 1.0, on_step=self.on_step,
                                  **(train_kw or {}))
        except torch.cuda.OutOfMemoryError:
            oom = True  # handled outside, once the traceback has released the tensors
        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            lab.mem_factor *= 1.5
            tid = f"trait:{lab.id}:self_model_memory"
            s.obj("Trait", tid, origin="furnace.self_model", epistemic="MEASURED",
                  name="self_model_memory", value=lab.mem_factor, unit="ratio",
                  last_prediction_error="out of memory")
            s.link(lab.id, "HAS_TRAIT", tid)
            lab.traits["self_model_memory"] = tid
            self._decide("KILL", "TRAIN", gid, "out of memory: predicted cost was wrong",
                         [lab.traits["usable_memory"], tid], predicted_bytes=pred["train_bytes"])
            self.log(f"  x {tag}: out of memory; my memory model was too optimistic, correcting it")
            return {**row, "status": "KILLED", "stage": "TRAIN"}
        bid = f"brain:{gid[7:]}@{self.home.id[5:]}"
        # an inheriting brain carries all the training of its ancestors
        secs, seen, depth = self.lineage.get(inheritance[3], (0.0, 0, 0)) if inheritance else (0.0, 0, 0)
        self.lineage[gid] = (secs + real["seconds"], seen + real["bytes_seen"],
                             depth + (1 if inheritance else 0))
        s.obj("Brain", bid, origin="furnace.train", epistemic="TESTED",
              **{k: v for k, v in real.items() if v is not None},
              params=pred["params"], active_params=pred["active_params"],
              inherited=inheritance[1] if inheritance else None,
              lineage_seconds=self.lineage[gid][0], lineage_bytes=self.lineage[gid][1],
              inherited_generations=self.lineage[gid][2])
        s.link(gid, "EXPRESSES", bid)
        s.link(bid, "TRAINED_ON", lab.id)
        for ds in self.datasets:
            s.link(bid, "FED_BY", ds)
        s.link(bid, "LIVES_IN", self.home.id)
        dtype_trait = lab.traits.get(f"format_{g['dtype']}")
        if dtype_trait:
            s.link(bid, "SHAPED_BY", dtype_trait, why="number format chosen by the genome")
        s.link(bid, "SHAPED_BY", lab.traits["usable_memory"], why="memory bound for training")
        self._learn_self(bid, pred, real)

        # ATTACK: does it really model language, or did it just survive?
        if real["diverged"]:
            self._decide("KILL", "ATTACK", bid, "training diverged (non-finite loss)",
                         [dtype_trait] if dtype_trait else [])
            self.log(f"  x {tag}: diverged in {g['dtype']}")
            return {**row, "brain": bid, "status": "KILLED", "stage": "ATTACK"}
        dev_bpb = B.bits_per_byte(model, g, self.corpus.dev, lab.device)
        noise_bpb = B.bits_per_byte(model, g, self.corpus.shuffled_dev(), lab.device)
        e_dev = self._evaluate(bid, "suite:dev", dev_bpb, shuffled_bpb=noise_bpb)
        if not math.isfinite(dev_bpb) or dev_bpb > noise_bpb - 0.25:
            self._decide("KILL", "ATTACK", bid,
                         "cannot tell language from shuffled bytes", [e_dev])
            self.log(f"  x {tag}: dev {dev_bpb:.3f} vs shuffled {noise_bpb:.3f}, learned no structure")
            return {**row, "brain": bid, "status": "KILLED", "stage": "ATTACK", "dev_bpb": dev_bpb}
        gap = dev_bpb - (real["train_loss_bpb"] or dev_bpb)

        # BENCHMARK
        lat = B.latency_ms(model, g, lab.device)
        s.obj("Brain", bid, origin="furnace.benchmark", epistemic="TESTED",
              lab_latency_ms=lat, dev_bpb=dev_bpb, memorization_gap=gap,
              home_latency_ms=pred.get("deploy_latency_ms", lat),
              home_latency_level="PREDICTED" if self.deploy else "MEASURED")
        cost = real["joules"] if real["joules"] is not None else real["seconds"]
        row.update(brain=bid, status="SURVIVED", dev_bpb=dev_bpb, gap=gap, model=model,
                   lineage_bytes=self.lineage[gid][1], lineage_seconds=self.lineage[gid][0],
                   joules=real["joules"], seconds=real["seconds"], cost=cost,
                   latency_ms=pred.get("deploy_latency_ms", lat), eval_dev=e_dev)
        self._remember(gid, model, dev_bpb)
        self.log(f"  + {tag}: dev {dev_bpb:.3f} bpb, {pred['params'] / 1e6:6.2f}M params, "
                 f"{real['steps']} steps" + (f", {real['joules']:.0f} J" if real['joules'] else "")
                 + (f", gap {gap:+.2f}" if gap > 0.15 else "")
                 + (f", carries {self.lineage[gid][0]:.0f} s of ancestral training"
                    if inheritance else ""))

        # COMPARE against the champion; the Q root decides promotion (XII)
        self._compare(row, tag)
        return row

    def adopt(self, path, log=print):
        """Bring back a champion saved by an earlier session (its checkpoint),
        re-measured here on the same sealed suites before it is trusted."""
        model, g = B.load_brain(path, self.lab.device)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        gid, dev = B.genome_id(g), self.lab.device
        bid = ck.get("brain") or f"brain:{gid[7:]}@{self.home.id[5:]}"
        self.sub.obj("Genome", gid, origin="furnace.adopt", epistemic="EXPERIMENTAL", home=self.home.id, **g)
        self.sub.obj("Brain", bid, origin="furnace.adopt", epistemic="TESTED", checkpoint=path,
                     params=B.params(g)[0])
        self.sub.link(gid, "EXPRESSES", bid)
        dev_bpb = B.bits_per_byte(model, g, self.corpus.dev, dev)
        nats = {"gate": B.byte_nats(model, g, self.corpus.gate, dev).cpu(),
                "qroot": B.byte_nats(model, g, self.corpus.qroot, dev).cpu()}
        q_bpb, gate_bpb = (nats[k].mean().item() / math.log(2) for k in ("qroot", "gate"))
        row = {"genome": gid, "g": g, "generation": -1, "params": B.params(g)[0], "brain": bid,
               "status": "SURVIVED", "dev_bpb": dev_bpb, "gap": 0.0, "model": model, "lineage_bytes": 0,
               "cost": 0.0, "latency_ms": 0.0, "joules": None, "seconds": 0.0, "nats": nats,
               "qroot_bpb": q_bpb, "q": B.q_language(q_bpb), "gate_bpb": gate_bpb, "checkpoint": path}
        row["eval_dev"] = self._evaluate(bid, "suite:dev", dev_bpb)
        row["eval_q"] = self._evaluate(bid, "suite:qroot-language", q_bpb)
        row["eval_gate"] = self._evaluate(bid, "suite:gate-dialog", gate_bpb)
        row["decision"] = self._decide("RESTORE", "COMPARE", bid, f"champion of an earlier session, re-measured: "
                                       f"dialog gate {gate_bpb:.3f} bpb, Q root {q_bpb:.3f} bpb",
                                       [row["eval_dev"], row["eval_gate"], row["eval_q"]])
        self.champion = row
        self.results.append(row)
        self._remember(gid, model, dev_bpb)
        if self.on_commit:
            self.on_commit(row)
        log(f"[restore] {bid}: dev {dev_bpb:.3f}, dialog gate {gate_bpb:.3f}, Q root {q_bpb:.3f} bpb")
        return row

    def _remember(self, gid, model, dev_bpb):
        """Keep the weights of the best few brains so children can inherit them."""
        worst = max(self.states.items(), key=lambda kv: kv[1][0]) if self.states else None
        if len(self.states) < self.keep or dev_bpb < worst[1][0]:
            self.states[gid] = (dev_bpb, B.snapshot(model))
            if len(self.states) > self.keep:
                del self.states[worst[0]]

    def _compare(self, row, tag):
        """Promotion is decided on sealed suites the search never selects on:
        the dialog gate must improve, with 95% confidence and by more than the
        Q root tolerance, and the language Q root must not get significantly
        worse beyond it. Both brains are scored on exactly the same bytes."""
        bid, model, g = row["brain"], row["model"], row["g"]
        champ = self.champion
        if champ is not None and row["dev_bpb"] >= champ["dev_bpb"]:
            row["model"] = None  # keep only champions in memory
            return
        dev = self.lab.device
        nats = {"gate": B.byte_nats(model, g, self.corpus.gate, dev).cpu(),
                "qroot": B.byte_nats(model, g, self.corpus.qroot, dev).cpu()}
        q_bpb, gate_bpb = (nats[k].mean().item() / math.log(2) for k in ("qroot", "gate"))
        e_q = self._evaluate(bid, "suite:qroot-language", q_bpb)
        e_g = self._evaluate(bid, "suite:gate-dialog", gate_bpb)
        row.update(qroot_bpb=q_bpb, q=B.q_language(q_bpb), gate_bpb=gate_bpb,
                   eval_q=e_q, eval_gate=e_g, nats=nats)
        if champ is None:
            did = self._decide("COMMIT", "COMPARE", bid, "first brain to survive the furnace",
                               [row["eval_dev"], e_g, e_q], q_language=row["q"])
        else:
            gain, lo, hi = B.paired_gain(champ["nats"]["gate"], nats["gate"])
            qgain, qlo, qhi = B.paired_gain(champ["nats"]["qroot"], nats["qroot"])
            wins = lo > 0 and gain > TOLERANCE_BPB
            regresses = qhi < 0 and -qgain > TOLERANCE_BPB
            evidence = [row["eval_dev"], e_g, e_q, champ["eval_gate"], champ["eval_q"]]
            verdict = (f"dialog gate {champ['gate_bpb']:.3f} -> {gate_bpb:.3f} bpb "
                       f"(gain {gain:+.3f}, 95% [{lo:+.3f}, {hi:+.3f}]); "
                       f"Q root {champ['qroot_bpb']:.3f} -> {q_bpb:.3f} bpb (gain {qgain:+.3f}, "
                       f"95% [{qlo:+.3f}, {qhi:+.3f}]); tolerance {TOLERANCE_BPB:.3f}")
            if wins and not regresses:
                self.sub.link(bid, "CHALLENGED", champ["brain"], won=True)
                self.sub.link(bid, "SUPERSEDES", champ["brain"])
                did = self._decide("COMMIT", "COMPARE", bid, verdict, evidence, q_language=row["q"])
            else:
                self.sub.link(bid, "CHALLENGED", champ["brain"], won=False)
                why = ("the language Q root got worse" if regresses else
                       "the sealed gate did not improve" if gain <= 0 else
                       "the gain on the sealed gate is not sure" if lo <= 0 else
                       "the gain on the sealed gate is within the Q root tolerance")
                self._decide("ROLLBACK", "COMPARE", bid, f"better on dev, but {why}: {verdict}", evidence)
                self.log(f"    rollback: dev improved, {why} (gate {gain:+.3f} bpb, "
                         f"95% [{lo:+.3f}, {hi:+.3f}])")
                row["model"] = row["nats"] = None
                return
        row["decision"] = did
        # a committed brain is never lost: its weights go to disk with its genome
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, bid.replace(":", "_").replace("@", "_") + ".pt")
        torch.save({"genome": g, "state": B.snapshot(model), "brain": bid, "q_language": row["q"]}, path)
        self.sub.obj("Brain", bid, origin="furnace.commit", epistemic="TESTED", checkpoint=path)
        row["checkpoint"] = path
        if champ is not None:
            champ["model"] = champ["nats"] = None
        self.champion = row
        if self.on_commit:
            self.on_commit(row)
        self.log(f"    COMMIT {bid}: dialog gate {gate_bpb:.3f} bpb, Q_LANGUAGE {row['q']:.4f} "
                 f"(frozen Q root {q_bpb:.3f} bpb)")

    def _learn_self(self, bid, pred, real):
        """Prediction vs reality corrects the self-model (XXI)."""
        lab, s = self.lab, self.sub
        if real["steps"] < 5:
            return
        ratio = pred["step_seconds"] / real["step_seconds"]  # <1: I was too optimistic
        old = lab.efficiency
        lab.efficiency = math.exp(0.7 * math.log(old) + 0.3 * math.log(max(1e-4, old * ratio)))
        tid = lab.traits["self_model_efficiency"]
        s.obj("Trait", tid, origin="furnace.self_model", epistemic="MEASURED",
              name="self_model_efficiency", value=lab.efficiency, unit="ratio",
              last_prediction_error=ratio)
        s.link(tid, "PROVEN_BY", bid)
        mem_err = (real["peak_bytes"] / pred["train_bytes"]) if real.get("peak_bytes") else None
        if mem_err:
            lab.mem_factor = math.exp(0.7 * math.log(lab.mem_factor)
                                      + 0.3 * math.log(lab.mem_factor * mem_err))
            s.obj("Trait", f"trait:{lab.id}:self_model_memory", origin="furnace.self_model",
                  epistemic="MEASURED", name="self_model_memory", value=lab.mem_factor,
                  unit="ratio", last_prediction_error=mem_err)
            s.link(lab.id, "HAS_TRAIT", f"trait:{lab.id}:self_model_memory")
            s.link(f"trait:{lab.id}:self_model_memory", "PROVEN_BY", bid)
            lab.traits["self_model_memory"] = f"trait:{lab.id}:self_model_memory"
        s.obj("Brain", bid, origin="furnace.self_model", epistemic="TESTED",
              predicted_step_seconds=pred["step_seconds"], step_time_error=1 / ratio,
              memory_error=mem_err)


def evolve(sub, lab, corpus, deploy=None, population=6, generations=3, seconds=60,
           seed=0, min_gain=0.01, patience=2, inherit_rate=0.7, log=print, furnace=None):
    """Grow brains for `deploy` (default: the lab itself). Returns the furnace,
    whose champion is the committed brain and whose results hold every run.
    `inherit_rate`: share of children that keep their parent's weights."""
    rng = random.Random(seed)
    stalled = 0
    home = deploy or lab
    dtypes = [d for d in ("fp32", "tf32", "bf16", "fp16")
              if d in lab.dtypes and not lab.dtypes[d].get("unsupported")]
    if lab.device.type == "cpu":
        dtypes = ["fp32"]  # bf16 on CPU is emulated: slower, not a choice worth paying for
    f = furnace or Furnace(sub, lab, corpus, deploy, seconds, log)
    # passing a furnace continues its evolution: same champion, parents and
    # inheritable weights, generations numbered after the last one
    pool = list(f.results)
    start = max((r["generation"] for r in pool), default=-1) + 1
    if pool:
        f.seconds = seconds
        rng = random.Random(seed + len(pool))
    best_hist = [min((r["dev_bpb"] for r in pool if r["status"] == "SURVIVED"), default=float("inf"))] \
        if pool else []
    sub.event("EVOLUTION_BEGIN", home.id, population=population, generations=generations,
              seconds=seconds, dtypes=dtypes, first_generation=start)
    for gen in range(start, start + generations):
        if not pool:
            ancestor = dict(B.ANCESTOR, dtype="fp32")
            batch = [(ancestor, (), ())] + [(B.random_genome(rng, dtypes), (), ())
                                             for _ in range(population - 1)]
        else:
            ranked = sorted([r for r in pool if r["status"] == "SURVIVED"], key=lambda r: r["dev_bpb"])
            parents = ranked[: max(2, len(ranked) // 2)] or [{"g": B.ANCESTOR, "genome": None}]
            batch = []
            seen = {r["genome"] for r in pool}
            while len(batch) < population:
                if len(parents) > 1 and rng.random() < 0.3:
                    a, b = rng.sample(parents, 2)
                    child = B.crossover(a["g"], b["g"], rng)
                    child, changed = B.mutate(child, rng, dtypes, n=1)
                    ps = (a["genome"], b["genome"])
                else:
                    a = parents[min(int(rng.expovariate(1.0)), len(parents) - 1)]
                    if a.get("genome") in f.states and rng.random() < inherit_rate:
                        # a child that keeps its parent's mind: change only what
                        # inheritance survives, and prefer growing
                        child, changed = B.mutate(a["g"], rng, dtypes, genes=B.INHERITABLE, grow=True)
                    else:
                        child, changed = B.mutate(a["g"], rng, dtypes)
                    ps = (a["genome"],)
                cid = B.genome_id(child)
                if cid in seen:
                    continue
                seen.add(cid)
                batch.append((child, tuple(p for p in ps if p), changed, a))
        log(f"generation {gen} in {home.name or home.id}:")
        t = time.perf_counter()
        for item in batch:
            g, parents, changed = item[:3]
            inh = None
            if len(item) > 3 and item[3].get("genome") in f.states:
                a = item[3]
                weights, mode = B.inherit(g, a["g"], f.states[a["genome"]][1])
                if weights is not None:
                    inh = (weights, mode, a["brain"], a["genome"])
            pool.append(f.run(g, gen, parents, changed, seed=seed + len(pool), inheritance=inh))
            del inh
        alive = [r for r in pool if r["status"] == "SURVIVED"]
        best = min((r["dev_bpb"] for r in alive), default=float("inf"))
        best_hist.append(best)
        gen_cost = time.perf_counter() - t
        # SEARCH DEATH RULE (LXVIII): stop when expected gain no longer pays
        if len(best_hist) >= 2:
            gain = best_hist[-2] - best_hist[-1]
            stalled = stalled + 1 if gain < min_gain else 0
            if stalled >= patience:
                did = f"decision:{home.id}:search-death-{gen}"
                sub.obj("Decision", did, origin="evolve", epistemic="OBSERVED",
                        verdict="SEARCH_DEATH", stage="EVOLVE", body=home.id,
                        reason=f"{stalled} generation(s) gained under {min_gain} bpb "
                               f"(last {gain:.4f}); another costs ~{gen_cost:.0f} s",
                        expected_gain=max(gain, 0) / 2, expected_cost_s=gen_cost)
                sub.event("SEARCH_DEATH", home.id, gain=gain, cost_s=gen_cost)
                log(f"search death: {stalled} generation(s) without a {min_gain} bpb gain. Stopping.")
                break
    f.results = pool
    _data_wall(sub, home, pool, log)
    front = _pareto([r for r in pool if r["status"] == "SURVIVED"], ["dev_bpb", "cost", "latency_ms"])
    for r in front:
        sub.obj("Brain", r["brain"], origin="evolve.pareto", epistemic="TESTED", pareto=True)
    sub.event("EVOLUTION_END", home.id, champion=f.champion["brain"] if f.champion else None,
              candidates=len(pool), survivors=sum(r["status"] == "SURVIVED" for r in pool))
    return f


def _data_wall(sub, home, pool, log):
    """If bigger brains do worse although they saw as many bytes, and they
    memorize more, the limit is the data, not the body (LXIV: 20M beats 1T
    when Q says so). If they lost having seen fewer bytes, the training
    budget held them back, and that is recorded instead."""
    alive = sorted([r for r in pool if r["status"] == "SURVIVED"], key=lambda r: r["dev_bpb"])
    if len(alive) < 3:
        return
    best = alive[0]
    bigger = [r for r in alive[1:] if r["params"] > 2 * best["params"]]
    fed = [r for r in bigger if r["lineage_bytes"] >= best["lineage_bytes"] and r["gap"] > best["gap"] + 0.1]
    starved = [r for r in bigger if r["lineage_bytes"] < best["lineage_bytes"]]
    if len(fed) >= 2:
        tid = f"trait:dataset:mlg-corpus:capacity-wall@{home.id[5:]}"
        sub.obj("Trait", tid, origin="evolve.data_wall", epistemic="INFERRED", name="capacity_wall",
                value=best["params"], unit="params",
                meaning=f"{len(fed)} brains with >2x the champion's parameters saw at least as many bytes, "
                        "memorized more and scored worse on dev: past this size the corpus, not the body, "
                        "limits quality")
        sub.link("dataset:mlg-corpus", "HAS_TRAIT", tid)
        for r in [best] + fed:
            sub.link(tid, "PROVEN_BY", r["eval_dev"])
        sub.event("WALL_FOUND", "dataset:mlg-corpus", params=best["params"])
        log(f"[discovery] {len(fed)} brains more than twice the best one's size saw as much text, memorized "
            f"more and did worse. The wall is the data ({best['params'] / 1e6:.1f}M params), not the body.")
    elif len(starved) >= 2:
        tid = f"trait:{home.id}:budget-wall"
        sub.obj("Trait", tid, origin="evolve.data_wall", epistemic="INFERRED", name="budget_wall",
                value=best["lineage_bytes"], unit="bytes seen",
                meaning=f"{len(starved)} brains with >2x the best one's parameters lost after seeing fewer "
                        "bytes: the training budget, not the data, limited them")
        sub.link(home.id, "HAS_TRAIT", tid)
        for r in [best] + starved:
            sub.link(tid, "PROVEN_BY", r["eval_dev"])
        log(f"[discovery] {len(starved)} bigger brains lost, but they saw fewer bytes than the best one. "
            "The wall is the training budget, not the data: a longer maturation can afford a bigger brain.")
