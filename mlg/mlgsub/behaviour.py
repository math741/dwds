"""The behaviour suite (data/trust/behaviour/suite.tsv): the whole MLG, as the
headset runs it, put through what it must do before anything is published.

Every "must" line has to pass. "quality" lines go through a Resident Core and
are scored: an answer passes when it routes to an expected organ, matches what
it must, avoids what it must not, and is not empty, looping or overlong.
"bare" lines are what a body with no organ from the lab yet must still do; they
run only on a bare mind (no core, no knowledge, no router).
"""
import os
import re
import tempfile

from .mind import Mind

SUITE = os.path.join(os.path.dirname(__file__), "..", "data", "trust", "behaviour", "suite.tsv")
QUALITY_SHARE = 0.6  # of quality lines that must pass for a brain to be published


def load(path=SUITE):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        gate, text, organs, must, must_not = (line.rstrip("\n").split("\t") + [""] * 5)[:5]
        rows.append({"gate": gate, "input": text, "organs": organs.split("|"), "must": must, "must_not": must_not})
    return rows


def run(core=None, know=None, router=None, log=print, rows=None):
    """Runs the suite on a fresh mind; returns (results, report)."""
    memory = tempfile.mktemp(suffix=".jsonl")
    m = Mind(core=core, know=know, router=router, memory=memory)
    m.body(device="Meta Quest 3S", cores=6, battery=80, temperature=30.0, threads=4, bytes_per_s=120)
    m.clock()
    m.lab("O laboratório está ligado. Treinando agora.")
    results = []
    bare = core is None and know is None and router is None
    for r in rows or load():
        if (r["gate"] == "quality" and core is None) or (r["gate"] == "bare") != bare:
            continue
        organ, reply = m.ask(r["input"])
        why = []
        if organ not in r["organs"]:
            why.append(f"organ {organ}")
        if r["must"] and not re.search(r["must"], reply, re.I):
            why.append(f"missing /{r['must']}/")
        if r["must_not"] and re.search(r["must_not"], reply, re.I):
            why.append(f"said /{r['must_not']}/")
        if not reply.strip() or len(reply) > 600:
            why.append("empty or overlong")
        results.append({**r, "organ": organ, "reply": reply, "ok": not why, "why": "; ".join(why)})
    m.close()
    if os.path.exists(memory):
        os.remove(memory)
    must = [x for x in results if x["gate"] in ("must", "bare")]
    quality = [x for x in results if x["gate"] == "quality"]
    report = {"must_passed": sum(x["ok"] for x in must), "must": len(must),
              "quality_passed": sum(x["ok"] for x in quality), "quality": len(quality)}
    report["quality_share"] = report["quality_passed"] / report["quality"] if quality else None
    report["passed"] = report["must_passed"] == report["must"] and (
        report["quality_share"] is None or report["quality_share"] >= QUALITY_SHARE)
    for x in results:
        log(f"  {'ok ' if x['ok'] else 'NO '} [{x['gate']:7s}] {x['input'][:38]:38s} -> {x['organ']:10s} "
            f"{x['reply'][:70]!r}" + (f"  <- {x['why']}" if not x["ok"] else ""))
    log(f"[behaviour] must {report['must_passed']}/{report['must']}"
        + (f", quality {report['quality_passed']}/{report['quality']}" if quality else "")
        + (" — passed" if report["passed"] else " — FAILED"))
    return results, report
