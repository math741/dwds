"""Carve the sealed dialog gate out of the MLG dialog corpus (run once; output is committed).

The language Q root is 2003 bytes, too small to tell two brains apart within its own
tolerance. The dialog gate adds held-out conversation: whole answer groups (an answer and
every question that leads to it) chosen at random, so no paraphrase of a gate answer is
ever seen in training. Corpus removes these lines from the training text at load time.

    python tools/carve_gate.py        # writes data/trust/gate/dialog.txt and data/trust/gate.lock
"""
import collections
import hashlib
import os
import random

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
GROUPS, SEED = 60, 741


def main():
    lines = open(os.path.join(ROOT, "core/data/corpus-dialog.txt"), encoding="utf-8").read().split("\n")
    pairs = {(lines[i], lines[i + 1]) for i in range(len(lines) - 1)
             if lines[i].startswith("U:") and lines[i + 1].startswith("M:")}
    by_q, by_a = collections.defaultdict(set), collections.defaultdict(set)
    for q, a in pairs:
        by_q[q].add(a)
        by_a[a].add(q)
    # an answer whose questions lead nowhere else can leave training without orphaning anything
    clean = sorted(a for a, qs in by_a.items() if len(qs) <= 3 and all(len(by_q[q]) == 1 for q in qs))
    chosen = random.Random(SEED).sample(clean, GROUPS)
    out = "".join(f"{q}\n{a}\n" for a in chosen for q in sorted(by_a[a]))
    os.makedirs(os.path.join(ROOT, "trust/gate"), exist_ok=True)
    with open(os.path.join(ROOT, "trust/gate/dialog.txt"), "w", encoding="utf-8") as fh:
        fh.write(out)
    sha = hashlib.sha256(out.encode()).hexdigest()
    with open(os.path.join(ROOT, "trust/gate.lock"), "w") as fh:
        fh.write(f"{sha}  dialog.txt\n")
    print(f"{GROUPS} answer groups, {len(out.encode())} bytes, sha256 {sha}")


if __name__ == "__main__":
    main()
