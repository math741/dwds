"""End to end without a notebook:  python -m mlgsub [--quick] [--extra wiki.txt]"""
import argparse
import os

from . import brain, explain
from .body import awaken
from .evolve import evolve
from .substrate import connect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "..", "data"),
                    help="folder holding core/data and trust/ (default: the bundled copy)")
    ap.add_argument("--extra", nargs="*", default=[], help="more training text files")
    ap.add_argument("--quick", action="store_true", help="tiny budgets, for a smoke test")
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=90)
    a = ap.parse_args()
    if a.quick:
        a.population, a.generations, a.seconds = 3, 2, 4

    sub = connect(fresh=True)
    print(f"substrate: {sub.mode}")
    body = awaken(sub, quick=a.quick)
    print("\n".join(explain.body_report(sub, body.id)))
    corpus = brain.Corpus(os.path.abspath(a.root), body.device, extra=a.extra,
                          max_train_bytes=200_000 if a.quick else None)
    print(f"corpus: {len(corpus.train)} train bytes, {len(corpus.dev)} dev, Q root sealed={corpus.qroot_sealed}, "
          f"{corpus.removed_lines} evaluation lines removed from training")
    f = evolve(sub, body, corpus, population=a.population, generations=a.generations, seconds=a.seconds)
    print("\n".join(explain.why_champion(sub, body.id)))
    if f.champion:
        print(brain.generate(f.champion["model"], f.champion["g"], "U: o que é a mlg\nM:", body.device))
    ok, n = sub.verify_events()
    print(f"event chain: {'INTACT' if ok else 'BROKEN at ' + str(n)} ({n} events)")


if __name__ == "__main__":
    main()
