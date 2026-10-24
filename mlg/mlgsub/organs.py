"""The organs a headset can carry before any Resident Core: the router and the
knowledge index (the MLG's own corpus and Portuguese Wikipedia). Nothing here
needs a GPU, so CI builds them on every change and publishes the release
`mlg-organs`; the app takes them until the lab publishes a brain of its own
(`mlg-brain`), so a fresh headset answers from Wikipedia offline on day one.

    python -m mlgsub.organs OUT --wiki-mb 512 --cache ~/.cache/mlg
"""
import argparse
import json
import os
import time

from . import behaviour, brain as B, export, know, router as R

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def build(out, wiki_mb, cache, log=print):
    os.makedirs(out, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    tsv = know.download_wikipedia(os.path.join(cache, f"wiki-pt-{wiki_mb}.tsv"), wiki_mb, log=log)
    wiki = list(know.wiki_passages(tsv, min_article=know.STUB_BYTES))
    passages = know.build(ROOT, os.path.join(out, "know.mlgk"), B.Corpus.TRAIN, set(), wiki, log=log)
    acc = R.build(ROOT, os.path.join(out, "router.mlgr"), log=log)
    _, report = behaviour.run(know=os.path.join(out, "know.mlgk"), router=os.path.join(out, "router.mlgr"),
                              wiki=True, log=log)
    names = ("know.mlgk", "router.mlgr")
    manifest = {
        "version": "organs-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
        "kind": "organs",
        "files": {n: {"sha256": export.sha256(os.path.join(out, n)), "bytes": os.path.getsize(os.path.join(out, n))}
                  for n in names},
        "router": {"intents": R.INTENTS, "held_out_accuracy": acc},
        "knowledge": {"passages": len(passages), "wikipedia_passages": len(wiki), "wikipedia_mib": wiki_mb},
        "behaviour": report,
    }
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    log(f"[organs] knowledge {manifest['files']['know.mlgk']['bytes'] / 2**20:.0f} MiB ({len(wiki):,} Wikipedia "
        f"passages), router {acc:.1%} held out, behaviour {'passed' if report['passed'] else 'FAILED'} -> {out}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--wiki-mb", type=int, default=512)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/mlg"))
    a = ap.parse_args()
    m = build(a.out, a.wiki_mb, a.cache)
    raise SystemExit(0 if m["behaviour"]["passed"] else 1)
