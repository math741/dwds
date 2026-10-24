"""The organs a headset can carry before any Resident Core: the router and the
knowledge index (the MLG's own corpus and Portuguese Wikipedia). Nothing here
needs a GPU, so CI builds them on every change and publishes the release
`mlg-organs`; the app takes them until the lab publishes a brain of its own
(`mlg-brain`), so a fresh headset answers from Wikipedia offline on day one.

    python -m mlgsub.organs build OUT --wiki-mb 512 --cache ~/.cache/mlg
    python -m mlgsub.organs web OUT SITE/organs --brain BRAIN   # the same, for the web page
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


def for_web(src, out, brain=None, part_mb=48, log=print):
    """The organs as the web page fetches them: in parts of `part_mb` MiB (static hosts
    limit file sizes), with the published brain's core when there is one (`brain`: a
    directory holding mlg-brain's manifest.json and core.mlgb)."""
    os.makedirs(out, exist_ok=True)
    manifest = json.load(open(os.path.join(src, "manifest.json")))
    if brain and os.path.exists(os.path.join(brain, "manifest.json")) and os.path.exists(os.path.join(brain, "core.mlgb")):
        b = json.load(open(os.path.join(brain, "manifest.json")))
        if export.sha256(os.path.join(brain, "core.mlgb")) == b["files"]["core.mlgb"]["sha256"]:
            manifest["files"]["core.mlgb"] = b["files"]["core.mlgb"]
            manifest.update(version=b["version"], core=b["core"], behaviour=b["behaviour"], kind="brain+organs")
            src_of = {"core.mlgb": brain}
            log(f"[web] Resident Core {b['core']['brain']} do mlg-brain ({b['version']})")
        else:
            src_of = {}
            log("[web] o core do mlg-brain não confere com o manifest dele: fica de fora")
    else:
        src_of = {}
    step = part_mb * 2**20
    for name, f in manifest["files"].items():
        with open(os.path.join(src_of.get(name, src), name), "rb") as fh:
            data = fh.read()
        f["parts"] = (len(data) + step - 1) // step
        for i in range(f["parts"]):
            with open(os.path.join(out, f"{name}.{i:03d}"), "wb") as o:
                o.write(data[i * step:(i + 1) * step])
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    parts = ", ".join(f"{n} ({f['parts']} partes)" for n, f in manifest["files"].items())
    log(f"[web] {parts} -> {out}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="router and knowledge index, through the behaviour suite")
    b.add_argument("out")
    b.add_argument("--wiki-mb", type=int, default=512)
    b.add_argument("--cache", default=os.path.expanduser("~/.cache/mlg"))
    w = sub.add_parser("web", help="the organs in parts for the web page, with the published core if any")
    w.add_argument("src")
    w.add_argument("out")
    w.add_argument("--brain")
    a = ap.parse_args()
    if a.cmd == "build":
        m = build(a.out, a.wiki_mb, a.cache)
        raise SystemExit(0 if m["behaviour"]["passed"] else 1)
    for_web(a.src, a.out, a.brain)
