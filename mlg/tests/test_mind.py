"""The MLG's mind (engine/mlg_mind.c, the headset's code) passes every "must"
line of the sealed behaviour suite, with a router trained here from scratch and
the knowledge index of its own corpus.

    python tests/test_mind.py
"""
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from mlgsub import behaviour, brain as B, know, router as R  # noqa: E402


def main():
    root = os.path.join(HERE, "..", "data")
    tmp = tempfile.mkdtemp()
    know.build(root, os.path.join(tmp, "know.mlgk"), B.Corpus.TRAIN, set(), [])
    qs = sorted({q for q, _ in know.dialog_pairs(root, ["core/data/corpus-dialog.txt"])})
    w, b, _ = R.train(R.examples(random.Random(0), per_intent=600, corpus_questions=qs))
    R.write(os.path.join(tmp, "router.mlgr"), w, b)
    _, report = behaviour.run(know=os.path.join(tmp, "know.mlgk"), router=os.path.join(tmp, "router.mlgr"))
    assert report["passed"], report
    # a headset before the lab has sent anything: still useful, never the same sentence twice
    _, report = behaviour.run()
    assert report["passed"], report
    from mlgsub.mind import Mind
    m = Mind()
    said = [m.ask(q)[1] for q in ("chora baby", "me fala de vulcões", "quem ganhou a copa")]
    assert len(set(said)) == 3 and all("não" in s.lower() for s in said), said
    m.close()
    print("the mind passes the behaviour suite, with its organs and bare")


if __name__ == "__main__":
    main()
