"""The retrieval expert's memory of the world, built in the lab.

Passages come from the MLG's own corpus (each answer, titled by a question
that leads to it, and its prose) and from Portuguese Wikipedia, cut at
sentence boundaries. engine/mlg_cli indexes them into a .mlgk the headset
maps and searches with BM25. Lines held for evaluation (dev, gate, Q root)
never enter the index, so the sealed suites stay unseen.

It also writes the conversations the Resident Core learns to answer from:
a passage the retrieval expert found, then the question, then the answer,
the same layout the headset builds at run time:

    C: <passage>
    U: <question>
    M: <answer>
"""
import os
import random
import re
import subprocess

ENGINE = os.path.join(os.path.dirname(__file__), "..", "engine")
CLI = os.path.join(ENGINE, "mlg_cli")
PASSAGE = 600   # bytes, about what the headset puts in front of a question
CONTEXT = 420   # bytes of a passage kept in a C: line
_SENT = re.compile(r"(?<=[.!?])\s+")


def build_cli():
    subprocess.run(["make", "-s", "-C", ENGINE], check=True)
    return CLI


def _clean(s):
    return " ".join(s.replace("\t", " ").split())


def dialog_pairs(root, files, held=frozenset()):
    """(question, answer) pairs of the MLG corpus, without the U:/M: marks."""
    out = []
    for f in files:
        lines = open(os.path.join(root, f), encoding="utf-8", errors="replace").read().split("\n")
        for q, a in zip(lines, lines[1:]):
            if q.startswith("U:") and a.startswith("M:") and q.strip() not in held and a.strip() not in held:
                out.append((q[2:].strip(), a[2:].strip()))
    return out


def mlg_passages(root, files, held=frozenset()):
    by_answer = {}
    for q, a in dialog_pairs(root, files, held):
        by_answer.setdefault(a, q)
    for a, q in by_answer.items():
        yield q, a
    for f in files:
        for line in open(os.path.join(root, f), encoding="utf-8", errors="replace"):
            line = line.strip()
            if len(line) > 60 and not line.startswith(("U:", "M:")) and line not in held:
                yield "MLG", line


def chunks(text, size=PASSAGE):
    """Sentences packed into passages of about `size` bytes."""
    cur = ""
    for s in _SENT.split(text):
        if cur and len((cur + " " + s).encode()) > size:
            yield cur
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        yield cur


def wiki_passages(path, limit_bytes=None):
    """Articles as written by the notebook: one per line, title<TAB>text,
    paragraphs separated by two spaces."""
    seen = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            title, _, text = line.rstrip("\n").partition("\t")
            if not text:
                continue
            for para in text.split("  "):
                for p in chunks(para):
                    if len(p) > 80:
                        yield title, p
            seen += len(line)
            if limit_bytes and seen >= limit_bytes:
                break


def build(root, out, train_files, held, wiki=None, wiki_bytes=None, max_df=0.05, log=print):
    """Write passages.tsv next to `out`, index it, return the passages."""
    cli = build_cli()
    tsv = out + ".tsv"
    passages = []
    with open(tsv, "w", encoding="utf-8") as fh:
        for source in [mlg_passages(root, train_files, held)] + ([wiki_passages(wiki, wiki_bytes)] if wiki else []):
            for title, text in source:
                title, text = _clean(title), _clean(text)
                if text:
                    fh.write(f"{title}\t{text}\n")
                    passages.append((title, text))
    r = subprocess.run([cli, "index", tsv, out, str(max_df)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    log(f"[knowledge] {r.stdout.strip()}, {os.path.getsize(out) / 2**20:.1f} MiB")
    return passages


def search_many(index, queries, k=3):
    r = subprocess.run([CLI, "batch", index, str(k)], input="\n".join(q.replace("\n", " ") for q in queries) + "\n",
                       capture_output=True, text=True, check=True)
    rows = r.stdout.split("\n")
    return [[int(x) for x in row.split("\t") if x] for row in rows[:len(queries)]]


def _context(text):
    b = text.encode()[:CONTEXT]
    return b.decode("utf-8", "ignore").rsplit(" ", 1)[0] if len(text.encode()) > CONTEXT else text


QUESTIONS = ["o que é {t}", "o que é {t}?", "me fala sobre {t}", "quem foi {t}", "fale sobre {t}",
             "você sabe o que é {t}?", "me explica {t}", "{t}", "o que você sabe sobre {t}?"]


def grounded_dialogs(index, passages, root, train_files, held, out, wiki_titles=200000, seed=0, log=print):
    """Conversations where the answer rests on a retrieved passage.

    MLG corpus questions get the passages the expert really finds for them
    (sometimes the right one, sometimes not, as on the headset). Wikipedia
    articles become "what is X" questions answered by their first sentences,
    shown either their own passage or, now and then, an unrelated one,
    which teaches the core to say what the passage says and nothing else."""
    rng = random.Random(seed)
    pairs = dialog_pairs(root, train_files, held)
    found = search_many(index, [q for q, _ in pairs], k=2)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for (q, a), ids in zip(pairs, found):
            ctx = " ".join(_context(passages[i][1]) for i in ids[:1 if rng.random() < 0.6 else 2])
            fh.write((f"C: {ctx}\n" if ctx else "") + f"U: {q}\nM: {a}\n")
            n += 1
        firsts = {}
        for i, (title, text) in enumerate(passages):
            if title != "MLG" and title not in firsts and len(firsts) < wiki_titles:
                firsts[title] = i
        titles = list(firsts)
        for title in titles:
            i = firsts[title]
            text = passages[i][1]
            answer = " ".join(_SENT.split(text)[:2])
            if len(answer) < 40:
                continue
            q = rng.choice(QUESTIONS).format(t=title)
            if rng.random() < 0.12:  # an unrelated passage: the honest answer is that it does not say
                other = passages[firsts[rng.choice(titles)]][1]
                fh.write(f"C: {_context(other)}\nU: {q}\nM: Não encontrei isso no que eu sei agora.\n")
            else:
                fh.write(f"C: {_context(text)}\nU: {q}\nM: {answer}\n")
            n += 1
    log(f"[knowledge] {n} grounded conversations written for the Resident Core")
    return out
