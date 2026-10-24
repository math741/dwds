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


MLG_TITLE = "§ "  # engine/mlg_mind.c answers these word for word when the question matches; not a word


def mlg_passages(root, files, held=frozenset()):
    """Each answer of the MLG corpus, titled by every question that leads to it."""
    by_answer = {}
    for q, a in dialog_pairs(root, files, held):
        qs = by_answer.setdefault(a, [])
        if q not in qs:
            qs.append(q)
    for a, qs in by_answer.items():
        yield MLG_TITLE + " | ".join(qs[:12]), a
    for f in files:
        for line in open(os.path.join(root, f), encoding="utf-8", errors="replace"):
            line = line.strip()
            if len(line) > 60 and not line.startswith(("U:", "M:")) and line not in held:
                yield MLG_TITLE.strip(), line  # the MLG's prose: no title words to weigh


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


# A Wikipedia paragraph is prose when it reads as sentences: it ends like one,
# has a dozen words, and is not a bibliography entry, a citation or a list item
# ("Vianna, Hélio. História do Brasil... 1994.", "Thalurania furcata - Beija-flor").
_CITATION = re.compile(r"ISBN|ISSN|Consultado em|Arquivado|\bdoi\b|\bpp?\. ?\d|\bed\. |Editora|Retrieved|"
                       r"^[A-ZÀ-Ý][^\s,.]+, [A-ZÀ-Ý][^.]{0,60}\. ")


_LIST_ITEM = re.compile(r"^\d{1,4}\s*[—–-]\s")  # "1481 — O maior dos três terremotos..." (date pages)


_CALENDAR = re.compile(r"^\d{1,2} de (janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|"
                       r"dezembro)$|^\d{1,4}( a\.C\.)?$|^Década de|^Século ")


_EMPTY_PARENS = re.compile(r"\s*\((?:\s*[;,]\s*)*\)")  # what is left of stripped pronunciations: "Marrocos (; ; )"


_LEFT_OPEN = re.compile(r"\(\s*[;,]\s*")  # "Japão (; oficialmente" -> "Japão (oficialmente"
_LONE_COMMA = re.compile(r"\s+([,;])")    # "oficialmente , ou" -> "oficialmente, ou"


def tidy(paragraph):
    return _LONE_COMMA.sub(r"\1", _LEFT_OPEN.sub("(", _EMPTY_PARENS.sub("", paragraph)))


def calendar(title):
    """Date, year and century pages: lists of events, not prose."""
    return bool(_CALENDAR.match(title))


def prose(paragraph):
    p = paragraph.strip()
    return (len(p.split()) >= 12 and p[-1] in '.!?"»' and not _CITATION.search(p) and not _LIST_ITEM.match(p))


STUB_BYTES = 1500  # Wikipedia articles shorter than this are stubs ("X é um município...")


def download_wikipedia(path, mb, log=print):
    """Portuguese Wikipedia, title<TAB>text per line (paragraphs split by two spaces),
    kept at `path` so later runs do not download it again."""
    if os.path.exists(path) and os.path.getsize(path) >= mb * 2**20 * 0.95:
        return path
    from datasets import load_dataset

    log(f"[feeding] streaming {mb} MiB of Portuguese Wikipedia")
    stream = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
    n = 0
    with open(path + ".part", "w", encoding="utf-8") as fh:
        for row in stream:
            title = " ".join(row["title"].split())
            paras = [" ".join(p.split()) for p in row["text"].split("\n") if len(p.strip()) > 40]
            if not paras:
                continue
            line = f"{title}\t{'  '.join(paras)}\n"
            fh.write(line)
            n += len(line.encode())
            if n >= mb * 2**20:
                break
    os.replace(path + ".part", path)
    return path


def wiki_passages(path, limit_bytes=None, min_article=0):
    """Articles as written by the notebook: one per line, title<TAB>text,
    paragraphs separated by two spaces. Articles shorter than `min_article`
    bytes (stubs) are skipped."""
    seen = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            title, _, text = line.rstrip("\n").partition("\t")
            if not text or len(text) < min_article or calendar(title):
                continue
            for para in text.split("  "):
                if not prose(para):
                    continue
                for p in chunks(tidy(para)):
                    if len(p) > 80:
                        yield title, p
            seen += len(line)
            if limit_bytes and seen >= limit_bytes:
                break


def build(root, out, train_files, held, wiki=(), max_df=0.05, log=print):
    """Index the MLG corpus (minus `held` lines) and `wiki` passages into `out`;
    returns the passages in index order (their ids)."""
    cli = build_cli()
    tsv = out + ".tsv"
    passages = []
    with open(tsv, "w", encoding="utf-8") as fh:
        for title, text in list(mlg_passages(root, train_files, held)) + list(wiki):
            title, text = _clean(title), _clean(text)
            if text:
                fh.write(f"{title}\t{text}\n")
                passages.append((title, text))
    r = subprocess.run([cli, "index", tsv, out, str(max_df)], capture_output=True, text=True)
    os.remove(tsv)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    log(f"[knowledge] {os.path.basename(out)}: {r.stdout.strip()}, {os.path.getsize(out) / 2**20:.1f} MiB")
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
            if not title.startswith(("MLG", MLG_TITLE.strip())) and title not in firsts and len(firsts) < wiki_titles:
                firsts[title] = i
        titles = list(firsts)
        for title in titles:
            i = firsts[title]
            text = passages[i][1]
            answer = " ".join(_SENT.split(text)[:2])
            if len(answer) < 40:
                continue
            q = rng.choice(QUESTIONS).format(t=title)
            if rng.random() < 0.08:  # an unrelated passage: the honest answer is that it does not say
                other = passages[firsts[rng.choice(titles)]][1]
                fh.write(f"C: {_context(other)}\nU: {q}\nM: Não encontrei isso no que eu sei agora.\n")
            else:
                fh.write(f"C: {_context(text)}\nU: {q}\nM: {answer}\n")
            n += 1
    log(f"[knowledge] {n} grounded conversations written for the Resident Core")
    return out
