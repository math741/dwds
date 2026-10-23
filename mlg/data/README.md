# Data bundled from MLG ZERO-1T

Byte-identical copies of the MLG's own text, kept in the same layout as the main tree, so this
folder runs on its own. None of it was written for Graph Hacks.

| Path | Role in the furnace |
|---|---|
| `core/data/corpus*.txt` | training text |
| `core/data/valid*.txt` | dev suite: selection of parents and champions |
| `trust/qroot/language.txt` | frozen Q root: the promotion gate, never used for selection |
| `trust/qroot.lock` | SHA-256 seal; the furnace refuses to evaluate if `language.txt` does not match |
| `trust/qroot/policy.json` | Q root formula and tolerance |

Lines of the Q root or dev suites that also appear in the training text are removed from
training at load time (`mlgsub/brain.py`, `Corpus`).
