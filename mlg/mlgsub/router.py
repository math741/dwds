"""The trainable router: which organ answers an intent (manifesto: "qual
expert chamar"). A linear map over hashed text features, trained from
scratch here and run on the headset by engine/mlg_router.c, which extracts
exactly the same features (see that file for their definition).

Intents and the organ behind each one on the headset:
    chat      conversation and questions about the MLG itself   Resident Core
    knowledge facts about the world                             retrieval + Resident Core
    math      arithmetic and units                               procedural expert, exact
    time      hour, date, weekday                                procedural expert
    remember  "lembra que...", "meu nome é..."                   persistent memory, write
    recall    "qual é meu nome?", "o que eu te disse?"           persistent memory, read
    body      the headset's own state: battery, heat, memory     body sensors
    lab       the Colab lab: evolution, live brain, link         live link
"""
import math
import random
import struct

import torch

from .text import normalize

INTENTS = ["chat", "knowledge", "math", "time", "remember", "recall", "body", "lab"]
DIM = 1 << 15

_N = ["um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez", "doze", "quinze",
      "vinte", "trinta", "cem", "mil"]
_OPS = ["mais", "menos", "vezes", "dividido por", "+", "-", "x", "*", "/"]
_THINGS = ["a capital da frança", "a fotossíntese", "galileu", "o sistema solar", "a revolução francesa",
           "um buraco negro", "o dna", "machado de assis", "a internet", "o brasil", "júpiter", "a segunda guerra",
           "a lua", "einstein", "o coração humano", "a amazônia", "pitágoras", "o oceano pacífico",
           "a independência do brasil", "os dinossauros", "a eletricidade", "um vulcão", "santos dumont",
           "a inflação", "o big bang", "a teoria da evolução", "o japão", "uma célula", "a gravidade"]
_FACTS = ["meu nome é {n}", "eu me chamo {n}", "lembra que eu gosto de {x}", "anota que amanhã tenho {x}",
          "guarda isso: {x}", "lembre que minha cor favorita é {c}", "não esquece que eu moro em {p}",
          "memoriza que meu aniversário é dia {d}", "lembra que {x}", "minha comida favorita é {x}"]
_NAMES = ["matheus", "ana", "joão", "maria", "lucas", "bia", "pedro", "carla"]
_XS = ["pizza", "prova de matemática", "futebol", "reunião às 10", "o dentista", "jogar no quest",
       "estudar física", "comprar pão"]


def _num(rng):
    return str(rng.randint(0, 999)) if rng.random() < 0.6 else rng.choice(_N)


def examples(rng, per_intent=1500, corpus_questions=()):
    """Labelled requests, written from templates and the MLG corpus."""
    gen = {
        "math": lambda: rng.choice([
            "quanto é {a} {o} {b}", "quanto dá {a} {o} {b}?", "calcula {a} {o} {b}", "{a} {o} {b}",
            "qual o resultado de {a} {o} {b}", "faz a conta {a} {o} {b}", "raiz quadrada de {a}",
            "quanto é {a} por cento de {b}", "{a} km em metros", "converte {a} graus para fahrenheit",
            "quantos minutos tem em {a} horas", "{a} ao quadrado", "me diz {a} {o} {b}"]).format(
            a=_num(rng), b=_num(rng), o=rng.choice(_OPS)),
        "time": lambda: rng.choice([
            "que horas são", "que horas são agora?", "me diz a hora", "que dia é hoje", "qual a data de hoje",
            "hoje é que dia da semana", "em que mês estamos", "que ano é", "data de hoje", "horas",
            "quantos dias faltam pro fim do mês", "é de manhã ou de tarde?", "qual é o dia de hoje?"]),
        "remember": lambda: rng.choice(_FACTS).format(n=rng.choice(_NAMES), x=rng.choice(_XS),
                                                      c=rng.choice(["azul", "verde", "preto"]),
                                                      p=rng.choice(["são paulo", "recife", "curitiba"]),
                                                      d=rng.randint(1, 28)),
        "recall": lambda: rng.choice([
            "qual é meu nome", "como eu me chamo?", "você lembra do que eu disse?", "o que eu te pedi pra lembrar",
            "do que você lembra sobre mim", "qual é minha cor favorita", "o que eu tenho amanhã?",
            "onde eu moro?", "quando é meu aniversário", "você sabe quem eu sou?", "o que você anotou",
            "me lembra o que eu falei ontem", "qual minha comida favorita?"]),
        "body": lambda: rng.choice([
            "quanto de bateria você tem", "como está sua bateria?", "você está quente?", "qual sua temperatura",
            "quanta memória você tem", "como está seu corpo", "você está esquentando?", "estado do sistema",
            "quanto processamento você tem", "mede seu corpo de novo", "qual seu desempenho agora",
            "o que você faz quando a temperatura sobe?", "quantos bytes por segundo você pensa?"]),
        "lab": lambda: rng.choice([
            "como está a evolução", "mostra o cérebro", "me mostra a mente viva", "conecta no laboratório",
            "o colab está ligado?", "quantos cérebros você já treinou", "quem é o campeão agora",
            "como está o treino", "mostra a evolução em tempo real", "por que esse cérebro venceu",
            "usa o cérebro grande", "você está ligada na a100?", "quantas gerações já rodaram"]),
        "knowledge": lambda: rng.choice([
            "o que é {t}", "quem foi {t}", "me fala sobre {t}", "explica {t}", "o que você sabe sobre {t}",
            "qual a importância de {t}", "como funciona {t}", "onde fica {t}", "quando aconteceu {t}",
            "me conta de {t}", "{t}"]).format(t=rng.choice(_THINGS)),
        "chat": lambda: rng.choice([
            "oi", "olá", "bom dia", "boa noite", "tudo bem?", "quem é você", "o que você é", "obrigado",
            "valeu", "você é inteligente?", "me conta uma curiosidade", "o que é a mlg", "o que você sente",
            "você gosta de mim?", "tchau", "até amanhã", "qual seu propósito", "como você funciona",
            "você pensa?", "o que é o q root", "por que você se chama mlg", "me ajuda", "estou triste",
            "kkkk", "legal", "conta uma piada", "você é melhor que o jarvis?", "e aí"]),
    }
    out = [(gen[i](), i) for i in INTENTS for _ in range(per_intent)]
    mlg_words = ("você", "voce", "mlg", "q root", "seu ", "sua ", "te ")
    for q in corpus_questions:
        out.append((q, "chat" if any(w in q.lower() for w in mlg_words) else "knowledge"))
    rng.shuffle(out)
    return out


def _kind_hash(kind, data):
    h = 2166136261 ^ ((kind * 0x9E3779B1) & 0xFFFFFFFF)
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def features(text, dim=DIM):
    """{bucket: value}, exactly as mlg_router_route() computes them."""
    s = b" " + normalize(text)[:1022] + b" "
    counts = {}

    def add(h):
        b = h & (dim - 1)
        counts[b] = counts.get(b, 0) + 1

    ws = [w for w in s[1:-1].split(b" ") if w]
    for i, w in enumerate(ws):
        add(_kind_hash(1, w))
        if i:
            add(_kind_hash(2, ws[i - 1] + b" " + w))
    for g in (3, 4):
        for i in range(len(s) - g + 1):
            add(_kind_hash(g, s[i:i + g]))
    vals = {b: math.log1p(c) for b, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vals.values())) or 1.0
    return {b: v / norm for b, v in vals.items()}


def train(data, dim=DIM, epochs=8, lr=0.5, seed=0, log=print):
    """Multinomial logistic regression on the hashed features; returns (w, b, held-out accuracy)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    data = list(data)
    rng.shuffle(data)
    cut = len(data) // 10
    test, fit = data[:cut], data[cut:]

    def tensor(rows):
        idx, val, ys = [], [], []
        for r, (text, intent) in enumerate(rows):
            for b, v in features(text, dim).items():
                idx.append((r, b))
                val.append(v)
            ys.append(INTENTS.index(intent))
        i = torch.tensor(idx).t()
        return torch.sparse_coo_tensor(i, torch.tensor(val), (len(rows), dim),
                                       check_invariants=False).coalesce(), torch.tensor(ys)

    X, y = tensor(fit)
    Xt, yt = tensor(test)
    W = torch.zeros(dim, len(INTENTS), requires_grad=True)
    b = torch.zeros(len(INTENTS), requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr / 10)
    for _ in range(epochs * 10):
        loss = torch.nn.functional.cross_entropy(torch.sparse.mm(X, W) + b, y) + 1e-5 * (W ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = ((torch.sparse.mm(Xt, W) + b).argmax(1) == yt).float().mean().item()
    log(f"[router] {len(fit)} requests learned, {acc:.1%} right on {len(test)} it never saw")
    return W.detach().t().contiguous(), b.detach(), acc


def write(path, w, b, dim=DIM):
    with open(path, "wb") as fh:
        fh.write(b"MLGR" + struct.pack("<3I", 1, len(INTENTS), dim))
        for name in INTENTS:
            fh.write(struct.pack("<I", len(name)) + name.encode())
        fh.write(b.float().numpy().astype("<f4").tobytes())
        fh.write(w.float().numpy().astype("<f4").tobytes())


def route(w, b, text):
    f = features(text, w.shape[1])
    logits = b.clone()
    for k, v in f.items():
        logits += w[:, k] * v
    return logits.softmax(0)
