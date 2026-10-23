"""Builds MLG_Quest.ipynb, the one-click lab: python make_notebook.py"""
import nbformat as nbf

REPO = "math741/dwds"
BRANCHES = ["main", "claude/nifty-hawking-w1ue0u"]

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# MLG ZERO-1T ΣΩ · o laboratório

**Uma MLG, dois corpos.** Este notebook é o laboratório: aqui a MLG acorda numa GPU sem saber
o que ela é, mede o próprio corpo, evolui cérebros do zero e amadurece o campeão por horas.
O **Quest 3S** é o corpo de campo: o app roda a MLG inteira offline, e enquanto este notebook
estiver rodando ele vê a evolução **ao vivo** e usa o cérebro grande daqui.

### Como usar
1. **Ambiente de execução → Alterar o tipo → A100** (ou outra GPU; a MLG mede o que receber).
2. *(uma vez só, opcional)* Na chave 🔑 à esquerda (**Secrets**), crie:
   - `MLG_LINK` = o código que o app mostra em **LIGAÇÃO** → toda execução acha seu Quest sozinha;
   - `GITHUB_TOKEN` = um token do GitHub com permissão de escrita em `math741/dwds` → o cérebro
     novo chega no Quest sozinho. Sem ele, os arquivos ficam no seu Drive em `MLG/quest`.
3. **Ambiente de execução → Executar tudo.** O único clique depois disso é o login do Google
   (para guardar tudo no Drive). Se o Colab cair, rode tudo de novo: ela continua de onde parou.

Nada aqui usa modelo pré-treinado de ninguém: todo cérebro nasce do zero, e só é promovido
quando passa nos portões selados (o Q root e o portão de diálogo)."""),
    code(f"""#@title 1 · Preparar o laboratório
HOURS = None        # None: a MLG decide pelo corpo (A100: 3 h). Ou um número de horas.
WIKI_MB = None      # None: decide pelo corpo (A100: 1024 MiB). 0 = só o corpus da MLG.
SERVE_HOURS = 1.0   # no fim, o cérebro do lab fica à disposição do Quest por este tempo
REPO, BRANCHES = "{REPO}", {BRANCHES!r}

import os, subprocess, sys
from google.colab import drive, userdata
drive.mount("/content/drive")
HOME = "/content/drive/MyDrive/MLG"
if os.path.exists("/content/dwds/mlg"):
    subprocess.run(["git", "-C", "/content/dwds", "pull", "-q"])
else:
    for br in BRANCHES:
        subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", br, f"https://github.com/{{REPO}}", "/content/dwds"])
        if os.path.exists("/content/dwds/mlg"):
            break
        subprocess.run(["rm", "-rf", "/content/dwds"])
!pip -q install falkordblite falkordb nvidia-ml-py paho-mqtt datasets
!make -s -C /content/dwds/mlg/engine
sys.path.insert(0, "/content/dwds/mlg")

import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
top = any(k in gpu for k in ("A100", "H100"))
HOURS = HOURS or (3.0 if top else 2.0 if "L4" in gpu else 1.0 if torch.cuda.is_available() else 0.2)
WIKI_MB = WIKI_MB if WIKI_MB is not None else (1024 if top else 256 if torch.cuda.is_available() else 0)
POP, GENS, SECS = (8, 5, 75) if top else (6, 4, 60) if torch.cuda.is_available() else (3, 2, 20)

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

print(f"corpo do laboratório: {{gpu}} · orçamento {{HOURS}} h · Wikipédia {{WIKI_MB}} MiB · Drive: {{HOME}}")"""),
    code("""#@title 2 · Ligar ao Quest (a mente aparece ao vivo no app)
from mlgsub.pipeline import Run
run = Run(HOME, hours=HOURS, quick=not torch.cuda.is_available())
run.link(secret("MLG_LINK"))"""),
    md("""## I · Despertar
A MLG pergunta ao driver o que ela é (**REPORTED**) e depois testa (**MEASURED**): memória escrita,
banda por tamanho (com bisseção do penhasco do cache), formatos numéricos e watts. O corpo de
campo é o Quest 3S: se o app estiver aberto e ligado, ele manda as medições dele e os limites
deixam de ser **ASSUMED**."""),
    code("""#@title 3 · Despertar
run.awaken();"""),
    md("""## II · Alimentar
A Wikipédia em português vira conhecimento: um índice BM25 que vai para o Quest, e conversas em
que a resposta se apoia num trecho encontrado, para o Resident Core aprender a usar a busca.
O roteador (qual órgão responde cada intenção) é treinado do zero aqui."""),
    code("""#@title 4 · Alimentar e treinar o roteador
run.feed(WIKI_MB)
run.train_router()
run.load_corpus()"""),
    md("""## III · Evoluir e amadurecer
Cada genoma passa por SANDBOX → TRAIN → ATTACK → BENCHMARK → COMPARE. O que não cabe no Quest
morre antes de rodar. Filhos herdam os pesos dos pais. Promoção só com ganho **seguro** no portão
de diálogo selado, sem piorar o Q root. Depois, o campeão vira a semente de um cérebro maior que
treina pelo resto do orçamento, salvando no Drive a cada 10 minutos.

**Abra o app no Quest → MENTE VIVA** para ver tudo nascer ao vivo. Você pode conversar com ela
durante o treino: o cérebro do laboratório responde entre um passo e outro."""),
    code("""#@title 5 · Evoluir
run.evolve(population=POP, generations=GENS, seconds=SECS)"""),
    code("""#@title 6 · Amadurecer (retoma sozinho se o Colab cair)
run.mature()"""),
    md("""## IV · Mandar para o Quest
O campeão vai para o formato do motor do Quest (pesos q8), e o próprio motor em C mede o Q root
de novo: o app recusa um cérebro que não reproduza essa medida."""),
    code("""#@title 7 · Exportar e publicar
run.export()
run.publish(secret("GITHUB_TOKEN"), REPO)"""),
    code("""#@title 8 · Conversar aqui (opcional)
for q in ["o que é a fotossíntese?", "quem foi Santos Dumont?", "o que você é?"]:
    print("você:", q)
    print("MLG:", run.ask(q), "\\n")"""),
    code("""#@title 9 · Ficar à disposição do Quest
run.serve(SERVE_HOURS)"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": [], "machine_shape": "hm"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"}})
nbf.write(nb, "MLG_Quest.ipynb")
print("wrote MLG_Quest.ipynb with", len(cells), "cells")
