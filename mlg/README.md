# MLG ZERO-1T ΣΩ · universal

**Uma mente, vários corpos.** O **Colab** é o laboratório: lá ela acorda numa GPU sem saber o
que é, mede o próprio corpo, evolui cérebros **do zero** e amadurece o campeão por horas. Os
corpos de campo rodam a mesma mente, em C, offline:

- **Meta Quest 3S** e **qualquer Android** (celular, tablet): o app `MLG.apk`.
- **Qualquer navegador** (computador, iPhone, o navegador do próprio Quest): a página
  <https://math741.github.io/dwds/>, instalável na tela inicial e offline depois da primeira vez.

Enquanto o Colab roda, todos os corpos veem a evolução **ao vivo**. Nada aqui usa modelo
pré-treinado de terceiros: todo cérebro nasce na fornalha e só chega a um corpo depois de passar
nos portões selados (Q root, portão de diálogo e a **prova de comportamento**).

## Começar

1. **Instale.** Quest ou Android: baixe o `MLG.apk` do release
   [mlg-apk](https://github.com/math741/dwds/releases/tag/mlg-apk) (o GitHub Actions gera a cada
   mudança). No Quest, com modo desenvolvedor: [SideQuest](https://sidequestvr.com) ou
   `adb install -r MLG.apk` (o `-r` mantém a memória); o app aparece em **Biblioteca → Fontes
   desconhecidas**. No navegador: abra a página acima.
2. **Já funciona sozinha.** Na primeira abertura, ela baixa os **órgãos de base** que o CI
   constrói do zero (release [mlg-organs](https://github.com/math741/dwds/releases/tag/mlg-organs)):
   o roteador e o conhecimento (corpus da MLG + Wikipédia em português). Sem Colab nenhum, ela já
   faz contas, sabe a hora, lembra o que você conta, mede o corpo e responde pela Wikipédia.
3. **O cérebro (Resident Core) vem do laboratório:**
   [MLG_Quest.ipynb no Colab](https://colab.research.google.com/github/math741/dwds/blob/claude/nifty-hawking-w1ue0u/mlg/MLG_Quest.ipynb)
   → **Executar tudo**. No app, toque em **LIGAÇÃO**, copie o código e crie no Colab (🔑 Secrets)
   `MLG_LINK` com ele. Quando o cérebro passar na prova, o Colab **entrega pelo link
   criptografado**, sem token nenhum (deixe a sessão aberta até o app dizer que recebeu). Com
   `GITHUB_TOKEN` nos Secrets, ele também publica no release `mlg-brain`, e todos os corpos
   pegam de lá. Se o cérebro ainda não passar na prova, rode o notebook de novo: ele continua
   amadurecendo de onde parou.

## Como ela é por dentro

A mente é uma só, `engine/mlg_mind.c`: o Quest a chama por JNI, o navegador por WebAssembly
(`engine/mlg_web.c`) e o laboratório por ctypes (`mlgsub/mind.py`), então o que a prova de
comportamento aprova é exatamente o que cada corpo faz.

| Órgão | O que faz | Vem de |
|---|---|---|
| **Roteador** | decide a intenção | treinado do zero (`router.py` → `router.mlgr`) |
| **Social / autoconhecimento** | cumprimentos; diz o que este corpo sabe fazer agora, a partir dos órgãos presentes; pede esclarecimento quando a mensagem não diz nada | `mlg_mind.c` |
| **Cálculo** | contas exatas por extenso ou com símbolos, porcentagens, raízes; data e hora | `mlg_mind.c` |
| **Memória** | lembra o que você conta, entre sessões | `mlg_mind.c` |
| **Busca** | BM25 sobre o corpus da MLG e a Wikipédia (só prosa: sem bibliografia, listas, calendários); o artigo que a pergunta nomeia responde primeiro, pelo trecho que cobre a pergunta; palavras pesam pela raridade | `know.py` → `know.mlgk`, `mlg_know.c` |
| **Resident Core** | o cérebro evoluído, pesos q8, cache KV; a resposta dele só é dita se tiver palavras de verdade e se apoiar na evidência — senão a busca responde, e o rótulo diz quem falou | a fornalha → `core.mlgb` |
| **Laboratório** | a mente viva em 3D; cérebro selado responde, cérebro ainda imaturo aparece como **rascunho** ao lado da resposta | `link.py` ⇄ `Link.kt` / `mlg-web.js` |
| **Trust core** | cadeia SHA-256 de eventos; recusa órgão com hash errado; mede o Q root de novo em cada corpo | `Trust.kt`, `Brains.kt`, `Transfer.kt` |

O HUD em three.js (`android/app/src/main/assets/hud`) é o mesmo no app e na página
(`web/build.sh` copia). Aberto sozinho num navegador, sem a mente, ele roda em demonstração.

**A prova de comportamento** (`data/trust/behaviour/suite.tsv`): linhas *bare* (corpo sem órgão
nenhum), *must* (com roteador e conhecimento), *wiki* (respostas da Wikipédia) e *quality* —
conversa aberta que nenhum trecho responde, então só o core passa, e só falando em palavras de
verdade. Um core de 30 minutos de CPU reprova (1/10), como deve.

## O que mudou em relação ao mlg-graphhacks

- **Roteador do MoE:** com top-1, o gate não recebia gradiente. Agora recebe, e o forward é o mesmo.
- **Avaliação:** cada byte é pontuado com janelas deslizantes, os mesmos bytes para qualquer contexto.
- **Promoção:** exige ganho seguro (bootstrap pareado, 95%) num **portão de diálogo selado**
  novo (`data/trust/gate`, 16,8 KB), sem regressão significativa no Q root de 2 KB.
- **Herança:** os filhos treinam com metade do LR de pico.
- **Parede de dados:** agora controla quantos bytes cada cérebro viu.
- **Cadeia de eventos:** guarda o assunto do evento, tem índice em `seq` e pode ser ancorada fora do grafo.

## Rodar localmente

```bash
pip install torch falkordblite falkordb paho-mqtt cryptography ziglang
make -C engine && python tests/test_engine.py   # o motor dos corpos bate com o PyTorch
python tests/test_mind.py                       # a mente passa na prova de comportamento
python tests/test_link.py                       # lab ⇄ headset pelo link, inclusive um core em partes
sh web/build.sh && node tests/test_web.js web/site   # a página, no Chromium (precisa do playwright)
python -m mlgsub.organs build out --wiki-mb 64  # órgãos de base com 64 MiB da Wikipédia
```

APK: `cd android && gradle assembleRelease` (SDK com NDK 26.3 e CMake 3.22).

## Configuração única no GitHub

- **Página web:** Settings → Pages → Source: **GitHub Actions**. Como o trabalho está no branch
  `claude/nifty-hawking-w1ue0u`, em Settings → Environments → `github-pages` adicione esse
  branch às regras de deploy (ou junte o branch na `main`).

## Limites honestos

- Um cérebro treinado do zero em poucas horas de GPU escreve no estilo dela e responde bem
  apoiado na busca. Em conversa aberta e longa, fica bem abaixo de modelos treinados com
  trilhões de palavras — e a prova de comportamento não deixa um cérebro que balbucia chegar
  aos corpos.
- **Voz:** usa o reconhecimento e a síntese de fala do sistema (no navegador, a Web Speech API:
  Chrome e Edge reconhecem; Safari e Firefox só falam). Sem reconhecimento, o teclado assume.
- **Link:** passa por brokers MQTT públicos, com tudo selado em AES-GCM pela chave do código de
  ligação; quem não tem o código não lê nem injeta nada.
- **Chave de assinatura:** a `android/mlg-sideload.keystore` está no repositório para que cada
  APK novo instale por cima do anterior sem apagar a memória. Serve para sideload, não para loja.
  O secret `MLG_KEYSTORE_B64` substitui essa chave.
