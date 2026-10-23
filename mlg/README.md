# MLG ZERO-1T ΣΩ · no Meta Quest 3S

**Uma MLG, dois corpos.** O **Colab** é o laboratório: lá ela acorda numa GPU sem saber o que é,
mede o próprio corpo, evolui cérebros **do zero** e amadurece o campeão por horas. O **Quest 3S**
é o corpo de campo: um app que roda a MLG inteira offline. Enquanto o Colab estiver rodando, o
Quest vê a evolução **ao vivo** e usa o cérebro grande do laboratório.

Nada aqui usa modelo pré-treinado de terceiros. Todo cérebro nasce na fornalha, e só é promovido
quando passa nos portões selados (o Q root e o portão de diálogo).

## Começar (3 passos)

1. **Instale o app no Quest.** Baixe o `MLG.apk` do release
   [mlg-apk](https://github.com/math741/dwds/releases/tag/mlg-apk), que o GitHub Actions gera
   sozinho a cada mudança. Com o Quest em modo desenvolvedor, instale com o
   [SideQuest](https://sidequestvr.com) ou com `adb install -r MLG.apk`. O app aparece em
   **Biblioteca → Fontes desconhecidas**.
2. **Abra o laboratório:** [MLG_Quest.ipynb no Colab](https://colab.research.google.com/github/math741/dwds/blob/claude/nifty-hawking-w1ue0u/mlg/MLG_Quest.ipynb).
   Escolha **A100** e clique em **Executar tudo**. O único clique depois é o login do Google,
   para salvar no Drive. Se o Colab cair, rode tudo de novo: ela continua de onde parou.
3. **Ligue os dois corpos (uma vez).** No app, toque em **LIGAÇÃO** e copie o código. No Colab,
   clique na chave 🔑 (Secrets) e crie `MLG_LINK` com esse código. Crie também `GITHUB_TOKEN`
   com um token que possa escrever neste repositório: aí todo cérebro novo chega no Quest
   sozinho, sem você copiar nada.

## Como ela é por dentro

| Órgão | No Quest | Vem de |
|---|---|---|
| **Despertar** | mede o headset: núcleos, memória, bateria, calor, bytes/s do motor por número de threads | `Mind.kt` |
| **Roteador** | decide a intenção e qual órgão responde | treinado do zero no lab (`router.py` → `router.mlgr`) |
| **Cálculo** | contas exatas, por extenso ou com símbolos; data e hora | `Procedural.kt` |
| **Memória** | lembra o que você pediu, entre sessões | `Memory.kt` |
| **Busca** | BM25 sobre o corpus da MLG e a Wikipédia em português, offline | `know.py` → `know.mlgk` |
| **Resident Core** | o cérebro evoluído, em C, pesos q8, com cache KV | a fornalha → `core.mlgb` |
| **Laboratório** | a mente viva ao vivo; o cérebro grande da A100 responde entre passos de treino | `link.py` ⇄ `Link.kt` |
| **Trust core** | cadeia SHA-256 de eventos; recusa órgão com selo errado; mede o Q root de novo no headset | `Trust.kt`, `Brains.kt` |

A interface é um HUD holográfico em three.js (`android/app/src/main/assets/hud`), com o núcleo,
os órgãos em órbita, a **mente viva** em 3D e a telemetria. Abrindo `hud/index.html` num
navegador comum, ele roda em modo demonstração.

## O que mudou em relação ao mlg-graphhacks

- **Roteador do MoE:** com top-1, o gate não recebia gradiente. Agora recebe, e o forward é o mesmo.
- **Avaliação:** cada byte é pontuado com janelas deslizantes, os mesmos bytes para qualquer contexto.
- **Promoção:** exige ganho seguro (bootstrap pareado, 95%) num **portão de diálogo selado**
  novo (`data/trust/gate`, 16,8 KB), sem regressão significativa no Q root de 2 KB.
- **Herança:** os filhos treinam com metade do LR de pico.
- **Parede de dados:** agora controla quantos bytes cada cérebro viu. "Perdeu por ter treinado
  menos" deixou de ser registrado como "o dado é o limite".
- **Cadeia de eventos:** guarda o assunto do evento, tem índice em `seq` e pode ser ancorada fora do grafo.

## Rodar localmente

```bash
pip install torch falkordblite falkordb paho-mqtt cryptography
make -C engine && python tests/test_engine.py   # o motor do Quest bate com o PyTorch
python -m mlgsub --quick                        # fornalha em miniatura
```

Para compilar o APK: `cd android && gradle assembleRelease`. Precisa do SDK com
NDK 26.3 e CMake 3.22.

## Limites honestos

- Um cérebro treinado do zero em poucas horas de A100 escreve no estilo dela e responde bem
  apoiado na busca. Em conversa aberta e longa, ele fica bem abaixo de modelos treinados com
  trilhões de palavras.
- **Voz:** usa o reconhecimento e a síntese de fala do sistema do Quest, quando existem. Se não
  houver, o app avisa e o teclado assume (a tecla de microfone do teclado dita).
- **Valores do Quest:** os números de desempenho do Quest no lab começam **ASSUMED**. Viram
  **MEASURED** quando o app, ligado, manda as medições dele.
- **Chave de assinatura:** a `android/mlg-sideload.keystore` está no repositório para que cada
  APK novo instale por cima do anterior sem apagar a memória. Serve para sideload no seu próprio
  headset, não para loja. O secret `MLG_KEYSTORE_B64` substitui essa chave.
