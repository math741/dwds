#!/bin/sh
# The MLG in any browser: the headset's engine and mind compiled to WebAssembly
# (zig cc: pip install ziglang), next to the headset's own HUD, in site/ (or $1).
#
#     sh web/build.sh            then serve web/site over http(s), e.g. python3 -m http.server -d web/site
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/site}
ENGINE=$HERE/../engine
HUD=$HERE/../android/app/src/main/assets/hud
ZIG=${ZIG:-python3 -m ziglang}
mkdir -p "$OUT"

EXPORTS="malloc free mlg_core_open_mem mlg_core_close mlg_know_open_mem mlg_know_close mlg_know_count
mlg_router_open_mem mlg_router_close mlg_router_count mlg_router_name mlg_mind_new mlg_mind_free mlg_mind_organs
mlg_mind_body mlg_mind_lab mlg_mind_think mlg_mind_finish mlg_mind_thought_json mlg_mind_memory_dump
mlg_mind_memory_load mlg_mind_memory_count web_thought_new web_thought_prompt web_speak web_clock web_core_info
web_core_bench mlg_core_bits_per_byte mlg_mind_spoke"
FLAGS=""
for e in $EXPORTS; do FLAGS="$FLAGS -Wl,--export=$e"; done

$ZIG cc -target wasm32-wasi -O3 -msimd128 -mexec-model=reactor -std=c11 -D_DEFAULT_SOURCE -s \
    -Wno-format-truncation $FLAGS -Wl,--initial-memory=33554432 -Wl,--max-memory=4294901760 \
    "$ENGINE/mlg_core.c" "$ENGINE/mlg_know.c" "$ENGINE/mlg_router.c" "$ENGINE/mlg_text.c" "$ENGINE/mlg_mind.c" \
    "$ENGINE/mlg_web.c" -o "$OUT/mlg.wasm" -lm

# the HUD is the headset's, file for file; the page adds the browser's body (mlg-web.js) before it
cp "$HUD/hud.js" "$HUD/three.module.min.js" "$HERE/mlg-web.js" "$HERE/mlg-worker.js" "$OUT/"
cp "$HERE/../data/trust/qroot/language.txt" "$OUT/"  # the Q root, re-measured on every body
sed -e 's#<script type="module" src="hud.js"></script>#<script src="mlg-web.js"></script>\n<script type="module" src="hud.js"></script>#' \
    -e 's#</head>#<link rel="manifest" href="manifest.webmanifest"><link rel="icon" href="icon.svg"><meta name="theme-color" content="\#05070A">\n</head>#' \
    "$HUD/index.html" > "$OUT/index.html"
cp "$HERE/manifest.webmanifest" "$HERE/sw.js" "$HERE/icon.svg" "$OUT/"
echo "web: $(du -h "$OUT/mlg.wasm" | cut -f1) mlg.wasm -> $OUT"
