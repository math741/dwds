"""The headset engine must compute what the lab trained.

Builds engine/mlg_cli, exports brains of several shapes, and checks the C
logits, bits per byte and greedy generation against PyTorch.

    python tests/test_engine.py
"""
import math
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from mlgsub import brain as B, export  # noqa: E402

ENGINE = os.path.join(HERE, "..", "engine")
CLI = os.path.join(ENGINE, "mlg_cli")
SHAPES = [
    dict(d_model=64, n_layers=2, n_heads=4, n_experts=1, top_k=1, ffn_mult=2, ctx=64),
    dict(d_model=96, n_layers=3, n_heads=2, n_experts=4, top_k=1, ffn_mult=3, ctx=128),
    dict(d_model=64, n_layers=2, n_heads=4, n_experts=4, top_k=2, ffn_mult=4, ctx=96),
]


def random_brain(shape, seed):
    g = dict(B.ANCESTOR, **shape, dtype="fp32", dropout=0.0)
    torch.manual_seed(seed)
    model = B.Brain(g)
    with torch.no_grad():  # weights big enough that every path matters
        for name, p in model.named_parameters():
            p.add_(torch.randn_like(p) * (0.3 if "ln" in name else 0.08))
    return model.eval(), g


def cli(*args):
    return subprocess.run([CLI, *map(str, args)], check=True, capture_output=True).stdout


def main():
    subprocess.run(["make", "-s", "-C", ENGINE], check=True)
    tmp = tempfile.mkdtemp()
    text = open(os.path.join(HERE, "..", "data", "trust", "qroot", "language.txt"), "rb").read()
    worst = 0.0
    for i, shape in enumerate(SHAPES):
        model, g = random_brain(shape, i)
        for wtype in (export.F32, export.Q8):
            path = os.path.join(tmp, f"b{i}-{wtype}.mlgb")
            export.write(path, model.state_dict(), g, wtype)
            ref = B.Brain(g)
            ref.load_state_dict(export.as_deployed(model.state_dict(), g, wtype))
            ref.eval()
            x = torch.tensor(list(text[: g["ctx"]]))
            open(os.path.join(tmp, "in.bin"), "wb").write(bytes(x.tolist()))
            cli("logits", path, os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.f32"))
            got = torch.frombuffer(bytearray(open(os.path.join(tmp, "out.f32"), "rb").read()),
                                   dtype=torch.float32).view(-1, 256)
            with torch.no_grad():
                want = ref(x[None])[0][0]
            err = (got - want).abs().max().item() / max(1.0, want.abs().max().item())
            worst = max(worst, err)
            assert err < 2e-4, f"shape {i} wtype {wtype}: logits differ by {err:.2e}"

            # bits per byte on the Q root, as the lab scores it
            c_bpb = float(cli("bpb", path, os.path.join(HERE, "..", "data", "trust", "qroot", "language.txt")))
            py_bpb = B.bits_per_byte(ref, g, torch.tensor(list(text), dtype=torch.uint8), torch.device("cpu"))
            assert abs(c_bpb - py_bpb) < 1e-3, f"bpb {c_bpb} vs {py_bpb}"

            # greedy generation inside one context
            prompt = text[:20]
            c_out = subprocess.run([CLI, "say", path, prompt, "0", "0"], check=True,
                                   capture_output=True).stdout
            idx = torch.tensor([list(prompt)])
            with torch.no_grad():
                for _ in range(12):
                    nxt = ref(idx)[0][0, -1].argmax()
                    idx = torch.cat([idx, nxt.view(1, 1)], 1)
            want_bytes = bytes(idx[0, len(prompt):].tolist())
            assert c_out.startswith(want_bytes), f"generation {c_out[:12]!r} vs {want_bytes!r}"
        print(f"shape {i} ({g['n_experts']} experts, top-{g['top_k']}): logits, bpb and generation match")
    print(f"engine matches PyTorch (worst relative logit error {worst:.1e})")


if __name__ == "__main__":
    main()
