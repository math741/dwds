"""The live link across languages: the lab (mlgsub/link.py) and the headset's
Link.kt (run as a JVM test) meet on a local MQTT broker, sealed with the same code.

    python tests/test_link.py        # needs amqtt, paho-mqtt, cryptography and gradle
"""
import asyncio
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from mlgsub import brain as B, link as L  # noqa: E402
from mlgsub.substrate import connect  # noqa: E402

PORT = 18831


def broker():
    from amqtt.broker import Broker

    async def run():
        b = Broker({"listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{PORT}"}},
                    "auth": {"allow-anonymous": True}, "topic-check": {"enabled": False}})
        await b.start()
        await asyncio.sleep(3600)
    asyncio.run(run())


def main():
    threading.Thread(target=broker, daemon=True).start()
    time.sleep(2)
    code = L.new_code()
    lab = L.Link(code, brokers=[("127.0.0.1", PORT, False)])
    sub = connect(path=os.path.join(os.environ.get("TMPDIR", "/tmp"), "mlg-link-test.db"), fresh=True)
    lab.attach(sub)
    g = dict(B.ANCESTOR, d_model=64, n_layers=2, n_heads=4, n_experts=1, ctx=64, dtype="fp32")
    lab.brain = (B.Brain(g), g, "brain:test")
    sub.obj("Body", "body:lab", origin="test", epistemic="MEASURED", name="lab")
    stop = threading.Event()

    def serve():  # what the furnace does between training steps
        while not stop.is_set():
            lab.tick()
            time.sleep(0.05)
    threading.Thread(target=serve, daemon=True).start()
    env = dict(os.environ, MLG_TEST_PORT=str(PORT), MLG_TEST_CODE=code, MLG_TEST_TOPIC=L.topic_base(code))
    r = subprocess.run(["gradle", "testReleaseUnitTest", "--no-daemon", "-q", "--tests", "ai.mlg.zero.LinkTest", "-i"],
                       cwd=os.path.join(HERE, "..", "android"), env=env, capture_output=True, text=True)
    stop.set()
    out = r.stdout + r.stderr
    for line in out.splitlines():
        if "mind items" in line or "FAILED" in line or "AssertionError" in line:
            print(line.strip())
    assert r.returncode == 0, out[-3000:]
    assert lab.field_report and lab.headsets, "the lab never heard the headset"
    print(f"link works both ways: headset measured {lab.field_report['decode_ms']} ms/byte, lab answered over MQTT")


if __name__ == "__main__":
    main()
