"""The live link: one MLG in two bodies at once.

The lab (Colab) and the field body (the Quest app) meet on a public MQTT
broker. Nothing readable crosses it: every message is AES-GCM sealed with a
key derived from the pairing code the headset shows on its first launch,
and the topics are a hash of that code too. Put the code in Colab's
Secrets as MLG_LINK once; every later run of the notebook finds the
headset by itself.

What flows, under mlg/<topic>/...:
    mind    lab -> headset   every object, relation and event the substrate writes, live
    train   lab -> headset   steps, loss and which brain is learning, every couple of seconds
    say     lab -> headset   the lab brain's answer, streamed; "sealed" when that brain passed
                             the behaviour suite (until then the headset shows it as a draft)
    ask     headset -> lab   a prompt for the lab brain (the headset routes, retrieves and builds it)
    offer   lab -> headset   a sealed Resident Core is ready: its sha256, size, what the lab measured
    pull    headset -> lab   send me parts [from, from + count) of the core with this sha256
    part    lab -> headset   one part of it, base64; the headset keeps the file only if it hashes right
    hello   headset -> lab   a headset arrived: the lab replies with the whole mind so far
    body    headset -> lab   the headset's own measurements (its awakening)
"""
import base64
import codecs
import hashlib
import json
import os
import queue
import threading
import time

BROKERS = [("broker.emqx.io", 8883, True), ("broker.hivemq.com", 8883, True), ("test.mosquitto.org", 8886, True)]
ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

CHUNK = 32 * 1024  # bytes of a core per part: well under what public brokers carry


def new_code():
    return "-".join("".join(ALPHABET[b % len(ALPHABET)] for b in os.urandom(4)) for _ in range(4))


def _norm(code):
    return "".join(c for c in code.upper() if c in ALPHABET).encode()


def topic_base(code):
    return "mlg/" + hashlib.sha256(b"mlg-link-topic:" + _norm(code)).hexdigest()[:24]


def key(code):
    return hashlib.sha256(b"mlg-link-key:" + _norm(code)).digest()


class Link:
    def __init__(self, code, log=print, brokers=BROKERS):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import paho.mqtt.client as mqtt

        self.code, self.log = code, log
        self.base = topic_base(code)
        self.aes = AESGCM(key(code))
        self.pending = queue.Queue()  # prompts from the headset
        self.outbox = []              # mind messages, flushed in batches
        self.lock = threading.Lock()
        self.brain = None             # (model, genome, brain id) that answers the headset
        self.training = None          # what the lab is training right now
        self.training_g = None
        self.brain_seconds = 0
        self.sealed = None            # the brain that passed the behaviour suite, if any
        self.frac = 0.0
        self.sub = None
        self.field_report = None      # the headset's own measurements
        self.offering = None          # the sealed core the headset may pull, and where it is
        self.pulls = queue.Queue()
        self.headsets = 0
        self._last_train = 0.0
        self._since = (None, time.time(), 0.0)  # (training, when, fraction) the time-left estimate starts from
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="mlg-lab-" + os.urandom(4).hex())
        self.client.on_connect = lambda c, u, f, rc, p=None: c.subscribe(
            [(f"{self.base}/{k}", 0) for k in ("ask", "hello", "body", "pull")])
        self.client.on_message = self._on_message
        self.broker = None
        tls_ready = False
        for host, port, tls in brokers:
            try:
                if tls and not tls_ready:
                    self.client.tls_set()
                    tls_ready = True
                self.client.connect(host, port, keepalive=30)
                self.broker = host
                break
            except Exception as e:  # try the next public broker
                log(f"[link] {host} unreachable ({type(e).__name__})")
        if not self.broker:
            raise ConnectionError("no MQTT broker reachable")
        self.client.loop_start()
        threading.Thread(target=self._flusher, daemon=True).start()
        threading.Thread(target=self._send_parts, daemon=True).start()
        log(f"[link] lab online at {self.broker}, waiting for the headset (code {code[:4]}-…)")

    # ---- sealed messages ---------------------------------------------------
    def _seal(self, obj):
        nonce = os.urandom(12)
        return nonce + self.aes.encrypt(nonce, json.dumps(obj, ensure_ascii=False).encode(), None)

    def _open(self, blob):
        return json.loads(self.aes.decrypt(blob[:12], blob[12:], None))

    def publish(self, kind, obj):
        self.client.publish(f"{self.base}/{kind}", self._seal(obj), qos=0)

    def _on_message(self, client, userdata, msg):
        try:
            data = self._open(msg.payload)
        except Exception:
            return  # not sealed with our code: ignore it
        kind = msg.topic.rsplit("/", 1)[-1]
        if kind == "ask":
            self.pending.put(data)
        elif kind == "hello":
            self.headsets += 1
            self.log(f"[link] headset connected: {data.get('device', 'Quest')}")
            threading.Thread(target=self._snapshot, daemon=True).start()
            if self.offering:
                self.publish("offer", self.offering[0])
        elif kind == "pull":
            self.pulls.put(data)
        elif kind == "body":
            self.field_report = data
            self.log(f"[link] the headset measured itself: {data.get('decode_ms', 0):.2f} ms/byte, "
                     f"{data.get('prefill_ms', 0):.3f} ms per prompt byte, {data.get('threads')} threads")

    # ---- the mind, live ----------------------------------------------------
    def attach(self, sub, furnace=None):
        """Mirror the substrate to the headset; let a furnace answer and report through the link."""
        self.sub = sub
        sub.observers.append(self._observe)
        if furnace is not None:
            furnace.on_step = self.tick
            furnace.on_commit = self.commit
            run = furnace.run

            def labelled(g, generation, *a, **kw):
                from .brain import genome_id
                self.training = {"genome": genome_id(g), "generation": generation,
                                 "shape": f"d{g['d_model']} L{g['n_layers']} E{g['n_experts']}k{g['top_k']}"}
                self.training_g = g
                try:
                    return run(g, generation, *a, **kw)
                finally:
                    self.training = None
            furnace.run = labelled

    def _observe(self, kind, payload):
        if "reason" in payload:
            payload = dict(payload, reason=str(payload["reason"])[:200])
        with self.lock:
            self.outbox.append([kind, payload])

    def _flusher(self):
        while True:
            time.sleep(0.3)
            with self.lock:
                batch, self.outbox = self.outbox, []
            for i in range(0, len(batch), 200):
                self.publish("mind", {"items": batch[i:i + 200]})

    def _snapshot(self):
        """Everything so far, so a headset that arrives late sees the whole mind."""
        if self.sub is None:
            return
        nodes = self.sub.q("MATCH (n:Object) WHERE n.type <> 'Event' RETURN n.id, n.type, n.epistemic, "
                           "coalesce(n.verdict, n.name, ''), n.params, n.dev_bpb, n.bits_per_byte LIMIT 5000")
        edges = self.sub.q("MATCH (a:Object)-[r]->(b:Object) WHERE a.type <> 'Event' AND b.type <> 'Event' "
                           "RETURN a.id, type(r), b.id LIMIT 15000")
        items = [["obj", {"id": i, "type": t, "ep": e, "name": n, "params": p, "dev_bpb": d, "bits_per_byte": b}]
                 for i, t, e, n, p, d, b in nodes] + [["link", {"a": a, "r": r, "b": b}] for a, r, b in edges]
        for i in range(0, len(items), 300):
            self.publish("mind", {"items": items[i:i + 300], "snapshot": True})
        if self.brain:
            self.publish("train", {"champion": self.brain[2], "sealed": self.brain[2] == self.sealed})

    # ---- the lab brain answers the headset ---------------------------------
    def commit(self, row):
        if row.get("model") is not None:
            self.brain = (row["model"], row["g"], row["brain"])
            self.brain_seconds = row.get("lineage_seconds") or row.get("seconds") or 0
            self.publish("train", {"champion": row["brain"], "params": row["params"],
                                   "trained_s": round(self.brain_seconds), "sealed": row["brain"] == self.sealed})

    def seal(self, brain_id):
        """This brain passed the behaviour suite: from now on its answers are answers, not drafts."""
        self.sealed = brain_id
        if self.brain and self.brain[2] == brain_id:
            self.publish("train", {"champion": brain_id, "sealed": True})

    def tick(self, model=None, steps=0, frac=0.0, loss=None):
        """Between training steps: report progress, answer what the headset asked.
        During maturation the brain still learning answers (it is the best one)."""
        now = time.time()
        self.frac = frac
        key = (self.training or {}).get("genome")
        if self._since[0] != key or frac < self._since[2]:
            self._since = (key, now, frac)  # how fast this brain's training moves, for "faltam ~1h52"
        if loss is not None and now - self._last_train > 2:
            self._last_train = now
            _, t0, f0 = self._since
            eta = (1 - frac) * (now - t0) / (frac - f0) if frac > f0 and now - t0 > 20 else None
            self.publish("train", {"steps": steps, "frac": round(frac, 4), "loss_bpb": loss / 0.6931471805599453,
                                   **({"eta_s": round(eta)} if eta is not None else {}), **(self.training or {})})
        while not self.pending.empty():
            self.answer(self.pending.get(), model if self.training and self.training["generation"] == -1 else None)

    def answer(self, ask, model=None):
        from . import brain as B

        brain = self.brain if model is None else (model, self.training_g, "maturing")
        trained = {"trained_s": round(self.brain_seconds)} if model is None else {"maturing": round(self.frac, 3)}
        if brain is None:
            self.publish("say", {"id": ask.get("id"), "err": "the lab has no committed brain yet", "done": True})
            return
        m, g, bid = brain
        trained["sealed"] = model is None and bid == self.sealed
        dec, buf, sent = codecs.getincrementaldecoder("utf-8")("replace"), [], [0]

        def on_byte(b):
            buf.append(dec.decode(bytes([b])))
            if len(buf) >= 12:
                self.publish("say", {"id": ask.get("id"), "chunk": "".join(buf), "done": False, "brain": bid, **trained})
                buf.clear()
            return False

        device = next(m.parameters()).device
        B.generate(m, g, ask.get("prompt", ""), device, n=int(ask.get("max", 320)),
                   temperature=float(ask.get("temp", 0.6)), top_p=0.9, seed=int(time.time()), on_byte=on_byte)
        self.publish("say", {"id": ask.get("id"), "chunk": "".join(buf) + dec.decode(b"", final=True),
                             "done": True, "brain": bid, **trained})

    # ---- a sealed core, delivered over the link -----------------------------
    def offer(self, path, meta):
        """Let the headset pull this Resident Core (it passed the behaviour suite): no
        release, no token. `meta` is what the lab measured (version, core, behaviour)."""
        from .export import sha256

        info = dict(meta, name="core.mlgb", sha256=sha256(path), bytes=os.path.getsize(path), chunk=CHUNK)
        self.offering = (info, path)
        self.publish("offer", info)
        self.log(f"[link] Resident Core oferecido ao Quest pelo link ({info['bytes'] / 2**20:.1f} MiB, "
                 f"sha256 {info['sha256'][:12]}…): ele baixa sozinho quando estiver ligado")

    def _send_parts(self):
        while True:
            req = self.pulls.get()
            if not self.offering or req.get("sha256") != self.offering[0]["sha256"]:
                continue
            info, path = self.offering
            first, count = int(req.get("from", 0)), min(int(req.get("count", 8)), 32)
            with open(path, "rb") as fh:
                for i in range(max(0, first), first + count):
                    fh.seek(i * CHUNK)
                    data = fh.read(CHUNK)
                    if not data:
                        break
                    self.publish("part", {"sha256": info["sha256"], "i": i, "data": base64.b64encode(data).decode()})

    def serve(self, seconds=None):
        """Keep the lab brain available to the headset (the last notebook cell)."""
        end = time.time() + seconds if seconds else None
        self.log("[link] the lab brain stays awake for the headset; stop the cell to end")
        while end is None or time.time() < end:
            try:
                self.answer(self.pending.get(timeout=1.0))
            except queue.Empty:
                pass


def code_from_env():
    """MLG_LINK from Colab Secrets or the environment, if the headset was paired."""
    try:
        from google.colab import userdata
        return userdata.get("MLG_LINK")
    except Exception:
        return os.environ.get("MLG_LINK")

