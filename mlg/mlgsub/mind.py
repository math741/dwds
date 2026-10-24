"""The headset's mind, driven from the lab: engine/mlg_mind.c through ctypes.

The same code decides on the Quest, so what the behaviour suite sees here is
what the headset will do:

    m = Mind(core="quest/core.mlgb", know="quest/know.mlgk", router="quest/router.mlgr")
    m.think("quanto é 7 vezes 8")   -> {"organ": "procedural", "reply": "7 × 8 = 56", ...}
    m.ask("o que é fotossíntese")   -> the answer the headset would give, with its own core
"""
import ctypes
import json
import os
import subprocess
import time

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine")
LIB = os.path.join(ENGINE, "libmlg.so")


class Thought(ctypes.Structure):
    _fields_ = [("intent", ctypes.c_char * 16), ("probs", ctypes.c_float * 16), ("n_intents", ctypes.c_int),
                ("organ", ctypes.c_char * 16), ("final", ctypes.c_int), ("reply", ctypes.c_char * 2048),
                ("prompt", ctypes.c_char * 4096), ("evidence_title", ctypes.c_char * 256),
                ("evidence", ctypes.c_char * 1536), ("evidence_score", ctypes.c_float)]


_lib = None


def lib():
    global _lib
    if _lib is None:
        subprocess.run(["make", "-s", "-C", ENGINE, "libmlg.so"], check=True)
        L = ctypes.CDLL(LIB)
        P, S, I = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
        err = ctypes.c_char_p
        for name, args, res in [
                ("mlg_core_open", [S, err, I], P), ("mlg_core_close", [P], None),
                ("mlg_know_open", [S, err, I], P), ("mlg_know_close", [P], None),
                ("mlg_router_open", [S, err, I], P), ("mlg_router_close", [P], None),
                ("mlg_mind_new", [S], P), ("mlg_mind_free", [P], None),
                ("mlg_mind_organs", [P, P, P, P], None), ("mlg_mind_body", [P, S], None),
                ("mlg_mind_lab", [P, S], None), ("mlg_mind_clock", [P, ctypes.c_longlong, I], None),
                ("mlg_mind_think", [P, S, ctypes.POINTER(Thought)], I),
                ("mlg_mind_finish", [P, ctypes.POINTER(Thought), S, ctypes.c_char_p, I], None),
                ("mlg_mind_speak", [P, ctypes.POINTER(Thought), ctypes.c_char_p, I, P, P, ctypes.c_uint64], I),
                ("mlg_mind_thought_json", [P, ctypes.POINTER(Thought), ctypes.c_char_p, I], I),
                ("mlg_mind_memory_count", [P], I), ("mlg_set_threads", [I], None),
                ("mlg_mind_spoke", [P], ctypes.c_char_p)]:
            f = getattr(L, name)
            f.argtypes, f.restype = args, res
        _lib = L
    return _lib


class Mind:
    def __init__(self, core=None, know=None, router=None, memory=None, threads=0):
        L = lib()
        self.L = L
        err = ctypes.create_string_buffer(256)

        def open_(fn, path):
            if not path:
                return None
            h = fn(path.encode(), err, 256)
            if not h:
                raise RuntimeError(err.value.decode())
            return h
        self.core = open_(L.mlg_core_open, core)
        self.know = open_(L.mlg_know_open, know)
        self.router = open_(L.mlg_router_open, router)
        self.m = L.mlg_mind_new(memory.encode() if memory else None)
        L.mlg_mind_organs(self.m, self.router, self.know, self.core)
        if threads:
            L.mlg_set_threads(threads)

    def body(self, **facts):
        self.L.mlg_mind_body(self.m, "\n".join(f"{k}={v}" for k, v in facts.items()).encode())

    def lab(self, status):
        self.L.mlg_mind_lab(self.m, status.encode())

    def clock(self, unix=None, tz_minutes=-180):
        self.L.mlg_mind_clock(self.m, int(unix or time.time()), tz_minutes)

    def _think(self, text):
        t = Thought()
        self.L.mlg_mind_think(self.m, text.encode(), ctypes.byref(t))
        return t

    def think(self, text):
        t = self._think(text)
        buf = ctypes.create_string_buffer(16384)
        self.L.mlg_mind_thought_json(self.m, ctypes.byref(t), buf, len(buf))
        return json.loads(buf.value.decode("utf-8", "replace"))

    def finish(self, text, raw):
        """What the mind makes of a Resident Core's raw output for this question."""
        t = self._think(text)
        out = ctypes.create_string_buffer(4096)
        self.L.mlg_mind_finish(self.m, ctypes.byref(t), raw.encode(), out, len(out))
        return out.value.decode("utf-8", "replace")

    def ask(self, text, seed=7):
        """The answer this body gives on its own: who gave it (organ) and the reply."""
        t = self._think(text)
        out = ctypes.create_string_buffer(4096)
        self.L.mlg_mind_speak(self.m, ctypes.byref(t), out, len(out), None, None, seed)
        return self.L.mlg_mind_spoke(self.m).decode(), out.value.decode("utf-8", "replace")

    def close(self):
        L = self.L
        L.mlg_mind_free(self.m)
        for h, f in ((self.core, L.mlg_core_close), (self.know, L.mlg_know_close), (self.router, L.mlg_router_close)):
            if h:
                f(h)
