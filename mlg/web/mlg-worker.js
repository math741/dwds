// The MLG's mind in a browser, off the page's thread: engine/mlg_*.c compiled to
// WebAssembly (web/build.sh) — the same router, experts, evidence and Resident
// Core as on the Quest. mlg-web.js sends it the organs and the questions; the
// core's bytes come back as they are made.
"use strict";

let X = null; // the engine's exports
const enc = new TextEncoder();
const dec = new TextDecoder();
let onByte = null;
let mind = 0;
const organs = { core: 0, know: 0, router: 0 };
const thoughts = new Map();

const heap = () => new Uint8Array(X.memory.buffer);
const view = () => new DataView(X.memory.buffer);

// what the engine asks of an operating system: a clock, and somewhere to print
const wasi = {
  fd_write(fd, iovs, n, written) {
    let total = 0, s = "";
    for (let i = 0; i < n; i++) {
      const p = view().getUint32(iovs + 8 * i, true), l = view().getUint32(iovs + 8 * i + 4, true);
      s += dec.decode(heap().subarray(p, p + l));
      total += l;
    }
    view().setUint32(written, total, true);
    if (s.trim()) console.log("[mlg]", s.trim());
    return 0;
  },
  clock_time_get(id, precision, out) {
    const ms = id === 0 ? Date.now() : performance.now();
    view().setBigUint64(out, BigInt(Math.round(ms * 1e6)), true);
    return 0;
  },
  fd_close: () => 0,
  fd_fdstat_get: () => 8, fd_fdstat_set_flags: () => 8, fd_prestat_get: () => 8, fd_prestat_dir_name: () => 8,
  fd_read: () => 8, fd_seek: () => 8, path_open: () => 44,
  proc_exit(code) { throw new Error("the engine exited with " + code); },
};

async function boot(url) {
  const imports = { wasi_snapshot_preview1: wasi, mlg: { on_byte: (b) => (onByte ? onByte(b) : 0) } };
  let r;
  try {
    r = await WebAssembly.instantiateStreaming(fetch(url), imports);
  } catch (e) { // a server that does not say application/wasm
    r = await WebAssembly.instantiate(await (await fetch(url)).arrayBuffer(), imports);
  }
  X = r.instance.exports;
  X._initialize();
}

function cstr(s) {
  const b = enc.encode(s);
  const p = X.malloc(b.length + 1);
  heap().set(b, p);
  heap()[p + b.length] = 0;
  return p;
}

function str(p, max = 1 << 20) {
  const h = heap();
  let e = p;
  while (h[e] && e - p < max) e++;
  return dec.decode(h.subarray(p, e));
}

function withStr(s, f) {
  const p = cstr(s);
  try { return f(p); } finally { X.free(p); }
}

function out(cap, f) {
  const p = X.malloc(cap);
  try { f(p, cap); return str(p, cap); } finally { X.free(p); }
}

async function sha256(bytes) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(d, (x) => x.toString(16).padStart(2, "0")).join("");
}

// ---- organs -------------------------------------------------------------------------
// bytes go straight into the engine's memory, which then owns them (core, knowledge)
async function intoEngine(src, progress) {
  if (src instanceof ArrayBuffer) {
    const p = X.malloc(src.byteLength);
    heap().set(new Uint8Array(src), p);
    return [p, src.byteLength];
  }
  const { url, parts, bytes, version } = src;
  const p = X.malloc(bytes);
  if (!p) throw new Error("sem memória para " + url);
  let at = 0;
  for (let i = 0; i < (parts || 1); i++) {
    const r = await fetch((parts ? `${url}.${String(i).padStart(3, "0")}` : url) + (version ? `?v=${version}` : ""));
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    const reader = r.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      heap().set(value, p + at);
      at += value.length;
      progress(at, bytes);
    }
  }
  if (at !== bytes) throw new Error(`${url}: ${at} de ${bytes} bytes`);
  return [p, bytes];
}

function close(name) {
  if (!organs[name]) return;
  ({ core: X.mlg_core_close, know: X.mlg_know_close, router: X.mlg_router_close })[name](organs[name]);
  organs[name] = 0;
}

async function load({ name, src, sha }, reply) {
  const err = X.malloc(256);
  try {
    const [p, n] = await intoEngine(src, (got, total) => reply({ op: "download", name, got, total }));
    // nothing is trusted on faith: the bytes must hash to what vouched for them
    if (sha && (await sha256(heap().subarray(p, p + n))) !== sha) {
      X.free(p);
      return { ok: false, why: "não confere com o selo" };
    }
    close(name);
    if (name === "router") {
      organs.router = X.mlg_router_open_mem(p, n, err, 256);
      X.free(p);
    } else {
      organs[name] = (name === "core" ? X.mlg_core_open_mem : X.mlg_know_open_mem)(p, n, err, 256);
    }
    X.mlg_mind_organs(mind, organs.router, organs.know, organs.core);
    if (!organs[name]) return { ok: false, why: str(err) };
    const info = {};
    if (name === "router") {
      info.intents = [];
      for (let i = 0; i < X.mlg_router_count(organs.router); i++) info.intents.push(str(X.mlg_router_name(organs.router, i)));
    } else if (name === "know") {
      info.passages = X.mlg_know_count(organs.know);
    } else {
      const a = X.malloc(80);
      X.web_core_info(organs.core, a);
      const v = new Float64Array(X.memory.buffer, a, 9).slice();
      X.web_core_bench(organs.core, a);
      const b = new Float64Array(X.memory.buffer, a, 2).slice();
      X.free(a);
      Object.assign(info, { d_model: v[0], n_layers: v[1], n_experts: v[3], top_k: v[4], ffn_mult: v[5], ctx: v[6],
        params: v[8], prefill_ms: b[0], decode_ms: b[1] });
    }
    return { ok: true, ...info };
  } catch (e) {
    return { ok: false, why: String(e.message || e) };
  } finally {
    X.free(err);
  }
}

// the Q root again, on this body: the engine must reproduce what the lab measured
function qroot(text) {
  if (!organs.core) return NaN;
  const b = enc.encode(text);
  const p = X.malloc(b.length);
  heap().set(b, p);
  const bpb = X.mlg_core_bits_per_byte(organs.core, p, b.length);
  X.free(p);
  return bpb;
}

// ---- thinking -----------------------------------------------------------------------
function think({ id, text, unix, tz, body, lab }) {
  X.web_clock(mind, unix, tz);
  withStr(body, (p) => X.mlg_mind_body(mind, p));
  withStr(lab, (p) => X.mlg_mind_lab(mind, p));
  const t = X.web_thought_new();
  withStr(text, (p) => X.mlg_mind_think(mind, p, t));
  thoughts.set(id, t);
  return { thought: JSON.parse(out(16384, (p, cap) => X.mlg_mind_thought_json(mind, t, p, cap))),
    prompt: str(X.web_thought_prompt(t)), memory: memory() };
}

function speak({ id, seed }, reply) {
  const t = thoughts.get(id);
  const stream = new TextDecoder();
  let pending = "";
  onByte = (b) => {
    pending += stream.decode(new Uint8Array([b]), { stream: true });
    if (pending.length >= 8 || pending.endsWith(" ")) { reply({ op: "chunk", id, text: pending }); pending = ""; }
    return 0;
  };
  let spoke = 0;
  const text = out(4096, (p, cap) => { spoke = X.web_speak(mind, t, p, cap, seed); });
  onByte = null;
  return { text, spoke, organ: str(X.mlg_mind_spoke(mind)) };
}

function finish({ id, raw }) {
  const t = thoughts.get(id);
  const text = withStr(raw, (r) => out(4096, (p, cap) => X.mlg_mind_finish(mind, t, r, p, cap)));
  return { text, organ: str(X.mlg_mind_spoke(mind)) };
}

function memory() {
  return out(1 << 18, (p, cap) => X.mlg_mind_memory_dump(mind, p, cap));
}

// ---- the page's requests, one at a time ------------------------------------------------
let queue = Promise.resolve();
self.onmessage = (e) => {
  const m = e.data;
  const reply = (x) => self.postMessage(x);
  queue = queue.then(async () => {
    let r;
    try {
      switch (m.op) {
        case "boot":
          await boot(m.wasm);
          mind = withStr("", (p) => X.mlg_mind_new(0));
          if (m.memory) withStr(m.memory, (p) => X.mlg_mind_memory_load(mind, p));
          r = { ok: true };
          break;
        case "load": r = await load(m, reply); break;
        case "qroot": r = qroot(m.text); break;
        case "think": r = think(m); break;
        case "speak": r = speak(m, reply); break;
        case "finish": r = finish(m); break;
        case "free": X.free(thoughts.get(m.id)); thoughts.delete(m.id); r = true; break;
        default: throw new Error("unknown op " + m.op);
      }
      reply({ op: "done", seq: m.seq, r });
    } catch (err) {
      reply({ op: "done", seq: m.seq, error: String(err.message || err) });
    }
  });
};
