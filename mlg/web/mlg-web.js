// The MLG's body in a browser, on any device: what Mind.kt and MainActivity.kt
// are on the Quest. It gives the HUD (the headset's own hud.js, unchanged) the
// same window.MLG bridge, measures what this body can measure, keeps the trust
// chain and the memory, talks to the lab over the same sealed link (MQTT over
// WebSocket here), and asks the mind in mlg-worker.js (the headset's C, compiled
// to WebAssembly) what to answer.
"use strict";
(() => {
  const ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";
  // host, WebSocket URL: the same brokers, in the same order, as mlgsub/link.py and Link.kt
  const BROKERS = [["broker.emqx.io", "wss://broker.emqx.io:8084/mqtt"], ["broker.hivemq.com", "wss://broker.hivemq.com:8884/mqtt"],
    ["test.mosquitto.org", "wss://test.mosquitto.org:8081/mqtt"]];
  const ORGANS = "organs/";
  const enc = new TextEncoder();
  const dec = new TextDecoder();

  const store = {
    get(k, d = null) { try { const v = localStorage.getItem("mlg." + k); return v === null ? d : v; } catch { return d; } },
    set(k, v) { try { localStorage.setItem("mlg." + k, v); } catch { /* private window: this session only */ } },
  };

  // ---- events to the HUD (hud.js defines window.mlg) ---------------------------------
  const early = [];
  function emit(kind, data) {
    if (window.mlg) window.mlg.on(kind, data);
    else early.push([kind, data]);
  }
  const note = (text) => emit("note", { text });

  // ---- SHA-256, synchronous (the trust chain is read by the HUD synchronously) -------
  const K = new Uint32Array([0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb,
    0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f,
    0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]);
  function sha256(text) {
    const b = enc.encode(text);
    const n = ((b.length + 9 + 63) >> 6) << 6;
    const m = new Uint8Array(n);
    m.set(b);
    m[b.length] = 0x80;
    const dv = new DataView(m.buffer);
    dv.setUint32(n - 4, b.length * 8);
    dv.setUint32(n - 8, Math.floor(b.length / 0x20000000));
    const h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
    const w = new Uint32Array(64);
    const r = (x, k) => (x >>> k) | (x << (32 - k));
    for (let o = 0; o < n; o += 64) {
      for (let i = 0; i < 16; i++) w[i] = dv.getUint32(o + 4 * i);
      for (let i = 16; i < 64; i++) {
        const s0 = r(w[i - 15], 7) ^ r(w[i - 15], 18) ^ (w[i - 15] >>> 3), s1 = r(w[i - 2], 17) ^ r(w[i - 2], 19) ^ (w[i - 2] >>> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
      }
      let [a, bb, c, d, e, f, g, hh] = h;
      for (let i = 0; i < 64; i++) {
        const t1 = (hh + (r(e, 6) ^ r(e, 11) ^ r(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i]) >>> 0;
        const t2 = ((r(a, 2) ^ r(a, 13) ^ r(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c))) >>> 0;
        hh = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = bb; bb = a; a = (t1 + t2) >>> 0;
      }
      [a, bb, c, d, e, f, g, hh].forEach((x, i) => { h[i] = (h[i] + x) >>> 0; });
    }
    return Array.from(h, (x) => x.toString(16).padStart(8, "0")).join("");
  }
  const hex = (buf) => Array.from(new Uint8Array(buf), (x) => x.toString(16).padStart(2, "0")).join("");

  // ---- the trust core: an append-only log, each event chained to the one before -------
  const trust = {
    lines: (store.get("trust", "") || "").split("\n").filter(Boolean),
    get last() { try { return JSON.parse(this.lines[this.lines.length - 1]); } catch { return null; } },
    event(kind, detail = {}) {
      const last = this.last;
      const body = { seq: (last ? last.seq : 0) + 1, kind, detail, prev: last ? last.hash : "0".repeat(64), at: Date.now() };
      const hash = sha256(JSON.stringify(body));
      this.lines.push(JSON.stringify({ ...body, hash }));
      if (this.lines.length > 5000) this.lines.splice(0, this.lines.length - 5000);
      store.set("trust", this.lines.join("\n"));
    },
    verify() { // the first broken seq, or 0 when intact
      let prev = null;
      for (const line of this.lines) {
        const o = JSON.parse(line);
        const { hash, ...body } = o;
        if ((prev !== null && body.prev !== prev) || sha256(JSON.stringify(body)) !== hash) return o.seq;
        prev = hash;
      }
      return 0;
    },
  };

  // ---- the mind, in its worker -----------------------------------------------------------
  const worker = new Worker("mlg-worker.js");
  const calls = new Map();
  const chunkSinks = new Map();
  let seq = 0;
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.op === "done") {
      const c = calls.get(m.seq);
      calls.delete(m.seq);
      if (m.error) c.reject(new Error(m.error)); else c.resolve(m.r);
    } else if (m.op === "chunk") chunkSinks.get(m.id)?.(m.text);
    else if (m.op === "download") emit("download", { name: m.name === "know" ? "conhecimento" : m.name, got: m.got, total: m.total });
  };
  const call = (op, args = {}, transfer = []) => new Promise((resolve, reject) => {
    const s = ++seq;
    calls.set(s, { resolve, reject });
    worker.postMessage({ op, seq: s, ...args }, transfer);
  });

  // ---- the body: what a browser lets a page measure -------------------------------------
  const body = {};
  const organs = {};
  let manifest = null;
  function measure(name, value, unit, level) {
    body[name] = value;
    emit("trait", { name, value, unit, level });
  }
  function device() {
    const ua = navigator.userAgent;
    if (/OculusBrowser|Quest/.test(ua)) return "Meta Quest (navegador)";
    if (/iPhone|iPad/.test(ua)) return /iPad/.test(ua) ? "iPad" : "iPhone";
    if (/Android/.test(ua)) return "Android (" + ((ua.match(/Android [^;)]*;\s*([^;)]+)/) || [])[1] || "celular") + ")";
    const p = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || "";
    if (/Mac/.test(p)) return "Mac";
    return "computador " + (/Win/.test(p) ? "Windows" : /Linux/.test(p) ? "Linux" : p || "").trim();
  }
  async function sense() {
    measure("device", device(), "", "REPORTED");
    measure("cores", navigator.hardwareConcurrency || 1, "", "REPORTED");
    if (navigator.deviceMemory) measure("memory_total", navigator.deviceMemory * 2 ** 30, "B", "REPORTED");
    try {
      const b = await navigator.getBattery();
      const put = () => measure("battery", Math.round(b.level * 100), "%", "REPORTED");
      put();
      b.addEventListener("levelchange", put);
    } catch { /* this browser does not say */ }
  }
  function bodyFacts() {
    let s = "";
    for (const k of ["device", "cores", "battery", "temperature"]) if (k in body) s += `${k}=${body[k]}\n`;
    if (organs.core && organs.core.decode_ms > 0) s += `threads=1\nbytes_per_s=${Math.round(1000 / organs.core.decode_ms)}\n`;
    return s;
  }

  // ---- the voice: Web Speech, where the browser has it ------------------------------------
  const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
  const voice = {
    muted: store.get("muted") === "1",
    canSpeak: "speechSynthesis" in window,
    canListen: !!Rec,
    say(text) {
      if (this.muted || !this.canSpeak) return;
      const u = new SpeechSynthesisUtterance(text.replace(/[§|]/g, " "));
      u.lang = "pt-BR";
      const v = speechSynthesis.getVoices().find((x) => x.lang && x.lang.toLowerCase().startsWith("pt-br"));
      if (v) u.voice = v;
      u.onstart = () => emit("speaking", { v: "1" });
      u.onend = () => emit("speaking", { v: "0" });
      speechSynthesis.cancel();
      speechSynthesis.speak(u);
    },
    hush() { if (this.canSpeak) speechSynthesis.cancel(); },
    listen() {
      if (!Rec) { emit("heard_error", { v: "este navegador não reconhece voz: use o teclado" }); return; }
      const r = new Rec();
      r.lang = "pt-BR";
      r.interimResults = true;
      r.onstart = () => emit("listening", { v: "1" });
      r.onend = () => emit("listening", { v: "0" });
      r.onerror = (e) => emit("heard_error", { v: e.error });
      r.onresult = (e) => {
        const res = e.results[e.results.length - 1];
        emit(res.isFinal ? "heard" : "heard_partial", { v: res[0].transcript });
      };
      r.start();
    },
  };

  // ---- the live link: MQTT 3.1.1 over WebSocket, every payload AES-GCM sealed ------------
  const norm = (code) => code.toUpperCase().split("").filter((c) => ALPHABET.includes(c)).join("");
  function newCode() {
    const r = crypto.getRandomValues(new Uint8Array(16));
    return Array.from(r, (x) => ALPHABET[x % ALPHABET.length]).join("").match(/.{4}/g).join("-");
  }
  let code = store.get("code") || newCode();
  store.set("code", code);

  class Link {
    constructor(code, onMessage, onState) {
      this.code = code; this.onMessage = onMessage; this.onState = onState;
      this.connected = false; this.broker = ""; this.running = true; this.ws = null;
    }
    async start() {
      const c = norm(this.code);
      this.base = "mlg/" + hex(await crypto.subtle.digest("SHA-256", enc.encode("mlg-link-topic:" + c))).slice(0, 24);
      this.key = await crypto.subtle.importKey("raw", await crypto.subtle.digest("SHA-256", enc.encode("mlg-link-key:" + c)),
        "AES-GCM", false, ["encrypt", "decrypt"]);
      this.loop();
    }
    stop() { this.running = false; try { this.ws && this.ws.close(); } catch { /* gone */ } }
    // brokers are tried in the lab's order, always from the first, so both ends meet on the same one
    async loop() {
      let attempt = 0;
      while (this.running) {
        let online = false;
        for (const [host, url] of BROKERS) {
          if (!this.running) return;
          online = await this.session(host, url);
          if (online) break;
        }
        if (online) attempt = 0;
        else await new Promise((r) => setTimeout(r, Math.min(30000, 2000 << Math.min(attempt++, 4))));
      }
    }
    session(host, url) {
      return new Promise((resolve) => {
        let online = false, buf = new Uint8Array(0), ping = null;
        let ws;
        try { ws = new WebSocket(url, ["mqtt"]); } catch { resolve(false); return; }
        ws.binaryType = "arraybuffer";
        const timeout = setTimeout(() => { if (!online) ws.close(); }, 8000);
        ws.onopen = () => {
          const id = "mlg-web-" + hex(crypto.getRandomValues(new Uint8Array(4)));
          this.send(ws, 0x10, [...this.str("MQTT"), 4, 0x02, 0, 30, ...this.str(id)]);
        };
        ws.onmessage = (e) => {
          const more = new Uint8Array(e.data);
          const all = new Uint8Array(buf.length + more.length);
          all.set(buf); all.set(more, buf.length);
          buf = all;
          for (;;) { // whole packets only; a frame may hold several, or half of one
            let len = 0, mult = 1, i = 1;
            for (; i < buf.length; i++) { len += (buf[i] & 0x7f) * mult; mult *= 128; if (!(buf[i] & 0x80)) break; }
            if (i >= buf.length || buf.length < i + 1 + len) break;
            const type = buf[0], pkt = buf.slice(i + 1, i + 1 + len);
            buf = buf.slice(i + 1 + len);
            if (type === 0x20) {
              if (pkt[1] !== 0) { ws.close(); return; }
              this.send(ws, 0x82, [0, 1, ...this.str(this.base + "/#"), 0]);
              online = true; this.ws = ws; this.connected = true; this.broker = host;
              clearTimeout(timeout);
              ping = setInterval(() => this.send(ws, 0xc0, []), 20000);
              this.onState("online");
            } else if ((type & 0xf0) === 0x30) this.receive(type, pkt);
          }
        };
        ws.onerror = () => {};
        ws.onclose = () => {
          clearTimeout(timeout);
          if (ping) clearInterval(ping);
          if (this.ws === ws) { this.ws = null; this.connected = false; this.onState("offline"); }
          resolve(online);
        };
      });
    }
    str(s) { const b = typeof s === "string" ? enc.encode(s) : s; return [b.length >> 8, b.length & 0xff, ...b]; }
    send(ws, type, body) {
      if (!ws || ws.readyState !== 1) return;
      const head = [type];
      let n = body.length;
      do { let d = n % 128; n = Math.floor(n / 128); if (n > 0) d |= 0x80; head.push(d); } while (n > 0);
      const p = new Uint8Array(head.length + body.length);
      p.set(head); p.set(body, head.length);
      ws.send(p);
    }
    async publish(kind, obj) {
      if (!this.ws) return;
      const nonce = crypto.getRandomValues(new Uint8Array(12));
      const sealed = new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv: nonce }, this.key, enc.encode(JSON.stringify(obj))));
      const topic = this.str(`${this.base}/${kind}`);
      const body = new Uint8Array(topic.length + 12 + sealed.length);
      body.set(topic); body.set(nonce, topic.length); body.set(sealed, topic.length + 12);
      this.send(this.ws, 0x30, body);
    }
    async receive(type, pkt) {
      const tl = (pkt[0] << 8) | pkt[1];
      const topic = dec.decode(pkt.subarray(2, 2 + tl));
      const start = 2 + tl + (((type >> 1) & 3) ? 2 : 0);
      const kind = topic.slice(topic.lastIndexOf("/") + 1);
      if (["ask", "hello", "body", "pull"].includes(kind)) return; // our own messages, echoed
      try {
        const blob = pkt.subarray(start);
        const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv: blob.subarray(0, 12) }, this.key, blob.subarray(12));
        this.onMessage(kind, JSON.parse(dec.decode(plain)));
      } catch { /* not sealed with our code: ignore it */ }
    }
  }

  // ---- a Resident Core coming over the link (Transfer.kt, in JavaScript) --------------------
  class Transfer {
    constructor(offer, pull) {
      this.offer = offer; this.sha = offer.sha256; this.size = offer.bytes; this.chunk = offer.chunk;
      this.parts = Math.ceil(this.size / this.chunk);
      this.buf = new Uint8Array(this.size); this.have = new Uint8Array(this.parts);
      this.count = 0; this.windowEnd = 0; this.lastHeard = Date.now(); this.pull = pull;
    }
    firstMissing() { const i = this.have.indexOf(0); return i < 0 ? this.parts : i; }
    ask() {
      const from = this.firstMissing();
      if (from >= this.parts) return;
      this.windowEnd = Math.min(this.parts, from + 8);
      this.lastHeard = Date.now();
      this.pull(from, this.windowEnd - from);
    }
    part(i, data) {
      if (i < 0 || i >= this.parts || this.have[i]) return this.count === this.parts;
      const expect = i === this.parts - 1 ? this.size - i * this.chunk : this.chunk;
      if (data.length !== expect) return false;
      this.buf.set(data, i * this.chunk);
      this.have[i] = 1; this.count++; this.lastHeard = Date.now();
      if (this.count === this.parts) return true;
      if (this.firstMissing() >= this.windowEnd) this.ask();
      return false;
    }
    stalled(ms) { return this.count < this.parts && Date.now() - this.lastHeard > ms; }
  }

  // ---- organs: the site's (CI's base organs and the published brain) or the link's ---------
  async function cached(key) {
    try { const c = await caches.open("mlg"); const r = await c.match(key); return r ? await r.arrayBuffer() : null; } catch { return null; }
  }
  async function keep(key, bytes) {
    try { const c = await caches.open("mlg"); await c.put(key, new Response(bytes)); } catch { /* no storage: this session only */ }
  }
  function linkMeta() { try { return JSON.parse(store.get("core-link") || "null"); } catch { return null; } }

  async function loadOrgan(name, file) {
    const f = manifest && manifest.files && manifest.files[file];
    if (!f) return null;
    const r = await call("load", { name, sha: f.sha256,
      src: { url: ORGANS + file, parts: f.parts || 0, bytes: f.bytes, version: f.sha256.slice(0, 16) } });
    if (!r.ok) note(`${file} recusado: ${r.why}`);
    return r;
  }

  async function loadCore() {
    const link = linkMeta();
    const site = manifest && manifest.files && manifest.files["core.mlgb"] ? manifest : null;
    // the newest of the two, as on the headset (Brains.core)
    const fromLink = link && (!site || (link.version || "") > (site.version || ""));
    let r;
    if (fromLink) {
      const bytes = linkBytes ? linkBytes.slice(0).buffer : await cached("link/core.mlgb");
      if (!bytes) { store.set("core-link", ""); return loadCore(); }
      r = await call("load", { name: "core", sha: link.sha256, src: bytes }, [bytes]);
    } else if (site) {
      r = await loadOrgan("core", "core.mlgb");
    }
    if (!r || !r.ok) return;
    const meta = fromLink ? link : site;
    const q = await (await fetch("language.txt")).text().catch(() => "");
    const bpb = q ? await call("qroot", { text: q }) : NaN;
    const lab = meta.core && meta.core.qroot_bpb_deployed;
    measure("decode_ms", r.decode_ms, "ms/byte", "MEASURED");
    measure("prefill_ms", r.prefill_ms, "ms/byte", "MEASURED");
    measure("qroot_bpb", bpb, "bits/byte", "MEASURED");
    if (lab && Number.isFinite(bpb) && Math.abs(bpb - lab) > 0.05) {
      note(`Meu motor mediu ${bpb.toFixed(3)} bits/byte no Q root e o laboratório ${lab.toFixed(3)}: não confio, recuso o core`);
      trust.event("TRUST_FAIL", { qroot_bpb: bpb, lab });
      await call("load", { name: "core", sha: "", src: new ArrayBuffer(0) }); // closes it
      return;
    }
    organs.core = { ...r, meta };
    emit("organ", { name: "core", ok: true, params: r.params, shape: `d${r.d_model} L${r.n_layers} E${r.n_experts}k${r.top_k} ctx${r.ctx}`,
      brain: (meta.core && meta.core.brain) || "", qroot_bpb: bpb, via: fromLink ? "link" : "release" });
    trust.event("CORE_LOADED", { params: r.params, qroot_bpb: bpb });
    report();
  }

  async function loadOrgans() {
    emit("phase", { v: "organs" });
    try {
      const r = await fetch(ORGANS + "manifest.json", { cache: "no-cache" });
      manifest = r.ok ? await r.json() : null;
    } catch { manifest = null; }
    const router = await loadOrgan("router", "router.mlgr");
    if (router) emit("organ", { name: "router", ok: router.ok, intents: router.intents || [] });
    const know = await loadOrgan("know", "know.mlgk");
    if (know) emit("organ", { name: "knowledge", ok: know.ok, passages: know.passages || 0 });
    await loadCore();
    emit("phase", { v: "ready" });
  }

  // ---- the lab --------------------------------------------------------------------------
  let link = null, labBrain = null, labSealed = false, labTrain = null, transfer = null, linkBytes = null;
  const labReplies = new Map();

  function connect() {
    if (link) link.stop();
    link = new Link(code, fromLab, (state) => {
      emit("link", { state, broker: link ? link.broker : "" });
      if (state === "online") hello();
    });
    link.start();
  }
  function hello() {
    link.publish("hello", { device: body.device || "navegador", organs: { core: !!organs.core, knowledge: true, router: true } });
    report();
  }
  function report() {
    const c = organs.core;
    if (c && link) link.publish("body", { decode_ms: c.decode_ms, prefill_ms: c.prefill_ms, threads: 1, active_params: c.params,
      device: body.device, cores: body.cores });
  }
  function fromLab(kind, data) {
    if (kind === "mind") emit("mind", data);
    else if (kind === "train") {
      if ("steps" in data) labTrain = data;
      if (data.champion) { labBrain = data.champion; labSealed = !!data.sealed; }
      emit("train", data);
    } else if (kind === "say") labReplies.get(data.id)?.(data);
    else if (kind === "offer") receive(data);
    else if (kind === "part" && transfer && transfer.sha === data.sha256) {
      const bytes = Uint8Array.from(atob(data.data), (c) => c.charCodeAt(0));
      const whole = transfer.part(data.i, bytes);
      emit("download", { name: "core pelo link", got: transfer.count * transfer.chunk, total: transfer.size });
      if (whole) arrived(transfer);
    }
  }
  function receive(offer) {
    const sha = offer.sha256;
    const link_ = linkMeta();
    const site = manifest && manifest.files && manifest.files["core.mlgb"];
    if (!sha || (link_ && link_.sha256 === sha) || (site && site.sha256 === sha) || (transfer && transfer.sha === sha)) return;
    note(`O laboratório me ofereceu um Resident Core selado (${(offer.bytes / 2 ** 20).toFixed(1)} MiB). Recebendo pelo link…`);
    transfer = new Transfer(offer, (from, count) => link.publish("pull", { sha256: sha, from, count }));
    transfer.ask();
  }
  async function arrived(t) {
    transfer = null;
    if (hex(await crypto.subtle.digest("SHA-256", t.buf)) !== t.sha) {
      note("O core que chegou pelo link não confere com o hash selado: recusado");
      trust.event("TRUST_FAIL", { file: "core-link", sha256: t.sha });
      return;
    }
    linkBytes = t.buf;
    await keep("link/core.mlgb", t.buf);
    store.set("core-link", JSON.stringify(t.offer));
    trust.event("CORE_RECEIVED", { sha256: t.sha, via: "link" });
    note("Resident Core recebido pelo link e conferido. Vou medi-lo no meu motor antes de usar.");
    thinking = thinking.then(async () => {
      await loadCore();
      reply(organs.core ? "Meu Resident Core chegou e passou no selo. Agora quem conversa é ele." :
        "O core chegou, mas não passou na minha medição.", "self");
    });
  }
  setInterval(() => {
    if (!transfer || !link || !link.connected) return;
    if (transfer.stalled(120000)) transfer = null;
    else if (transfer.stalled(6000)) transfer.ask();
  }, 2000);

  function labStatus() {
    const t = labTrain;
    if (!link || !link.connected) return "Estou sem ligação com o laboratório agora. Quando o Colab rodar, eu vejo a evolução ao vivo.";
    if (!t && !labBrain) return "Estou ligada ao laboratório e esperando o Colab acordar. Abri a mente viva para você ver.";
    return "O laboratório está ligado. " + (labBrain ? `O campeão é ${labBrain.split(":").pop().slice(0, 12)}. ` : "") +
      (t && t.shape ? `Agora treinando ${t.shape}, passo ${t.steps}.` : "");
  }

  /** The lab brain's raw continuation and how much training it carries, streamed to the HUD as `chunks`. */
  function askLab(prompt, chunks) {
    return new Promise((resolve) => {
      const id = hex(crypto.getRandomValues(new Uint8Array(4)));
      let text = "", info = "", timer = null;
      const end = (v) => { clearTimeout(timer); labReplies.delete(id); resolve(v); };
      const wait = (s) => { clearTimeout(timer); timer = setTimeout(() => end(text ? { raw: text, info } : null), s * 1000); };
      labReplies.set(id, (m) => {
        if (m.err) return end(null);
        info = "maturing" in m ? `amadurecendo ${Math.round(m.maturing * 100)}%` : `${Math.max(1, Math.round((m.trained_s || 0) / 60))} min de treino`;
        if (m.chunk) { text += m.chunk; emit(chunks, { text: m.chunk }); }
        if (m.done) end({ raw: text, info });
        else wait(15);
      });
      emit("thinking", { organ: "lab" });
      link.publish("ask", { id, prompt, max: 360, temp: 0.6 });
      wait(6);
    });
  }

  // ---- thinking -----------------------------------------------------------------------------
  let memoryLines = store.get("memory", "");
  function reply(text, organ) {
    emit("say_done", { text, organ });
    voice.say(text);
    trust.event("REPLY", { organ, sha: sha256(text).slice(0, 16) });
  }
  let thinking = Promise.resolve();
  let tid = 0;
  async function think(text) {
    emit("user", { text });
    const id = ++tid;
    const r = await call("think", { id, text, unix: Date.now() / 1000, tz: -new Date().getTimezoneOffset(), body: bodyFacts(), lab: labStatus() });
    memoryLines = r.memory;
    store.set("memory", memoryLines);
    const th = r.thought;
    try {
      emit("route", { intent: th.intent, probs: th.probs || {} });
      trust.event("INTENT", { intent: th.intent, organ: th.organ });
      if (th.evidence) emit("evidence", { title: th.evidence_title.replace(/^§/, "").trim(), text: th.evidence, score: th.score });
      if (th.final === 1) {
        if (th.organ === "lab") emit("show", { view: "mind" });
        reply(th.reply, th.organ);
        return;
      }
      // a Resident Core speaks: the lab's when it passed the behaviour suite, this body's otherwise
      const online = labBrain && link && link.connected;
      if (online && labSealed) {
        const lab = await askLab(r.prompt, "say_chunk");
        if (lab) {
          const f = await call("finish", { id, raw: lab.raw });
          reply(f.text, f.organ === "core" ? "lab:" + lab.info : f.organ);
          return;
        }
      }
      emit("thinking", { organ: "core" });
      chunkSinks.set(id, (t) => emit("say_chunk", { text: t }));
      const sp = await call("speak", { id, seed: Math.floor(Math.random() * 2 ** 31) });
      chunkSinks.delete(id);
      reply(sp.text, sp.organ);
      // a brain still evolving does not answer, but you can watch it try
      if (online && !labSealed) askLab(r.prompt, "draft_chunk").then((d) => {
        if (d) emit("draft", { text: d.raw.trim().replace(/^M:/, "").trim().split("\n")[0].slice(0, 320), info: d.info });
      });
    } finally {
      call("free", { id });
    }
  }

  // ---- the bridge the HUD calls (MainActivity.Bridge on the Quest) ------------------------
  window.MLG = {
    ask(text) {
      if (!text || !text.trim()) return;
      thinking = thinking.then(() => think(text)).catch((e) => reply("Tive um erro pensando nisso: " + e.message, "trust"));
    },
    listen() { voice.listen(); },
    mute(on) { voice.muted = !!on; store.set("muted", on ? "1" : "0"); if (on) voice.hush(); },
    code() { return code; },
    pair(c) {
      const n = norm(c);
      if (n.length !== 16) return;
      code = n.match(/.{4}/g).join("-");
      store.set("code", code);
      connect();
    },
    memory() {
      return JSON.stringify(memoryLines.split("\n").filter(Boolean).map((l) => { try { return JSON.parse(l).v; } catch { return ""; } }).filter(Boolean));
    },
    trust() {
      const last = trust.last;
      return JSON.stringify({ events: last ? last.seq : 0, head: last ? last.hash : "0".repeat(64), broken: trust.verify() });
    },
    voice() { return JSON.stringify({ speak: voice.canSpeak, listen: voice.canListen }); },
  };

  // ---- awakening -----------------------------------------------------------------------------
  async function awaken() {
    for (const [k, d] of early.splice(0)) emit(k, d);
    emit("phase", { v: "awakening" });
    trust.event("AWAKENING", { note: "no architecture given; discovering body" });
    note("Acordei. Não vou confiar no nome deste corpo: vou medir.");
    emit("voice", { v: (voice.canSpeak ? "1" : "0") + (voice.canListen ? "1" : "0") });
    await sense();
    await call("boot", { wasm: "mlg.wasm", memory: memoryLines });
    connect();
    await loadOrgans();
    trust.event("BODY_KNOWN", body);
    const b = (organs.core && organs.core.meta.behaviour) || (manifest && manifest.behaviour);
    if (b) note(`Prova de comportamento: obrigatórias ${b.must_passed}/${b.must}` + (b.quality ? `, qualidade ${b.quality_passed}/${b.quality}.` : "."));
    reply(organs.core ? "Estou acordada. Medi meu corpo e meus órgãos passaram no selo."
      : "Estou acordada neste navegador, mas meu Resident Core ainda não chegou do laboratório. Já sei calcular, lembrar, " +
        "medir meu corpo e responder o que está no meu conhecimento.", "social");
    setInterval(() => { thinking = thinking.then(refresh); }, 20 * 60 * 1000);
  }
  async function refresh() { // new organs on the site: take them while awake
    try {
      const r = await fetch(ORGANS + "manifest.json", { cache: "no-cache" });
      const m = r.ok ? await r.json() : null;
      if (m && (!manifest || m.version !== manifest.version)) { await loadOrgans(); hello(); }
    } catch { /* offline: keep what is here */ }
  }
  if ("serviceWorker" in navigator && location.protocol === "https:") navigator.serviceWorker.register("sw.js").catch(() => {});
  window.addEventListener("DOMContentLoaded", () => {
    thinking = thinking.then(awaken).catch((e) => note("falha ao acordar: " + e.message));
  });
})();
