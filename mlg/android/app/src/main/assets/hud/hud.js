// The MLG HUD: the core and its organs, the living mind streamed from the lab,
// the dialogue and the body's telemetry. The Mind (Kotlin) calls mlg.on(kind, data);
// the page calls window.MLG (the Android bridge). Opened in a browser without
// the bridge, it plays a demo so the design can be seen anywhere.
import * as THREE from "./three.module.min.js";

const $ = (id) => document.getElementById(id);
const bridge = window.MLG || null;
const C = { teal: 0x19e3b1, orange: 0xff8a3d, gold: 0xffd166, red: 0xff4d5e, violet: 0x9d7bff, blue: 0x4c8dff };
const TYPE_COLOR = { Body: 0xff8a3d, Trait: 0xffb648, Genome: 0x4c8dff, Brain: 0x19e3b1, Evaluation: 0x9d7bff,
  Decision: 0xff4d5e, Suite: 0xd36bff, Dataset: 0x8a9aa0, Experiment: 0x6f7f86, Measurement: 0x56656b };
const ORGANS = [
  { id: "router", label: "ROTEADOR", color: C.gold },
  { id: "core", label: "CORE", color: C.teal },
  { id: "knowledge", label: "BUSCA", color: C.violet },
  { id: "procedural", label: "CÁLCULO", color: C.orange },
  { id: "memory", label: "MEMÓRIA", color: C.blue },
  { id: "body", label: "CORPO", color: 0xff6fae },
  { id: "lab", label: "LAB", color: 0x7cf0ff },
];
const ORGAN_OF = { math: "procedural", time: "procedural", remember: "memory", recall: "memory", body: "body",
  lab: "lab", knowledge: "knowledge", chat: "core", social: "procedural" };

// ---- scene -------------------------------------------------------------------
const canvas = $("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
renderer.setClearColor(0x05070a, 1);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05070a, 0.018);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 400);
camera.position.set(0, 2.2, 15);

function glowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  r.addColorStop(0, "rgba(255,255,255,1)");
  r.addColorStop(0.2, "rgba(255,255,255,0.55)");
  r.addColorStop(0.5, "rgba(255,255,255,0.12)");
  r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}
const GLOW = glowTexture();
function glow(color, size, opacity = 1) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: GLOW, color, transparent: true, opacity,
    blending: THREE.AdditiveBlending, depthWrite: false }));
  s.scale.setScalar(size);
  return s;
}
function label(text, color = "#cffaf0", size = 0.55) {
  const c = document.createElement("canvas");
  const g = c.getContext("2d");
  g.font = "600 44px monospace";
  c.width = Math.ceil(g.measureText(text).width) + 24;
  c.height = 64;
  g.font = "600 44px monospace";
  g.fillStyle = color;
  g.shadowColor = color;
  g.shadowBlur = 12;
  g.fillText(text, 12, 46);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  s.scale.set(size * c.width / 64, size, 1);
  return s;
}

// the core: a nucleus inside a lattice, orbits, and the organs on a ring
const coreGroup = new THREE.Group();
scene.add(coreGroup);
const nucleus = glow(C.teal, 3.2);
coreGroup.add(nucleus);
const heart = glow(0xe8fff8, 1.1);
coreGroup.add(heart);
const lattice = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.IcosahedronGeometry(1.35, 1)),
  new THREE.LineBasicMaterial({ color: C.teal, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending }));
coreGroup.add(lattice);
const shell = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.DodecahedronGeometry(2.0, 0)),
  new THREE.LineBasicMaterial({ color: C.orange, transparent: true, opacity: 0.3, blending: THREE.AdditiveBlending }));
coreGroup.add(shell);
const rings = [];
for (const [r, tilt, color, op] of [[3.0, 1.2, C.teal, 0.5], [3.6, -0.5, C.orange, 0.35], [4.3, 0.25, C.gold, 0.25]]) {
  const ring = new THREE.Mesh(new THREE.TorusGeometry(r, 0.012, 6, 160),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, blending: THREE.AdditiveBlending }));
  ring.rotation.x = tilt;
  coreGroup.add(ring);
  rings.push(ring);
}
const organRing = new THREE.Group();
coreGroup.add(organRing);
const organNodes = {};
ORGANS.forEach((o, i) => {
  const a = (i / ORGANS.length) * Math.PI * 2;
  const g = new THREE.Group();
  g.position.set(Math.cos(a) * 4.7, Math.sin(a) * 0.8, Math.sin(a) * 4.7 * 0.35);
  const halo = glow(o.color, 1.1, 0.75);
  const dot = glow(0xffffff, 0.28);
  const tag = label(o.label, "#" + o.color.toString(16).padStart(6, "0"), 0.42);
  tag.position.y = -0.62;
  g.add(halo, dot, tag);
  organRing.add(g);
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), g.position]),
    new THREE.LineBasicMaterial({ color: o.color, transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending }));
  organRing.add(line);
  organNodes[o.id] = { group: g, halo, line, color: o.color, heat: 0, alive: o.id === "router" || o.id === "procedural" || o.id === "memory" || o.id === "body" };
});

// dust
const dust = new THREE.Points(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(
  Array.from({ length: 2400 }, () => (Math.random() - 0.5) * 90), 3)),
  new THREE.PointsMaterial({ color: 0x2b6f62, size: 0.06, transparent: true, opacity: 0.7 }));
scene.add(dust);
const grid = new THREE.GridHelper(80, 80, 0x0c3b33, 0x081f1b);
grid.position.y = -6;
scene.add(grid);

// pulses: something travelling from the nucleus to an organ
const pulses = [];
function pulse(organ) {
  const o = organNodes[organ];
  if (!o) return;
  const s = glow(o.color, 0.7);
  coreGroup.add(s);
  pulses.push({ s, to: o, t: 0 });
  o.heat = 1;
}

// ---- the living mind ----------------------------------------------------------
const mindGroup = new THREE.Group();
mindGroup.visible = false;
scene.add(mindGroup);
const mind = { nodes: new Map(), edges: [], lines: null, dirty: true };
const SHOWN_TYPES = new Set(["Body", "Trait", "Genome", "Brain", "Evaluation", "Decision", "Suite", "Dataset"]);
const MAX_NODES = 700;
const effects = [];

function mindNode(o) {
  if (!SHOWN_TYPES.has(o.type)) return null;
  let n = mind.nodes.get(o.id);
  const color = o.type === "Decision" ? (o.verdict === "COMMIT" || o.verdict === "RESTORE" ? C.gold :
    o.verdict === "ROLLBACK" ? 0xff9f43 : o.verdict === "MATURE" ? C.violet : C.red) : (TYPE_COLOR[o.type] || 0x888888);
  if (!n) {
    if (mind.nodes.size >= MAX_NODES) return null;
    const size = o.type === "Body" ? 1.6 : o.type === "Brain" ? 0.9 : o.type === "Decision" ? 0.8 : 0.5;
    const s = glow(color, size, 0.95);
    s.position.set((Math.random() - 0.5) * 6, (Math.random() - 0.5) * 6, (Math.random() - 0.5) * 6);
    mindGroup.add(s);
    n = { id: o.id, type: o.type, s, v: new THREE.Vector3(), born: performance.now(), size };
    mind.nodes.set(o.id, n);
    s.scale.setScalar(0.01);
    if (o.type === "Decision" && (o.verdict === "COMMIT" || o.verdict === "KILL" || o.verdict === "RESTORE")) burst(n, color, o.verdict !== "KILL");
  }
  n.s.material.color.setHex(color);
  n.meta = Object.assign(n.meta || {}, o);
  return n;
}

function burst(n, color, crown) {
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.38, 48),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, side: THREE.DoubleSide, blending: THREE.AdditiveBlending }));
  ring.position.copy(n.s.position);
  mindGroup.add(ring);
  effects.push({ m: ring, t: 0, follow: n, grow: crown ? 5 : 9 });
  if (crown) toast(n.meta?.verdict === "RESTORE" ? "CAMPEÃO RESTAURADO" : "COMMIT · novo campeão");
}

function mindEdge(e) {
  if (!mind.nodes.has(e.a) || !mind.nodes.has(e.b)) return;
  mind.edges.push({ a: e.a, b: e.b, gold: e.r === "INHERITED_FROM" || e.r === "SUPERSEDES" || e.r === "CHOSE" });
  mind.dirty = true;
}

function rebuildLines() {
  if (mind.lines) { mindGroup.remove(mind.lines); mind.lines.geometry.dispose(); }
  const pos = new Float32Array(mind.edges.length * 6), col = new Float32Array(mind.edges.length * 6);
  const teal = new THREE.Color(0x1d6f62), gold = new THREE.Color(C.gold);
  mind.edges.forEach((e, i) => {
    const c = e.gold ? gold : teal;
    col.set([c.r, c.g, c.b, c.r, c.g, c.b], i * 6);
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  mind.lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55,
    blending: THREE.AdditiveBlending, depthWrite: false }));
  mindGroup.add(mind.lines);
  mind.dirty = false;
}

function layout(dt) {
  const ns = [...mind.nodes.values()];
  const k = Math.min(dt, 0.05);
  for (let i = 0; i < ns.length; i++) {  // repulsion
    const a = ns[i].s.position;
    for (let j = i + 1; j < ns.length; j++) {
      const b = ns[j].s.position;
      let dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
      const d2 = dx * dx + dy * dy + dz * dz + 0.05;
      if (d2 > 36) continue;
      const f = 0.35 / d2;
      dx *= f; dy *= f; dz *= f;
      ns[i].v.x += dx; ns[i].v.y += dy; ns[i].v.z += dz;
      ns[j].v.x -= dx; ns[j].v.y -= dy; ns[j].v.z -= dz;
    }
  }
  for (const e of mind.edges) {  // springs
    const A = mind.nodes.get(e.a), B = mind.nodes.get(e.b);
    if (!A || !B) continue;
    const d = A.s.position.clone().sub(B.s.position);
    const f = (d.length() - 1.1) * 0.04;
    d.normalize().multiplyScalar(f);
    A.v.sub(d); B.v.add(d);
  }
  const now = performance.now();
  for (const n of ns) {
    n.v.addScaledVector(n.s.position, -0.02);  // gravity to the centre
    n.v.multiplyScalar(0.8);
    if (n.v.lengthSq() > 1) n.v.normalize();
    n.s.position.addScaledVector(n.v, k * 12);
    const age = (now - n.born) / 600;
    n.s.scale.setScalar(n.size * (age < 1 ? 0.01 + age * (1 + 0.6 * Math.sin(age * Math.PI)) : 1));
  }
  if (mind.dirty) rebuildLines();
  if (mind.lines) {
    const p = mind.lines.geometry.attributes.position.array;
    mind.edges.forEach((e, i) => {
      const A = mind.nodes.get(e.a).s.position, B = mind.nodes.get(e.b).s.position;
      p.set([A.x, A.y, A.z, B.x, B.y, B.z], i * 6);
    });
    mind.lines.geometry.attributes.position.needsUpdate = true;
  }
  $("mindcount").textContent = mind.nodes.size;
}

// ---- view, camera, animation ----------------------------------------------------
let view = "core";
function setView(v) {
  view = v;
  coreGroup.visible = v === "core";
  mindGroup.visible = v === "mind";
  $("mindinfo").style.display = $("legend").style.display = v === "mind" ? "flex" : "none";
  $("mindinfo").style.display = v === "mind" ? "block" : "none";
  document.querySelectorAll(".tab[data-view]").forEach((t) => t.classList.toggle("on", t.dataset.view === v));
}
document.querySelectorAll(".tab[data-view]").forEach((t) => t.addEventListener("click", () => setView(t.dataset.view)));

let drag = null, yaw = 0, pitch = 0.12, camDist = 15;
canvas.addEventListener("pointerdown", (e) => (drag = [e.clientX, e.clientY]));
window.addEventListener("pointerup", () => (drag = null));
window.addEventListener("pointermove", (e) => {
  if (!drag) return;
  yaw -= (e.clientX - drag[0]) * 0.005;
  pitch = Math.max(-0.6, Math.min(0.9, pitch + (e.clientY - drag[1]) * 0.004));
  drag = [e.clientX, e.clientY];
});
canvas.addEventListener("wheel", (e) => (camDist = Math.max(7, Math.min(40, camDist + e.deltaY * 0.01))));

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  // the core sits between the two panels
  camera.setViewOffset(w, h, -w * 0.03, 0, w, h);
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

let thinking = 0, speaking = 0, level = 0, last = performance.now();
function frame(now) {
  const dt = (now - last) / 1000;
  last = now;
  const t = now / 1000;
  yaw += dt * (view === "mind" ? 0.06 : 0.025);
  const dist = view === "mind" ? camDist * 1.6 : camDist;
  camera.position.set(Math.sin(yaw) * dist * Math.cos(pitch), Math.sin(pitch) * dist + 1, Math.cos(yaw) * dist * Math.cos(pitch));
  camera.lookAt(0, 0, 0);
  const energy = Math.max(thinking, speaking * (0.6 + 0.4 * Math.sin(t * 18)), level);
  nucleus.scale.setScalar(3.2 + Math.sin(t * 2) * 0.15 + energy * 1.6);
  nucleus.material.color.setHex(thinking > 0.2 ? C.gold : C.teal);
  heart.scale.setScalar(1.1 + energy * 0.7);
  lattice.rotation.y += dt * (0.25 + energy * 2.5);
  lattice.rotation.x += dt * 0.11;
  shell.rotation.y -= dt * (0.12 + energy);
  shell.rotation.z += dt * 0.05;
  rings.forEach((r, i) => (r.rotation.z += dt * (0.15 + i * 0.07) * (i % 2 ? -1 : 1) * (1 + energy * 3)));
  organRing.rotation.y += dt * 0.04;
  for (const o of Object.values(organNodes)) {
    o.heat *= Math.pow(0.25, dt);
    o.halo.scale.setScalar(1.1 + o.heat * 1.6);
    o.halo.material.opacity = (o.alive ? 0.75 : 0.18) + o.heat * 0.25;
    o.line.material.opacity = 0.08 + o.heat * 0.7;
  }
  for (let i = pulses.length - 1; i >= 0; i--) {
    const p = pulses[i];
    p.t += dt * 1.8;
    p.s.position.copy(p.to.group.position).applyMatrix4(organRing.matrix).multiplyScalar(Math.min(p.t, 1));
    p.s.material.opacity = 1 - Math.max(0, p.t - 0.8) * 5;
    if (p.t > 1) { coreGroup.remove(p.s); pulses.splice(i, 1); }
  }
  thinking *= Math.pow(0.6, dt);
  level *= Math.pow(0.1, dt);
  if (mindGroup.visible || mind.nodes.size < 300) layout(dt);
  for (let i = effects.length - 1; i >= 0; i--) {
    const e = effects[i];
    e.t += dt;
    e.m.position.copy(e.follow.s.position);
    e.m.lookAt(camera.position);
    e.m.scale.setScalar(1 + e.t * e.grow);
    e.m.material.opacity = Math.max(0, 0.9 - e.t * 0.7);
    if (e.t > 1.3) { mindGroup.remove(e.m); effects.splice(i, 1); }
  }
  dust.rotation.y += dt * 0.004;
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---- panels --------------------------------------------------------------------
const log = $("log");
let streaming = null, drafting = null;
function addMsg(cls, who, text, organ) {
  const m = document.createElement("div");
  m.className = "msg " + cls;
  m.innerHTML = `<div class="who"></div><div class="text"></div>`;
  m.querySelector(".who").textContent = who;
  if (organ) {
    const o = document.createElement("span");
    o.className = "organ";
    o.textContent = organ;
    m.querySelector(".who").appendChild(o);
  }
  m.querySelector(".text").textContent = text;
  log.appendChild(m);
  while (log.children.length > 80) log.removeChild(log.firstChild);
  log.scrollTop = log.scrollHeight;
  return m;
}
const ORGAN_NAME = { self: "AUTOCONHECIMENTO", social: "SOCIAL", procedural: "CÁLCULO · EXATO", memory: "MEMÓRIA", body: "CORPO", lab: "LABORATÓRIO", core: "RESIDENT CORE",
  knowledge: "BUSCA", chat: "CORE", trust: "TRUST CORE" };
function organName(o) {
  if (!o) return "";
  if (o.startsWith("lab:")) return "CÉREBRO DO LAB · " + o.slice(4).toUpperCase();
  return ORGAN_NAME[o] || o.toUpperCase();
}

const kv = (rows) => rows.map(([k, v, lvl]) =>
  `<div class="kv"><span>${k}</span><span>${v}${lvl ? `<span class="lvl ${lvl}">${lvl}</span>` : ""}</span></div>`).join("");
const bodyRows = {};
const NAMES = { device: "dispositivo", cores: "núcleos", memory_total: "memória", memory_free: "livre", battery: "bateria",
  temperature: "temperatura", thermal_status: "térmico", threads: "threads", decode_ms: "pensamento", prefill_ms: "leitura",
  qroot_bpb: "Q root aqui" };
function fmt(name, v, unit) {
  if (unit === "B") return (v / 2 ** 30).toFixed(1) + " GiB";
  if (name === "decode_ms") return (1000 / v).toFixed(0) + " bytes/s";
  if (name === "prefill_ms") return (1000 / v).toFixed(0) + " bytes/s";
  if (name === "qroot_bpb") return Number(v).toFixed(3) + " bpb";
  if (typeof v === "number" && !Number.isInteger(v)) return v.toFixed(1) + (unit ? " " + unit : "");
  return v + (unit && unit !== "0-6" ? " " + unit : "");
}
function renderBody() {
  $("body").innerHTML = kv(Object.entries(bodyRows).map(([n, r]) => [NAMES[n] || n, fmt(n, r.value, r.unit), r.level]));
}
const organs = {};
function renderOrgans() {
  const rows = [];
  if (organs.core) rows.push(["Resident Core", organs.core.ok ? `${organs.core.shape} · ${(organs.core.params / 1e6).toFixed(1)}M${organs.core.via === "link" ? " · via link" : ""}` : "ausente"]);
  else rows.push(["Resident Core", "esperando o laboratório"]);
  rows.push(["Busca", organs.knowledge ? `${organs.knowledge.passages.toLocaleString("pt-BR")} trechos` : "ausente"]);
  rows.push(["Roteador", organs.router ? `${organs.router.intents.length} intenções` : "regras simples"]);
  rows.push(["Cálculo", "exato"], ["Memória", "persistente"]);
  $("organs").innerHTML = kv(rows);
}
const lab = { state: "offline", broker: "", champion: "", train: null };
// "~1h52", "~7 min": how long the lab's training has left
const eta = (sec) => { const m = Math.round(sec / 60), h = Math.floor(m / 60); return h ? `~${h}h${String(m % 60).padStart(2, "0")}` : `~${m} min`; };
function renderLab() {
  const t = lab.train || {};
  const maturing = t.generation === -1;
  $("lab").innerHTML = kv([
    ["ligação", `<span class="dot ${lab.state === "online" ? "on" : ""}"></span>${lab.state === "online" ? "online" : "offline"}`],
    ["broker", lab.broker || "—"],
    ["campeão", lab.champion ? lab.champion.replace("brain:", "").slice(0, 12) : "—"],
    ["treinando", t.shape ? `${t.shape} · ${maturing ? "amadurecendo" : "g" + t.generation}` : "—"],
    ["passo", t.steps ? `${t.steps} · ${t.loss_bpb ? t.loss_bpb.toFixed(3) + " bpb" : ""}` : "—"],
    ["progresso", t.steps ? `${((t.frac || 0) * 100).toFixed(0)}%${t.eta_s ? " · faltam " + eta(t.eta_s) : ""}` : "—"],
  ]);
  $("labmeter").style.width = ((t.frac || 0) * 100).toFixed(1) + "%";
  $("mindstate").textContent = lab.state !== "online" ? "esperando o laboratório" : !t.shape ? "ligada ao laboratório"
    : maturing ? `amadurecendo ${((t.frac || 0) * 100).toFixed(0)}%${t.eta_s ? " · faltam " + eta(t.eta_s) : ""}` : "evoluindo agora";
}
function renderTrust() {
  if (!bridge) { $("trust").innerHTML = kv([["cadeia", "demo"]]); return; }
  const t = JSON.parse(bridge.trust());
  $("trust").innerHTML = kv([["eventos", t.events], ["cadeia", t.broken ? `QUEBRADA em ${t.broken}` : "íntegra"],
    ["cabeça", t.head.slice(0, 16) + "…"]]);
}
function renderRoutes(probs, best) {
  $("routes").innerHTML = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([n, p]) =>
    `<div class="r ${n === best ? "best" : ""}"><span>${n}</span><i style="width:${Math.max(2, p * 170)}px"></i>${(p * 100).toFixed(0)}%</div>`).join("");
}
function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(toast.h);
  toast.h = setTimeout(() => t.classList.remove("show"), 2600);
}
setInterval(() => ($("clock").textContent = new Date().toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" })), 1000);

// ---- events from the Mind --------------------------------------------------------
const PHASE = { awakening: "DESPERTANDO", organs: "ÓRGÃOS", ready: "PRONTA" };
window.mlg = {
  on(kind, d) {
    switch (kind) {
      case "phase": $("phase").textContent = PHASE[d.v] || d.v; if (d.v === "ready") renderTrust(); break;
      case "note": $("topnote").textContent = d.text; addMsg("note", "MLG · NOTA", d.text); break;
      case "trait": bodyRows[d.name] = d; renderBody(); thinking = 0.4; break;
      case "bench": $("topnote").textContent = `medindo: ${d.threads} threads → ${(1000 / d.decode_ms).toFixed(0)} bytes/s`; thinking = 0.6; break;
      case "organ":
        organs[d.name] = d;
        if (organNodes[d.name]) organNodes[d.name].alive = !!d.ok;
        pulse(d.name);
        renderOrgans();
        break;
      case "download": $("topnote").textContent = `recebendo ${d.name}: ${(d.got / 2 ** 20).toFixed(0)} / ${(d.total / 2 ** 20).toFixed(0)} MiB`; break;
      case "user": addMsg("user", "VOCÊ", d.text); break;
      case "route":
        renderRoutes(d.probs, d.intent);
        pulse("router");
        setTimeout(() => pulse(ORGAN_OF[d.intent] || "core"), 250);
        thinking = 1;
        break;
      case "evidence": {
        const e = document.createElement("div");
        e.className = "evidence";
        e.innerHTML = `<b>EVIDÊNCIA</b> · <span></span>`;
        e.querySelector("span").textContent = `${d.title} — ${d.text.slice(0, 160)}…`;
        log.appendChild(e);
        pulse("knowledge");
        break;
      }
      case "thinking": thinking = 1; pulse(d.organ === "lab" ? "lab" : "core"); break;
      case "say_chunk":
        if (!streaming) { streaming = addMsg("", "MLG", ""); streaming.querySelector(".text").innerHTML = '<span class="t"></span><span class="cursor"></span>'; }
        streaming.querySelector(".t").textContent += d.text;
        log.scrollTop = log.scrollHeight;
        thinking = 1;
        break;
      case "say_done":
        if (streaming) streaming.remove();
        streaming = null;
        addMsg("", "MLG", d.text, organName(d.organ));
        speaking = 0.6;
        renderTrust();
        break;
      case "draft_chunk":
        if (!drafting) { drafting = addMsg("draft", "CÉREBRO EM EVOLUÇÃO · TENTANDO", ""); drafting.querySelector(".text").innerHTML = '<span class="t"></span><span class="cursor"></span>'; }
        drafting.querySelector(".t").textContent += d.text;
        log.scrollTop = log.scrollHeight;
        pulse("lab");
        break;
      case "draft":
        if (drafting) drafting.remove();
        drafting = null;
        if (d.text) addMsg("draft", "RASCUNHO DO CÉREBRO DO LAB", d.text, (d.info || "").toUpperCase());
        break;
      case "speaking": speaking = d.v === "1" ? 1 : 0; break;
      case "level": level = Math.min(1, Math.max(0, (parseFloat(d.v) + 2) / 12)); break;
      case "listening": $("mic").classList.toggle("live", d.v === "1"); break;
      case "heard_partial": $("input").value = d.v; break;
      case "heard": $("mic").classList.remove("live"); if (d.v) { $("input").value = ""; bridge?.ask(d.v); } break;
      case "heard_error": $("mic").classList.remove("live"); $("topnote").textContent = "voz: " + d.v; break;
      case "voice": {
        const [speak, listen] = d.v.split("");
        if (listen !== "1") $("input").placeholder = "este aparelho não tem reconhecimento de voz: use o teclado (a tecla de microfone dele dita)";
        if (speak !== "1") $("mute").textContent = "SEM VOZ";
        break;
      }
      case "link": lab.state = d.state; lab.broker = d.broker; organNodes.lab.alive = d.state === "online"; renderLab(); break;
      case "train":
        if (d.champion) { lab.champion = d.champion; toast("CAMPEÃO NO LAB · " + d.champion.replace("brain:", "").slice(0, 12)); }
        if (d.steps) lab.train = d;
        pulse("lab");
        renderLab();
        break;
      case "mind":
        for (const [k, o] of d.items) {
          if (k === "obj") mindNode(o);
          else if (k === "link") mindEdge(o);
        }
        pulse("lab");
        break;
      case "show": setView(d.view); break;
    }
  },
};

// ---- input ----------------------------------------------------------------------
function send() {
  const v = $("input").value.trim();
  if (!v) return;
  $("input").value = "";
  if (bridge) bridge.ask(v);
  else demoAsk(v);
}
$("send").addEventListener("click", send);
$("input").addEventListener("keydown", (e) => e.key === "Enter" && send());
$("mic").addEventListener("click", () => (bridge ? bridge.listen() : toast("voz só com a MLG acordada (app ou página)")));
let muted = false;
$("mute").addEventListener("click", () => {
  muted = !muted;
  bridge?.mute(muted);
  $("mute").textContent = muted ? "VOZ OFF" : "VOZ ON";
});
$("opensettings").addEventListener("click", () => {
  $("code").textContent = bridge ? bridge.code() : "DEMO-0000-0000-0000";
  const mem = bridge ? JSON.parse(bridge.memory()) : ["meu nome é Matheus"];
  $("memlist").innerHTML = mem.length ? mem.map((m) => `<div class="kv"><span>•</span><span></span></div>`).join("") : "<p>nada guardado ainda</p>";
  [...$("memlist").querySelectorAll(".kv span:last-child")].forEach((s, i) => (s.textContent = mem[i]));
  $("modal").style.display = "flex";
});
$("closemodal").addEventListener("click", () => ($("modal").style.display = "none"));
$("savecode").addEventListener("click", () => {
  const c = $("newcode").value.trim();
  if (c && bridge) { bridge.pair(c); toast("ligando com o novo código"); }
  $("modal").style.display = "none";
});
renderOrgans();
renderLab();
renderTrust();

// ---- demo, when opened outside the headset ------------------------------------
function demoAsk(text) {
  const on = window.mlg.on;
  on("user", { text });
  const intent = /\d|vezes|mais|menos/.test(text) ? "math" : /lembr|nome/.test(text) ? "remember" : "knowledge";
  on("route", { intent, probs: { [intent]: 0.91, chat: 0.05, knowledge: intent === "knowledge" ? 0.91 : 0.03, lab: 0.01 } });
  const answer = intent === "math" ? "9 × 9 = 81" : "A fotossíntese é o processo em que plantas usam luz, água e gás carbônico para produzir glicose e oxigênio.";
  if (intent === "knowledge") on("evidence", { title: "Fotossíntese", text: "A fotossíntese é um processo físico-químico realizado pelos seres vivos clorofilados…", score: 14.2 });
  let i = 0;
  const h = setInterval(() => {
    if (i < answer.length) on("say_chunk", { text: answer.slice(i, (i += 6)) });
    else { clearInterval(h); on("say_done", { text: answer, organ: intent === "math" ? "procedural" : "core" }); }
  }, 60);
}

if (!bridge) {
  const on = window.mlg.on;
  const steps = [
    () => on("phase", { v: "awakening" }),
    () => on("note", { text: "Acordei. Não vou confiar no nome deste corpo: vou medir." }),
    () => on("trait", { name: "device", value: "Oculus Quest 3S", unit: "", level: "REPORTED" }),
    () => on("trait", { name: "cores", value: 6, unit: "", level: "REPORTED" }),
    () => on("trait", { name: "memory_total", value: 7.6 * 2 ** 30, unit: "B", level: "REPORTED" }),
    () => on("trait", { name: "battery", value: 83, unit: "%", level: "REPORTED" }),
    () => on("trait", { name: "temperature", value: 31.5, unit: "C", level: "REPORTED" }),
    () => on("link", { state: "online", broker: "broker.emqx.io" }),
    () => on("organ", { name: "router", ok: true, intents: ["chat", "knowledge", "math", "time", "remember", "recall", "body", "lab"] }),
    () => on("organ", { name: "knowledge", ok: true, passages: 412833 }),
    () => on("bench", { threads: 4, prefill_ms: 0.21, decode_ms: 6.4 }),
    () => on("trait", { name: "threads", value: 4, unit: "", level: "MEASURED" }),
    () => on("trait", { name: "decode_ms", value: 6.4, unit: "ms/byte", level: "MEASURED" }),
    () => on("trait", { name: "qroot_bpb", value: 1.712, unit: "bits/byte", level: "MEASURED" }),
    () => on("organ", { name: "core", ok: true, params: 27.4e6, shape: "d512 L8 E1k1 ctx1024" }),
    () => on("phase", { v: "ready" }),
    () => on("say_done", { text: "Estou acordada, Matheus. Medi meu corpo e meus órgãos passaram no selo.", organ: "chat" }),
  ];
  steps.forEach((f, i) => setTimeout(f, 300 + i * 260));
  // a fake evolution streaming in, to see the living mind
  let g = 0;
  const genome = (id, gen) => ["obj", { id, type: "Genome", generation: gen }];
  setTimeout(() => {
    const items = [["obj", { id: "body:cuda0", type: "Body", name: "A100" }], ["obj", { id: "body:quest3s", type: "Body", name: "Quest" }],
      ["obj", { id: "suite:gate", type: "Suite" }], ["obj", { id: "suite:qroot", type: "Suite" }]];
    for (let i = 0; i < 8; i++) items.push(["obj", { id: "t" + i, type: "Trait" }], ["link", { a: i < 5 ? "body:cuda0" : "body:quest3s", r: "HAS_TRAIT", b: "t" + i }]);
    on("mind", { items });
  }, 1500);
  let prev = null;
  setInterval(() => {
    const id = "g" + g, b = "b" + g, ev = "e" + g, d = "d" + g;
    const items = [genome(id, g >> 2), ["obj", { id: b, type: "Brain" }], ["link", { a: id, r: "EXPRESSES", b }],
      ["link", { a: b, r: "TRAINED_ON", b: "body:cuda0" }], ["obj", { id: ev, type: "Evaluation" }], ["link", { a: b, r: "EVALUATED_BY", b: ev }],
      ["link", { a: ev, r: "ON_SUITE", b: "suite:gate" }]];
    if (prev) items.push(["link", { a: id, r: "INHERITED_FROM", b: prev }]);
    const verdict = g % 5 === 0 ? "COMMIT" : g % 3 === 0 ? "KILL" : g % 4 === 0 ? "ROLLBACK" : null;
    if (verdict) items.push(["obj", { id: d, type: "Decision", verdict }], ["link", { a: d, r: verdict === "COMMIT" ? "CHOSE" : "KILLED", b }]);
    on("mind", { items });
    on("train", { steps: 120 * (g + 1), frac: (g % 10) / 10, loss_bpb: 2.4 - g * 0.01, shape: "d384 L6 E4k1", generation: g >> 2 });
    if (verdict === "COMMIT") on("train", { champion: "brain:" + (1e11 + g * 7919).toString(16) });
    if (!verdict || verdict !== "KILL") prev = b;
    g++;
  }, 1400);
}
