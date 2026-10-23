/* The living mind: replays the substrate graph from FalkorDB in the order
 * the MLG created it. Physics by d3-force, everything drawn on one canvas.
 * Called by viz.py once d3 is loaded. With LIVE_URL it keeps polling that
 * URL and animates every object the MLG writes, as it writes it. */
function mlgMind(MIND_ID, MIND_DATA, MIND_SECONDS, MIND_HEIGHT, LIVE_URL) {
  const root = document.getElementById(MIND_ID);
  const cv = root.querySelector("canvas");
  const ctx = cv.getContext("2d");
  const tip = root.querySelector(".mind-tip");
  const ticker = root.querySelector(".mind-ticker");
  const hud = root.querySelector(".mind-hud");
  const bar = root.querySelector(".mind-progress > div");
  const H = MIND_HEIGHT || Math.max(480, window.innerHeight - 32);
  root.style.height = H + "px";
  let W = root.clientWidth, dpr = window.devicePixelRatio || 1;
  function resize() {
    W = root.clientWidth;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + "px"; cv.style.height = H + "px";
  }
  resize();
  window.addEventListener("resize", resize);

  const COL = {Body: "#ff6b35", Trait: "#ffb627", Experiment: "#8d99ae", Measurement: "#5c7aff",
    Genome: "#4cc9f0", Brain: "#2ee6a6", Evaluation: "#b388ff", Decision: "#ff4d6d",
    Suite: "#f72585", Dataset: "#adb5bd"};
  const SIZE = {Body: 17, Dataset: 11, Suite: 9, Trait: 6.5, Experiment: 5, Measurement: 2.4,
    Genome: 5.5, Brain: 8.5, Evaluation: 3.8, Decision: 5.5};
  const DEAD = "#4a5060", GOLD = "#ffd166";

  // glow sprites, one per colour
  const sprites = {};
  function sprite(c) {
    if (sprites[c]) return sprites[c];
    const s = document.createElement("canvas"); s.width = s.height = 96;
    const g = s.getContext("2d"), gr = g.createRadialGradient(48, 48, 0, 48, 48, 48);
    gr.addColorStop(0, c + "ff"); gr.addColorStop(0.18, c + "aa"); gr.addColorStop(0.45, c + "33");
    gr.addColorStop(1, c + "00");
    g.fillStyle = gr; g.fillRect(0, 0, 96, 96);
    return (sprites[c] = s);
  }

  // ---- data ----------------------------------------------------------
  const K = ["id", "type", "level", "label", "created", "reason", "stage", "dev_bpb", "value",
    "unit", "body", "pareto", "bpb", "params", "inherited"];
  const all = [], byId = new Map(), adj = new Map(), allEdges = [], edgeKeys = new Set();
  // new objects join the end of the timeline; known ones get their latest properties
  function addNodes(rows) {
    for (const r of rows) {
      const old = byId.get(r[0]);
      if (old) { K.forEach((k, j) => { if (r[j] !== null && r[j] !== undefined) old[k] = r[j]; }); continue; }
      const n = {}; K.forEach((k, j) => (n[k] = r[j])); n.order = all.length;
      all.push(n); byId.set(n.id, n); adj.set(n.id, []);
    }
  }
  // returns true when a relation appeared between two objects already on screen
  function addEdges(rows) {
    let fresh = false;
    for (const [a, r, b] of rows) {
      const key = a + "|" + r + "|" + b;
      if (edgeKeys.has(key) || !byId.has(a) || !byId.has(b)) continue;
      edgeKeys.add(key);
      const e = {a, r, b, phase: Math.random()};
      allEdges.push(e); adj.get(a).push(e); adj.get(b).push(e);
      if (byId.get(a).shown && byId.get(b).shown) {
        e.on = true; e.born = clock;
        links.push({source: a, target: b, r, phase: e.phase, born: clock});
        fresh = true;
      }
    }
    return fresh;
  }

  const nodes = [], links = [], effects = [], sparks = [], dust = [];
  for (let i = 0; i < 70; i++) dust.push({x: Math.random(), y: Math.random(),
    v: 0.002 + Math.random() * 0.006, r: Math.random() * 1.3 + 0.3, p: Math.random() * 6});
  const stats = {Trait: 0, Genome: 0, Brain: 0, killed: 0, commit: 0, rollback: 0};
  const champions = new Map(); // body -> brain node
  const bodies = [];

  const sim = d3.forceSimulation(nodes)
    .force("charge", d3.forceManyBody().strength((n) => -14 * (SIZE[n.type] || 4)).distanceMax(320))
    .force("link", d3.forceLink(links).id((n) => n.id)
      .distance((l) => l.target.type === "Measurement" || l.source.type === "Measurement" ? 16
        : l.r === "LIVES_IN" || l.r === "TRAINED_ON" ? 90 : l.r === "HAS_TRAIT" ? 42 : 34)
      .strength((l) => l.r === "TRAINED_ON" || l.r === "FED_BY" ? 0.05 : 0.35))
    .force("collide", d3.forceCollide((n) => (SIZE[n.type] || 4) + 2.5))
    .force("x", d3.forceX((n) => n.anchor ? n.anchor.x : 0).strength((n) => n.type === "Body" ? 0.12 : 0.012))
    .force("y", d3.forceY(0).strength((n) => n.type === "Body" ? 0.12 : 0.012))
    .alphaDecay(0.012).velocityDecay(0.32);
  sim.stop();

  // ---- replay --------------------------------------------------------
  let cursor = 0, play = true, speed = 1, last = performance.now(), clock = 0, finished = false;
  addNodes(MIND_DATA.nodes); addEdges(MIND_DATA.edges);
  const initial = all.length;
  const rate = Math.max(1, all.length) / Math.max(8, MIND_SECONDS); // nodes per second at 1x
  let accum = 0;

  function fmt(v, u) {
    if (typeof v !== "number") return String(v);
    if (u === "B") { const s = ["B", "KiB", "MiB", "GiB"]; let i = 0; while (v >= 1024 && i < 3) { v /= 1024; i++; } return v.toFixed(1) + " " + s[i]; }
    return (Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3)) + (u ? " " + u : "");
  }
  function say(text, color) {
    const d = document.createElement("div");
    d.className = "mind-line"; d.style.color = color; d.textContent = text;
    ticker.prepend(d);
    while (ticker.children.length > 4) ticker.lastChild.remove();
  }
  function narrate(n) {
    const c = COL[n.type];
    if (n.type === "Body") say(`body discovered: ${n.label} [${n.level}]`, c);
    else if (n.type === "Trait") say(`trait ${n.label} = ${fmt(n.value, n.unit)} [${n.level}]`, c);
    else if (n.type === "Genome") say(`genome ${n.id.slice(7, 13)} conceived` +
      (n.inherited ? ` · inherits its parent's mind (${n.inherited})` : ""), n.inherited ? GOLD : c);
    else if (n.type === "Suite") say(`ruler sealed: ${n.id.slice(6)}`, c);
    else if (n.type === "Dataset") say(`food: ${n.id.slice(8)}`, c);
    else if (n.type === "Decision") {
      const v = n.label, col = v === "COMMIT" ? GOLD : v === "SEARCH_DEATH" ? "#8d99ae" : COL.Decision;
      say(`${v} @ ${n.stage}: ${n.reason || ""}`, col);
    }
  }

  function spawn(n) {
    const sz = SIZE[n.type] || 4;
    n.r = sz; n.born = clock; n.alpha = 1;
    // appear next to something it is connected to
    let near = null;
    for (const e of adj.get(n.id)) {
      const o = byId.get(e.a === n.id ? e.b : e.a);
      if (o.shown) { near = o; break; }
    }
    if (n.type === "Body") {
      n.anchor = {x: bodies.length === 0 ? 0 : 260 * bodies.length};
      bodies.push(n);
      if (bodies.length === 2) bodies[0].anchor.x = -150, bodies[1].anchor.x = 150;
      n.x = n.anchor.x; n.y = (Math.random() - 0.5) * 40;
    } else if (near) {
      const a = Math.random() * Math.PI * 2;
      n.x = near.x + Math.cos(a) * 18; n.y = near.y + Math.sin(a) * 18;
      n.vx = Math.cos(a) * 2; n.vy = Math.sin(a) * 2;
    } else {
      n.x = (Math.random() - 0.5) * 120; n.y = (Math.random() - 0.5) * 120;
    }
    n.shown = true; nodes.push(n);
    for (const e of adj.get(n.id)) {
      const o = byId.get(e.a === n.id ? e.b : e.a);
      if (o.shown && !e.on) {
        e.on = true; e.born = clock;
        links.push({source: e.a, target: e.b, r: e.r, phase: e.phase, born: clock});
      }
    }
    if (n.type !== "Measurement" && n.type !== "Evaluation" && n.type !== "Experiment")
      effects.push({kind: "ring", node: n, t0: clock, color: COL[n.type], dur: n.type === "Body" ? 1.8 : 0.9});
    if (n.type in stats) stats[n.type]++;
    if (n.type === "Decision") decide(n);
    narrate(n);
  }

  function targets(d) {
    return adj.get(d.id).filter((e) => e.a === d.id && (e.r === "KILLED" || e.r === "CHOSE"))
      .map((e) => byId.get(e.b)).filter((x) => x.shown);
  }
  function decide(d) {
    const v = d.label;
    for (const t of targets(d)) {
      if (v === "KILL" || v === "ROLLBACK") {
        t.dead = true; t.deadAt = clock;
        stats[v === "KILL" ? "killed" : "rollback"]++;
        effects.push({kind: "shock", node: t, t0: clock, color: COL.Decision, dur: 1.1});
      } else if (v === "COMMIT") {
        const prev = champions.get(d.body);
        if (prev) { prev.champion = false; prev.retired = true; }
        champions.set(d.body, t);
        t.champion = true; t.crownAt = clock; stats.commit++;
        effects.push({kind: "burst", node: t, t0: clock, color: GOLD, dur: 1.6});
        for (let i = 0; i < 26; i++) {
          const a = Math.random() * Math.PI * 2, s = 40 + Math.random() * 110;
          sparks.push({x: t.x, y: t.y, vx: Math.cos(a) * s, vy: Math.sin(a) * s, t0: clock, life: 0.7 + Math.random() * 0.8});
        }
      }
    }
    if (v === "SEARCH_DEATH") effects.push({kind: "dim", t0: clock, dur: 1.4});
  }

  function step(dt) {
    if (!play) return;
    const backlog = all.length - cursor;
    if (backlog <= 0) {
      if (!finished) {
        finished = true;
        say(LIVE_URL ? "caught up. listening to FalkorDB for the next thought…"
          : "the mind is complete. drag, zoom, hover.", "#e9ecef");
      }
      accum = 0;
      return;
    }
    finished = false;
    // history plays at the chosen pace; live arrivals catch up within a few seconds
    const r = LIVE_URL && cursor >= initial ? Math.max(3, backlog / 2.5) : rate;
    accum += dt * r * speed;
    let added = false;
    while (accum >= 1 && cursor < all.length) { spawn(all[cursor++]); accum -= 1; added = true; }
    if (added) {
      sim.nodes(nodes); sim.force("link").links(links);
      sim.alpha(Math.max(sim.alpha(), 0.35));
    }
  }

  if (LIVE_URL) {
    const badge = root.querySelector(".mind-live");
    if (badge) badge.style.display = "inline-block";
    let busy = false;
    setInterval(() => {
      if (busy) return;
      busy = true;
      fetch(LIVE_URL, {cache: "no-store"}).then((r) => r.json()).then((d) => {
        addNodes(d.nodes);
        if (addEdges(d.edges)) { sim.force("link").links(links); sim.alpha(Math.max(sim.alpha(), 0.3)); }
        if (badge) badge.title = "last update " + new Date().toLocaleTimeString();
      }).catch(() => { if (badge) badge.title = "FalkorDB not reachable, retrying"; })
        .finally(() => { busy = false; });
    }, 2000);
  }

  // ---- camera --------------------------------------------------------
  let cam = {x: 0, y: 0, k: 1}, auto = true, transform = d3.zoomIdentity;
  function fit(dt) {
    if (!auto || !nodes.length) return;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const n of nodes) { x0 = Math.min(x0, n.x); x1 = Math.max(x1, n.x); y0 = Math.min(y0, n.y); y1 = Math.max(y1, n.y); }
    const k = Math.min(2.4, (W - 120) / Math.max(60, x1 - x0), (H - 170) / Math.max(60, y1 - y0));
    const f = 1 - Math.exp(-dt * 2.2);
    cam.k += (k - cam.k) * f; cam.x += ((x0 + x1) / 2 - cam.x) * f; cam.y += ((y0 + y1) / 2 - cam.y) * f;
    transform = d3.zoomIdentity.translate(W / 2 - cam.x * cam.k, H / 2 + 20 - cam.y * cam.k).scale(cam.k);
    d3.select(cv).property("__zoom", transform);
  }

  // ---- drawing -------------------------------------------------------
  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#03050b"; ctx.fillRect(0, 0, W, H);
    const vg = ctx.createRadialGradient(W / 2, H / 2, 40, W / 2, H / 2, Math.max(W, H) * 0.75);
    vg.addColorStop(0, "rgba(40,60,140,0.18)"); vg.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = vg; ctx.fillRect(0, 0, W, H);
    for (const p of dust) {
      p.y -= p.v * 0.016; if (p.y < 0) p.y = 1;
      ctx.globalAlpha = 0.25 + 0.25 * Math.sin(clock * 1.3 + p.p);
      ctx.fillStyle = "#9fb4ff"; ctx.beginPath();
      ctx.arc(p.x * W, p.y * H, p.r, 0, 6.283); ctx.fill();
    }
    ctx.globalAlpha = 1;

    ctx.save();
    ctx.translate(transform.x, transform.y); ctx.scale(transform.k, transform.k);
    const k = transform.k;

    // edges and the particles that run along them
    ctx.lineWidth = 0.7 / k;
    for (const l of links) {
      const s = l.source, t = l.target;
      if (s.x === undefined || t.x === undefined) continue;
      const grow = Math.min(1, (clock - l.born) / 0.5);
      const dead = s.dead || t.dead, heir = l.r === "INHERITED_FROM";
      ctx.globalAlpha = (heir ? 0.6 : dead ? 0.07 : 0.2) * grow;
      ctx.strokeStyle = heir ? GOLD : dead ? DEAD : COL[s.type];
      ctx.lineWidth = (heir ? 1.8 : 0.7) / k;
      ctx.beginPath(); ctx.moveTo(s.x, s.y);
      ctx.lineTo(s.x + (t.x - s.x) * grow, s.y + (t.y - s.y) * grow); ctx.stroke();
      if ((heir || !dead) && grow === 1 && s.type !== "Measurement" && t.type !== "Measurement") {
        // inheritance flows from the parent brain (target) into the child genome (source)
        const u = (clock * (heir ? 0.6 : 0.35) + l.phase) % 1, a = heir ? t : s, b = heir ? s : t;
        const px = a.x + (b.x - a.x) * u, py = a.y + (b.y - a.y) * u, z = heir ? 8 : 5;
        ctx.globalAlpha = 0.85 * Math.sin(u * Math.PI);
        ctx.drawImage(sprite(heir ? GOLD : COL[s.type]), px - z, py - z, 2 * z, 2 * z);
      }
    }

    // nodes
    for (const n of nodes) {
      const age = clock - n.born;
      const pop = age < 0.6 ? easeBack(age / 0.6) : 1;
      const breathe = 1 + 0.07 * Math.sin(clock * 2.2 + n.order);
      const r = n.r * pop * breathe;
      const fade = n.dead ? Math.max(0.28, 1 - (clock - n.deadAt) * 0.9) : 1;
      const col = n.dead && clock - n.deadAt > 0.6 ? DEAD : n.champion ? GOLD : COL[n.type];
      if (!n.dead || clock - n.deadAt < 0.6) {
        ctx.globalAlpha = (n.type === "Measurement" ? 0.5 : 0.75) * fade;
        const g = r * (n.champion ? 7 : n.type === "Body" ? 5 : 4.2);
        ctx.drawImage(sprite(col), n.x - g, n.y - g, g * 2, g * 2);
      }
      ctx.globalAlpha = fade;
      ctx.fillStyle = col; ctx.beginPath();
      if (n.type === "Decision") {
        ctx.moveTo(n.x, n.y - r); ctx.lineTo(n.x + r, n.y); ctx.lineTo(n.x, n.y + r); ctx.lineTo(n.x - r, n.y);
        ctx.closePath();
      } else if (n.type === "Body") {
        for (let i = 0; i < 6; i++) {
          const a = i * Math.PI / 3 + clock * 0.15;
          ctx[i ? "lineTo" : "moveTo"](n.x + Math.cos(a) * r, n.y + Math.sin(a) * r);
        }
        ctx.closePath();
      } else ctx.arc(n.x, n.y, r, 0, 6.283);
      ctx.fill();
      if (n.level === "REPORTED" || n.level === "ASSUMED" || n.level === "INFERRED") {
        ctx.setLineDash(n.level === "INFERRED" ? [1.5 / k, 1.5 / k] : [3 / k, 2.5 / k]);
        ctx.strokeStyle = "rgba(255,255,255,0.75)"; ctx.lineWidth = 0.8 / k;
        ctx.beginPath(); ctx.arc(n.x, n.y, r + 2.5, 0, 6.283); ctx.stroke(); ctx.setLineDash([]);
      }
      if (n.champion) {
        const ca = Math.min(1, (clock - n.crownAt) / 0.8);
        ctx.strokeStyle = GOLD; ctx.lineWidth = 1.4 / k; ctx.globalAlpha = ca;
        ctx.setLineDash([5 / k, 4 / k]); ctx.lineDashOffset = -clock * 18 / k;
        ctx.beginPath(); ctx.arc(n.x, n.y, r + 7, 0, 6.283); ctx.stroke(); ctx.setLineDash([]);
        for (let i = 0; i < 3; i++) {
          const a = clock * 1.6 + i * 2.094;
          ctx.drawImage(sprite(GOLD), n.x + Math.cos(a) * (r + 11) - 4, n.y + Math.sin(a) * (r + 11) - 4, 8, 8);
        }
      }
      if (n.retired) {
        ctx.globalAlpha = 0.6; ctx.strokeStyle = "#cfd8e3"; ctx.lineWidth = 0.8 / k;
        ctx.beginPath(); ctx.arc(n.x, n.y, r + 5, 0, 6.283); ctx.stroke();
      }
    }

    // effects
    for (let i = effects.length - 1; i >= 0; i--) {
      const e = effects[i], u = (clock - e.t0) / e.dur;
      if (u >= 1) { effects.splice(i, 1); continue; }
      if (e.kind === "dim") continue;
      const n = e.node;
      ctx.globalAlpha = (1 - u) * (e.kind === "ring" ? 0.8 : 1);
      ctx.strokeStyle = e.color;
      ctx.lineWidth = (e.kind === "ring" ? 1.4 : 2.6 * (1 - u) + 0.4) / k;
      const R = n.r + (e.kind === "ring" ? 4 + 34 * easeOut(u) : e.kind === "shock" ? 6 + 70 * easeOut(u) : 8 + 120 * easeOut(u));
      ctx.beginPath(); ctx.arc(n.x, n.y, R, 0, 6.283); ctx.stroke();
      if (e.kind === "shock") {
        ctx.globalAlpha = (1 - u) * 0.35; ctx.fillStyle = e.color;
        ctx.beginPath(); ctx.arc(n.x, n.y, R * 0.6, 0, 6.283); ctx.fill();
      }
    }
    for (let i = sparks.length - 1; i >= 0; i--) {
      const s = sparks[i], u = (clock - s.t0) / s.life;
      if (u >= 1) { sparks.splice(i, 1); continue; }
      const t = clock - s.t0;
      const x = s.x + s.vx * t * (1 - u * 0.5), y = s.y + s.vy * t * (1 - u * 0.5) + 30 * t * t;
      ctx.globalAlpha = 1 - u;
      ctx.drawImage(sprite(GOLD), x - 4, y - 4, 8, 8);
    }

    // labels
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const n of nodes) {
      const show = n.type === "Body" || n.type === "Suite" || n.type === "Dataset" || n.champion
        || n === hover || (n.type === "Trait" && k > 0.9) || (n.type === "Brain" && k > 1.3 && !n.dead);
      if (!show) continue;
      const fs = (n.type === "Body" ? 13 : n.champion ? 12 : 10) / k;
      ctx.font = `${n.type === "Body" || n.champion ? 600 : 400} ${fs}px "IBM Plex Sans", system-ui, sans-serif`;
      ctx.globalAlpha = Math.min(1, (clock - n.born) * 2) * (n.dead ? 0.45 : 0.92);
      ctx.fillStyle = n.champion ? GOLD : "#e9ecef";
      ctx.fillText(labelOf(n), n.x, n.y + n.r + 6 / k);
    }
    ctx.restore();

    const dim = effects.find((e) => e.kind === "dim");
    if (dim) {
      ctx.globalAlpha = 0.35 * Math.sin(Math.PI * (clock - dim.t0) / dim.dur);
      ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
    }
    ctx.globalAlpha = 1;
  }

  function labelOf(n) {
    if (n.type === "Body") return n.label;
    if (n.type === "Trait") return `${n.label} ${fmt(n.value, n.unit)}`;
    if (n.type === "Brain") return n.champion ? `CHAMPION ${n.dev_bpb ? n.dev_bpb.toFixed(3) + " bpb" : ""}` :
      n.dev_bpb ? n.dev_bpb.toFixed(3) + " bpb" : n.id.slice(6, 12);
    if (n.type === "Suite") return n.id.slice(6);
    if (n.type === "Dataset") return n.id.slice(8);
    return n.label;
  }
  const easeOut = (u) => 1 - Math.pow(1 - u, 3);
  const easeBack = (u) => { const c = 1.9; return 1 + (c + 1) * Math.pow(u - 1, 3) + c * Math.pow(u - 1, 2); };

  function drawHud() {
    const alive = stats.Brain - stats.killed - stats.rollback;
    hud.innerHTML = [
      ["traits", stats.Trait, COL.Trait], ["genomes", stats.Genome, COL.Genome],
      ["brains", stats.Brain, COL.Brain], ["killed", stats.killed + stats.rollback, COL.Decision],
      ["committed", stats.commit, GOLD]]
      .map(([k, v, c]) => `<div><b style="color:${c}">${v}</b><span>${k}</span></div>`).join("");
    bar.style.width = (100 * cursor / Math.max(1, all.length)) + "%";
  }

  // ---- interaction ---------------------------------------------------
  let hover = null;
  function nodeAt(px, py) {
    const [x, y] = transform.invert([px, py]);
    let best = null, bd = Infinity;
    for (const n of nodes) {
      const d = Math.hypot(n.x - x, n.y - y);
      if (d < Math.max(n.r + 4, 9 / transform.k) && d < bd) { best = n; bd = d; }
    }
    return best;
  }
  cv.addEventListener("mousemove", (ev) => {
    const b = cv.getBoundingClientRect();
    hover = nodeAt(ev.clientX - b.left, ev.clientY - b.top);
    if (hover) {
      const n = hover, rel = adj.get(n.id).filter((e) => e.on).slice(0, 6)
        .map((e) => e.a === n.id ? `${e.r} → ${labelOf(byId.get(e.b))}` : `${labelOf(byId.get(e.a))} → ${e.r}`);
      tip.innerHTML = `<b style="color:${COL[n.type]}">${n.type}</b> <span>${n.level || ""}</span><br>${n.id}` +
        (n.reason ? `<br><i>${n.reason}</i>` : "") + (n.value !== null && n.value !== undefined && n.type !== "Decision" ? `<br>${fmt(n.value, n.unit)}` : "") +
        (n.dev_bpb ? `<br>dev ${n.dev_bpb.toFixed(3)} bits/byte` : "") + (n.bpb ? `<br>${n.bpb.toFixed(3)} bits/byte` : "") +
        (rel.length ? `<hr>${rel.join("<br>")}` : "");
      tip.style.display = "block";
      tip.style.left = Math.min(W - 280, ev.clientX - b.left + 14) + "px";
      tip.style.top = Math.min(H - 150, ev.clientY - b.top + 14) + "px";
      cv.style.cursor = "pointer";
    } else { tip.style.display = "none"; cv.style.cursor = "grab"; }
  });
  cv.addEventListener("mouseleave", () => { hover = null; tip.style.display = "none"; });
  const sel = d3.select(cv);
  // the drag subject lives in screen space; the node it holds lives in the world
  sel.call(d3.drag().container(cv)
    .subject((ev) => { const n = nodeAt(ev.x, ev.y); return n ? {n, x: ev.x, y: ev.y} : null; })
    .on("start", (ev) => { sim.alphaTarget(0.25); ev.subject.n.fx = ev.subject.n.x; ev.subject.n.fy = ev.subject.n.y; })
    .on("drag", (ev) => { const [x, y] = transform.invert([ev.x, ev.y]); ev.subject.n.fx = x; ev.subject.n.fy = y; })
    .on("end", (ev) => { sim.alphaTarget(0); ev.subject.n.fx = null; ev.subject.n.fy = null; }));
  sel.call(d3.zoom().scaleExtent([0.15, 8])
    .on("zoom", (ev) => { if (ev.sourceEvent) auto = false; transform = ev.transform; }));

  const btn = (s) => root.querySelector(s);
  btn(".mind-play").onclick = function () { play = !play; this.textContent = play ? "❚❚ pause" : "▶ play"; };
  root.querySelectorAll(".mind-speed").forEach((b) => b.onclick = function () {
    speed = +this.dataset.s;
    root.querySelectorAll(".mind-speed").forEach((x) => x.classList.toggle("on", x === this));
  });
  btn(".mind-fit").onclick = () => { auto = true; };
  btn(".mind-all").onclick = () => { while (cursor < all.length) spawn(all[cursor++]); sim.nodes(nodes); sim.force("link").links(links); sim.alpha(0.6); };
  btn(".mind-replay").onclick = () => {
    for (const n of all) { n.shown = false; n.dead = n.champion = n.retired = false; delete n.fx; delete n.fy; }
    for (const e of allEdges) e.on = false;
    nodes.length = links.length = effects.length = sparks.length = bodies.length = 0;
    champions.clear(); Object.keys(stats).forEach((s) => (stats[s] = 0));
    cursor = 0; accum = 0; finished = false; auto = true; ticker.innerHTML = ""; play = true;
    btn(".mind-play").textContent = "❚❚ pause"; sim.nodes(nodes); sim.force("link").links(links);
  };

  // reduced motion: show the mind at once, no replay (live arrivals still appear)
  if (window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches) {
    btn(".mind-all").onclick();
    if (!LIVE_URL) { play = false; btn(".mind-play").textContent = "▶ play"; }
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now; clock += dt;
    step(dt);
    if (sim.alpha() > 0.002 || sim.alphaTarget() > 0) sim.tick();
    fit(dt); draw(); drawHud();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
