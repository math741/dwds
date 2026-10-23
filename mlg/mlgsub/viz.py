"""Seeing the mind: the substrate graph rendered live from FalkorDB."""
import html
import json
import uuid

COLORS = {
    "Body": "#e8590c", "Experiment": "#868e96", "Measurement": "#ced4da",
    "Trait": "#f59f00", "Genome": "#4c6ef5", "Brain": "#12b886",
    "Evaluation": "#7950f2", "Decision": "#fa5252", "Suite": "#be4bdb",
    "Dataset": "#495057", "Event": "#adb5bd",
}
DEFAULT = ["Body", "Trait", "Genome", "Brain", "Decision", "Suite", "Dataset", "Evaluation"]


def graph_html(sub, types=DEFAULT, height=640, limit=1500):
    nodes = sub.q(
        "MATCH (n:Object) WHERE n.type IN $t "
        "RETURN n.id, n.type, n.epistemic, coalesce(n.verdict, n.name, n.id), n.pareto LIMIT $l",
        t=list(types), l=limit)
    edges = sub.q(
        "MATCH (a:Object)-[r]->(b:Object) WHERE a.type IN $t AND b.type IN $t "
        "RETURN a.id, type(r), b.id LIMIT $l", t=list(types), l=limit * 3)
    vn = []
    for nid, typ, level, label, pareto in nodes:
        short = str(label).split(":")[-1][:18]
        size = 26 if typ in ("Body", "Dataset") else 16 if typ in ("Brain", "Decision") else 11
        vn.append({"id": nid, "label": short, "group": typ, "size": size,
                   "title": f"{typ} | {nid} | {level}",
                   "color": COLORS.get(typ, "#999"),
                   "borderWidth": 4 if pareto else 1,
                   "font": {"size": 10, "color": "#ccc"}})
    ve = [{"from": a, "to": b, "label": r, "arrows": "to",
           "font": {"size": 7, "color": "#888", "strokeWidth": 0},
           "color": {"color": "#555"}} for a, r, b in edges]
    div = "mlg" + uuid.uuid4().hex[:8]
    legend = " ".join(
        f'<span style="color:{COLORS[t]}">&#9679; {t}</span>' for t in types if t in COLORS)
    return f"""
<div style="background:#111;padding:6px;border-radius:8px;font:12px sans-serif;color:#ccc">
  <div>{legend} &nbsp; <b>{len(vn)}</b> nodes, <b>{len(ve)}</b> edges, live from FalkorDB ({html.escape(sub.mode)})</div>
  <div id="{div}" style="height:{height}px"></div>
</div>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<script>
(function() {{
  var data = {{nodes: new vis.DataSet({json.dumps(vn)}), edges: new vis.DataSet({json.dumps(ve)})}};
  new vis.Network(document.getElementById("{div}"), data, {{
    physics: {{solver: "forceAtlas2Based", stabilization: {{iterations: 250}}}},
    interaction: {{hover: true, tooltipDelay: 80}}
  }});
}})();
</script>"""


def show(sub, **kw):
    from IPython.display import HTML, display

    display(HTML(graph_html(sub, **kw)))


MIND_TYPES = ["Body", "Experiment", "Measurement", "Trait", "Suite", "Dataset", "Genome",
              "Brain", "Evaluation", "Decision"]
D3 = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"
FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
         "&family=IBM+Plex+Sans:wght@400;600&display=swap")
_CSS = """
#ID{position:relative;width:100%;height:Hpx;background:#03050b;border-radius:14px;overflow:hidden;
  font-family:'IBM Plex Sans',system-ui,-apple-system,sans-serif;color:#e9ecef;user-select:none}
#ID canvas{display:block;width:100%;height:Hpx;cursor:grab}
#ID .mind-top{position:absolute;left:18px;top:14px;pointer-events:none}
#ID .mind-title{font-size:15px;font-weight:600;letter-spacing:.05em}
#ID .mind-sub{font-size:11px;color:#8d99ae;margin-top:3px}
#ID .mind-legend{position:absolute;left:18px;top:56px;font-size:10px;color:#8d99ae;display:flex;
  flex-direction:column;gap:4px;pointer-events:none}
#ID .mind-legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;
  box-shadow:0 0 8px currentColor}
#ID .mind-hud{position:absolute;right:18px;top:12px;display:flex;gap:18px;pointer-events:none}
#ID .mind-hud div{display:flex;flex-direction:column;align-items:flex-end}
#ID .mind-hud b{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums;
  text-shadow:0 0 14px currentColor}
#ID .mind-hud span{font-size:9px;color:#8d99ae;text-transform:uppercase;letter-spacing:.12em}
#ID .mind-ticker{position:absolute;left:18px;bottom:54px;font:12px 'IBM Plex Mono',ui-monospace,Menlo,monospace;
  pointer-events:none;max-width:78%}
#ID .mind-line{animation:mindin .45s cubic-bezier(.2,.9,.3,1.3);white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;text-shadow:0 0 10px currentColor;margin-top:3px}
#ID .mind-line:nth-child(2){opacity:.55}#ID .mind-line:nth-child(3){opacity:.3}
#ID .mind-line:nth-child(4){opacity:.14}
@keyframes mindin{from{opacity:0;transform:translateX(-14px)}to{opacity:1;transform:none}}
#ID .mind-bar{position:absolute;left:0;right:0;bottom:0;height:44px;display:flex;align-items:center;
  gap:8px;padding:0 14px;background:linear-gradient(transparent,rgba(0,0,0,.6))}
#ID button{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.14);color:#e9ecef;
  border-radius:8px;padding:5px 11px;font:11px 'IBM Plex Mono',ui-monospace,monospace;cursor:pointer;
  transition:all .15s}
#ID button:hover{background:rgba(255,255,255,.15);transform:translateY(-1px)}
#ID button.on{border-color:#ffd166;color:#ffd166;box-shadow:0 0 10px rgba(255,209,102,.35)}
#ID .mind-progress{flex:1;height:3px;background:rgba(255,255,255,.1);border-radius:2px;overflow:hidden}
#ID .mind-progress div{height:100%;width:0;background:linear-gradient(90deg,#4cc9f0,#2ee6a6,#ffd166);
  box-shadow:0 0 10px #2ee6a6;transition:width .2s linear}
#ID .mind-tip{position:absolute;display:none;max-width:280px;background:rgba(8,12,26,.92);
  border:1px solid rgba(255,255,255,.16);border-radius:10px;padding:9px 11px;font-size:11px;
  line-height:1.55;pointer-events:none;backdrop-filter:blur(8px);word-break:break-all}
#ID .mind-tip span{color:#8d99ae}#ID .mind-tip i{color:#ffb4c2;font-style:normal}
#ID .mind-tip hr{border:0;border-top:1px solid rgba(255,255,255,.12);margin:6px 0}
#ID button:focus-visible{outline:2px solid #4cc9f0;outline-offset:2px}
#ID .mind-live{display:none;margin-left:10px;padding:1px 7px;border:1px solid #ff4d6d;border-radius:6px;
  color:#ff4d6d;font-size:10px;letter-spacing:.14em;vertical-align:2px;pointer-events:auto;
  animation:mindpulse 1.4s ease-in-out infinite;text-shadow:0 0 8px #ff4d6d}
@keyframes mindpulse{50%{opacity:.35}}
@media (max-width:640px){#ID .mind-legend{display:none}
  #ID .mind-hud{left:18px;right:auto;top:52px;gap:12px}#ID .mind-hud b{font-size:16px}
  #ID .mind-bar{gap:5px;padding:0 8px}#ID button{padding:4px 7px}
  #ID .mind-speed[data-s="2"],#ID .mind-fit{display:none}
  #ID .mind-ticker{font-size:10px;max-width:92%}}
"""
_LEGEND = [("Body", "#ff6b35", "body"), ("Trait", "#ffb627", "trait of the body"),
           ("Measurement", "#5c7aff", "measurement"), ("Genome", "#4cc9f0", "brain genome"),
           ("Brain", "#2ee6a6", "trained brain"), ("Decision", "#ff4d6d", "furnace decision"),
           ("Suite", "#f72585", "frozen ruler"), ("Inherit", "#ffd166", "inheritance, champion")]


def mind_data(sub, types=MIND_TYPES):
    """Nodes in creation order and their relations, as read from FalkorDB."""
    nodes = sub.q(
        "MATCH (n:Object) WHERE n.type IN $t "
        "RETURN n.id, n.type, n.epistemic, coalesce(n.verdict, n.name, n.id), n.created, "
        "n.reason, n.stage, n.dev_bpb, n.value, n.unit, n.body, n.pareto, n.bits_per_byte, "
        "coalesce(n.params, n.pred_params), n.inherited ORDER BY n.created", t=list(types))
    edges = sub.q(
        "MATCH (a:Object)-[r]->(b:Object) WHERE a.type IN $t AND b.type IN $t "
        "RETURN a.id, type(r), b.id", t=list(types))
    return {"nodes": nodes, "edges": edges, "mode": sub.mode}


def mind_html(sub, seconds=45, height=720, types=MIND_TYPES, data=None, live_url=None):
    """The living mind: every node replayed in the order the MLG created it.
    Pass `data` (from mind_data) to render a saved mind without a database,
    `live_url` to keep polling for new objects, and height=0 to fill the window."""
    import os

    d = data or mind_data(sub, types)
    nodes, edges, mode = d["nodes"], d["edges"], d["mode"]
    data = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, default=str)
    data = data.replace("</", "<\\/")
    div = "mind" + uuid.uuid4().hex[:8]
    js = open(os.path.join(os.path.dirname(__file__), "mind.js"), encoding="utf-8").read()
    legend = "".join(f'<div><i style="color:{c};background:{c}"></i>{label}</div>'
                     for _, c, label in _LEGEND)
    return f"""<link rel="stylesheet" href="{FONTS}">
<div id="{div}">
<style>{_CSS.replace("#ID", "#" + div).replace("Hpx", f"{height}px" if height else "calc(100vh - 32px)")}</style>
<canvas></canvas>
<div class="mind-top"><div class="mind-title">MLG ZERO-1T &Sigma;&Omega; &middot; living mind<span class="mind-live">&#9679; LIVE</span></div>
<div class="mind-sub">{"live from" if live_url else "replayed from"} FalkorDB &middot; {len(nodes)} objects &middot; {len(edges)} relations &middot; {html.escape(mode)}</div></div>
<div class="mind-legend">{legend}</div>
<div class="mind-hud"></div><div class="mind-ticker"></div><div class="mind-tip"></div>
<div class="mind-bar"><button class="mind-play">&#10074;&#10074; pause</button>
<button class="mind-speed on" data-s="1">1x</button><button class="mind-speed" data-s="2">2x</button>
<button class="mind-speed" data-s="4">4x</button><div class="mind-progress"><div></div></div>
<button class="mind-all">show all</button><button class="mind-fit">fit</button>
<button class="mind-replay">&#8634; replay</button></div>
</div>
<script>
{js}
(function () {{
  var data = {data};
  function go() {{ mlgMind("{div}", data, {float(seconds)}, {int(height)}, {json.dumps(live_url)}); }}
  if (window.d3 && window.d3.forceSimulation) go();
  else {{ var s = document.createElement("script"); s.src = "{D3}"; s.onload = go; document.head.appendChild(s); }}
}})();
</script>"""


def mind(sub, **kw):
    """Display the living mind in a notebook."""
    from IPython.display import HTML, display

    display(HTML(mind_html(sub, **kw)))


_live = {}


def live(sub, port=8765, open_window=True):
    """Serve the living mind as its own full-window page that keeps reading
    FalkorDB while the MLG works: start it before evolve() and watch every
    genome, brain, KILL, COMMIT and inheritance appear as it is written.
    In Colab the page opens through the notebook's port proxy."""
    import http.server
    import threading

    _live["sub"] = sub
    if port not in _live:
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    s = _live["sub"]
                    if self.path.split("?")[0].endswith("/data"):
                        body = json.dumps(mind_data(s), ensure_ascii=False, default=str).encode()
                        ctype = "application/json"
                    else:
                        page = mind_html(s, seconds=20, height=0, live_url="data")
                        body = ("<!doctype html><meta charset=utf-8><meta name=viewport "
                                "content='width=device-width,initial-scale=1'><title>MLG Living Mind</title>"
                                "<body style='margin:0;padding:16px;background:#000207'>" + page).encode()
                        ctype = "text/html; charset=utf-8"
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as e:  # the page retries; never break the evolution
                    self.send_error(500, f"{type(e).__name__}: {e}")

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True, name="mlg-live").start()
        _live[port] = server
    try:
        from google.colab import output

        if open_window:
            output.serve_kernel_port_as_window(port, anchor_text="open the living mind (live)")
        url = output.eval_js(f"google.colab.kernel.proxyPort({port})")
        print(f"live mind: {url}")
    except ImportError:
        url = f"http://localhost:{port}/"
        print(f"live mind: {url}")
        if open_window:
            import webbrowser

            webbrowser.open(url)
    return url


def plot_evolution(furnace, title=None):
    """dev quality against cost, coloured by fate; the champion is a star."""
    import matplotlib.pyplot as plt

    rows = [r for r in furnace.results if r.get("dev_bpb") is not None]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for status, color in (("SURVIVED", "#12b886"), ("KILLED", "#fa5252")):
        pts = [r for r in rows if r["status"] == status]
        ax[0].scatter([r["params"] / 1e6 for r in pts], [r["dev_bpb"] for r in pts],
                      c=color, label=status.lower(), alpha=0.8)
        ax[1].scatter([r.get("cost", 0) for r in pts], [r["dev_bpb"] for r in pts],
                      c=color, alpha=0.8)
    ch = furnace.champion
    if ch:
        ax[0].scatter([ch["params"] / 1e6], [ch["dev_bpb"]], marker="*", s=380, c="#f59f00",
                      edgecolors="k", label="committed", zorder=5)
        ax[1].scatter([ch["cost"]], [ch["dev_bpb"]], marker="*", s=380, c="#f59f00",
                      edgecolors="k", zorder=5)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("parameters (millions, log)")
    ax[0].set_ylabel("dev bits per byte (lower is better)")
    ax[0].legend()
    energy = any(r.get("joules") for r in rows)
    ax[1].set_xlabel("training energy (J)" if energy else "training time (s), energy NOT MEASURED")
    ax[1].set_ylabel("dev bits per byte")
    fig.suptitle(title or f"Brains grown for {furnace.home.id}")
    fig.tight_layout()
    plt.show()
