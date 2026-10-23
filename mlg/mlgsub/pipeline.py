"""The whole lab, stage by stage, as the Colab notebook runs it (MLG_Quest.ipynb).

Every stage leaves its result in `home` (Google Drive in Colab) and records
it in state.json, so a run that dies (Colab does) picks up where it stopped:
evolution is not repeated once it committed a champion, maturation resumes
from its last checkpoint, and the export always reflects the latest champion.

    run = Run(home, root, hours=3)
    run.link(); run.awaken(); run.feed(wiki_mb=1024); run.train_router()
    run.load_corpus(); run.evolve(); run.mature(); run.export(); run.publish(token)
"""
import json
import os
import shutil
import subprocess
import time

import torch

from . import brain as B, export, explain, know, mature as M, router as R
from .body import awaken, field_body
from .evolve import evolve as evolve_brains, Furnace
from .substrate import connect

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
CLI = os.path.join(os.path.dirname(__file__), "..", "engine", "mlg_cli")


class Run:
    def __init__(self, home, root=ROOT, hours=3.0, quick=False, scratch=None, log=print):
        self.home, self.root, self.hours, self.quick, self.log = home, os.path.abspath(root), hours, quick, log
        self.scratch = scratch or ("/content/mlg-work" if os.path.isdir("/content") else os.path.join(home, "work"))
        for d in (home, self.scratch, self.path("quest"), self.path("champions"), self.path("ckpt")):
            os.makedirs(d, exist_ok=True)
        os.environ["MLG_CHAMPIONS"] = self.path("champions")
        self.state = json.load(open(self.path("state.json"))) if os.path.exists(self.path("state.json")) else {}
        self.t0 = time.time()
        self.lnk = None
        self.sub = self.body = self.quest = self.corpus = self.furnace = None
        self.wiki_txt = self.rag = None
        self.router_acc = None

    def path(self, *p):
        return os.path.join(self.home, *p)

    def save(self, **kw):
        self.state.update(kw)
        with open(self.path("state.json") + ".tmp", "w") as fh:
            json.dump(self.state, fh, indent=1)
        os.replace(self.path("state.json") + ".tmp", self.path("state.json"))

    def left(self):
        """Hours of the budget still unspent in this session."""
        return self.hours - (time.time() - self.t0) / 3600

    # ---- I. the link and the awakening ---------------------------------------
    def link(self, code=None):
        from . import link as L

        code = code or L.code_from_env()
        if not code:
            code = self.state.get("link_code") or L.new_code()
            self.save(link_code=code)
            self.log("\n" + "=" * 64 + f"\n  CÓDIGO DE LIGAÇÃO:   {code}\n" + "=" * 64 +
                     "\n  No Quest: app MLG -> LIGAÇÃO -> digite este código.\n"
                     "  Para nunca mais digitar: Colab -> Secrets (chave à esquerda) ->\n"
                     "  MLG_LINK = o código que o app mostra.\n")
        try:
            self.lnk = L.Link(code, log=self.log)
        except Exception as e:
            self.log(f"[link] sem ligação nesta execução ({e}); o treino segue sem o Quest")
        return self.lnk

    def awaken(self):
        self.sub = connect(path=os.path.join(self.scratch, "substrate.db"), fresh=True)
        self.log(f"substrate: {self.sub.mode}")
        if self.lnk:
            self.lnk.attach(self.sub)
        self.body = awaken(self.sub, quick=self.quick, log=self.log)
        self.log("\n".join(explain.body_report(self.sub, self.body.id)))
        report = self.state.get("quest_body")
        if self.lnk:
            for _ in range(20):  # a headset that is on reports its body within seconds
                if self.lnk.field_report:
                    break
                time.sleep(1)
            if self.lnk.field_report:
                report = self.lnk.field_report
                self.save(quest_body=report)
        self.quest = field_body(self.sub, self.body, measured=report, log=self.log)
        return self.body

    # ---- II. feeding ---------------------------------------------------------------
    def wikipedia(self, mb):
        """Portuguese Wikipedia, title<TAB>text per line (paragraphs split by two spaces),
        kept in home so later runs do not download it again."""
        path = self.path(f"wiki-pt-{mb}.tsv")
        if os.path.exists(path) and os.path.getsize(path) >= mb * 2**20 * 0.95:
            return path
        from datasets import load_dataset

        self.log(f"[feeding] streaming {mb} MiB of Portuguese Wikipedia")
        stream = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
        n = 0
        with open(path + ".part", "w", encoding="utf-8") as fh:
            for row in stream:
                title = " ".join(row["title"].split())
                paras = [" ".join(p.split()) for p in row["text"].split("\n") if len(p.strip()) > 40]
                if not paras:
                    continue
                line = f"{title}\t{'  '.join(paras)}\n"
                fh.write(line)
                n += len(line.encode())
                if n >= mb * 2**20:
                    break
        os.replace(path + ".part", path)
        return path

    def feed(self, wiki_mb=1024):
        held = B.held_lines(self.root)
        wiki = []
        if wiki_mb:
            try:
                tsv = self.wikipedia(wiki_mb)
                self.wiki_txt = os.path.join(self.scratch, "wiki.txt")
                with open(tsv, encoding="utf-8") as src, open(self.wiki_txt, "w", encoding="utf-8") as dst:
                    for line in src:
                        title, _, text = line.rstrip("\n").partition("\t")
                        dst.write(f"{title}\n\n" + "\n".join(text.split("  ")) + "\n\n")
                wiki = list(know.wiki_passages(tsv))
            except Exception as e:
                self.log(f"[feeding] Wikipedia indisponível ({type(e).__name__}: {e}); só o corpus da MLG")
        # the index the Resident Core learns from leaves the sealed lines out; the one the
        # headset carries knows everything
        train_index = os.path.join(self.scratch, "know-train.mlgk")
        passages = know.build(self.root, train_index, B.Corpus.TRAIN, held, wiki, log=self.log)
        know.build(self.root, os.path.join(self.scratch, "know.mlgk"), B.Corpus.TRAIN, set(), wiki, log=self.log)
        self.rag = know.grounded_dialogs(train_index, passages, self.root, B.Corpus.TRAIN, held,
                                         os.path.join(self.scratch, "grounded.txt"),
                                         wiki_titles=2000 if self.quick else 200000, log=self.log)
        self.sub.obj("Dataset", "dataset:knowledge", origin="know.build", epistemic="KNOWN",
                     passages=len(passages), wikipedia_passages=len(wiki))
        self.sub.event("FED", "dataset:knowledge", passages=len(passages))
        del passages, wiki

    def train_router(self):
        import random

        qs = sorted({q for q, _ in know.dialog_pairs(self.root, ["core/data/corpus-dialog.txt"])})
        w, b, acc = R.train(R.examples(random.Random(0), per_intent=300 if self.quick else 1500,
                                       corpus_questions=qs), log=self.log)
        R.write(os.path.join(self.scratch, "router.mlgr"), w, b)
        self.router_acc = acc
        self.sub.obj("Brain", "router:intent", origin="router.train", epistemic="TESTED", kind="router",
                     intents=len(R.INTENTS), held_out_accuracy=acc)
        self.sub.event("ROUTER_TRAINED", "router:intent", accuracy=acc)

    def load_corpus(self):
        self.corpus = B.Corpus(self.root, self.body.device, extra=[self.wiki_txt] if self.wiki_txt else [],
                               own_extra=[self.rag] if self.rag else [],
                               max_train_bytes=2_000_000 if self.quick else None)
        self.log(f"{len(self.corpus.train) / 2**20:,.1f} MiB of training text ({self.corpus.own_bytes / 2**20:,.1f} MiB "
                 f"in the MLG's own voice) · {self.corpus.removed_lines} evaluation lines removed · "
                 f"Q root sealed: {self.corpus.qroot_sealed} · gate sealed: {self.corpus.gate_sealed}")

    # ---- III. evolution and maturation -----------------------------------------------
    def evolve(self, population=8, generations=5, seconds=75):
        self.furnace = Furnace(self.sub, self.body, self.corpus, self.quest, seconds, self.log)
        if self.lnk:
            self.lnk.attach(self.sub, self.furnace)
        tell = self.furnace.on_commit

        def committed(row):  # every champion is on disk and in state.json the moment it wins
            self.save(champion=row["checkpoint"], champion_brain=row["brain"])
            if tell:
                tell(row)
        self.furnace.on_commit = committed
        champ = self.state.get("champion")
        if champ and os.path.exists(champ):
            self.log("[resume] evolution already committed a champion in an earlier session")
            self.furnace.adopt(champ, log=self.log)
            if not self.state.get("evolved"):
                evolve_brains(self.sub, self.body, self.corpus, deploy=self.quest, population=population,
                              generations=1, seconds=seconds, log=self.log, furnace=self.furnace)
        else:
            evolve_brains(self.sub, self.body, self.corpus, deploy=self.quest, population=population,
                          generations=generations, seconds=seconds, log=self.log, furnace=self.furnace)
        self._keep_champion(evolved=True)
        self.log("\n".join(explain.why_champion(self.sub, self.quest.id)))

    def _keep_champion(self, **kw):
        c = self.furnace.champion
        if c and c.get("checkpoint"):
            self.save(champion=c["checkpoint"], champion_brain=c["brain"], **kw)

    def mature(self, reserve_hours=0.3):
        if self.state.get("matured"):
            self.log("[resume] maturation already finished in an earlier session")
            return
        g, hours, inh = self.state.get("mature_genome"), self.state.get("mature_hours"), None
        if g is None:  # planned once; a resumed session keeps growing the same brain
            hours = max(0.05, self.left() - reserve_hours)
            g, inh = M.plan(self.furnace, hours, log=self.log)
            self.save(mature_genome=g, mature_hours=hours)
        elif self.furnace.champion and self.furnace.champion["genome"] in self.furnace.states:
            weights, mode = B.inherit(g, self.furnace.champion["g"], self.furnace.states[self.furnace.champion["genome"]][1])
            if weights is not None:
                c = self.furnace.champion
                inh = (weights, mode, c["brain"], c["genome"])
        M.mature(self.furnace, g, hours, self.path("ckpt"), inheritance=inh,
                 checkpoint_every=120 if self.quick else 600, log=self.log)
        self._keep_champion(matured=True)
        self.log("\n".join(explain.why_champion(self.sub, self.quest.id)))

    # ---- IV. what goes to the headset ------------------------------------------------
    def export(self):
        c = self.furnace.champion
        out = self.path("quest")
        core = os.path.join(out, "core.mlgb")
        export.write(core, B.snapshot(c["model"]), c["g"], export.Q8)
        subprocess.run(["make", "-s", "-C", os.path.dirname(CLI)], check=True)
        engine_bpb = float(subprocess.run([CLI, "bpb", core, os.path.join(self.root, B.Corpus.QROOT)],
                                          capture_output=True, text=True, check=True).stdout)
        for name in ("know.mlgk", "router.mlgr"):
            shutil.copy(os.path.join(self.scratch, name), os.path.join(out, name))
        files = {n: {"sha256": export.sha256(os.path.join(out, n)), "bytes": os.path.getsize(os.path.join(out, n))}
                 for n in ("core.mlgb", "know.mlgk", "router.mlgr")}
        head = self.sub.anchor(self.path("chain.anchor"))
        manifest = {
            "version": time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
            "files": files,
            "core": {"brain": c["brain"], "genome": c["g"], "params": c["params"], "dev_bpb": c["dev_bpb"],
                     "gate_bpb": c.get("gate_bpb"), "qroot_bpb": c.get("qroot_bpb"),
                     "qroot_bpb_deployed": engine_bpb, "weights": "q8"},
            "router": {"intents": R.INTENTS, "held_out_accuracy": self.router_acc},
            "lab": {"body": self.body.name, "chain_head": head},
        }
        with open(os.path.join(out, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        self.sub.event("EXPORTED", c["brain"], version=manifest["version"], qroot_bpb_deployed=engine_bpb)
        self.log(f"[export] {c['brain']}: q8 core {files['core.mlgb']['bytes'] / 2**20:.1f} MiB, "
                 f"Q root {c.get('qroot_bpb', 0):.3f} bpb in the lab, {engine_bpb:.3f} bpb in the headset engine; "
                 f"knowledge {files['know.mlgk']['bytes'] / 2**20:.0f} MiB -> {out}")
        return manifest

    def publish(self, token, repo="math741/dwds", tag="mlg-brain"):
        """Upload the organs as assets of a GitHub release, where the app looks for them."""
        import requests

        if not token:
            self.log("[publish] sem GITHUB_TOKEN nos Secrets: os órgãos ficaram no Drive (MLG/quest). "
                     "Copie para o Quest em Android/data/ai.mlg.zero/files/mind, ou adicione o token e rode esta célula.")
            return None
        api, h = f"https://api.github.com/repos/{repo}", {"Authorization": f"Bearer {token}",
                                                          "Accept": "application/vnd.github+json"}
        r = requests.get(f"{api}/releases/tags/{tag}", headers=h)
        if r.status_code == 404:
            r = requests.post(f"{api}/releases", headers=h, json={
                "tag_name": tag, "name": "MLG · cérebro atual", "prerelease": True,
                "body": "Órgãos da MLG para o Quest, publicados pelo notebook. O app baixa daqui sozinho."})
        r.raise_for_status()
        rel = r.json()
        names = ["core.mlgb", "know.mlgk", "router.mlgr", "manifest.json"]  # the manifest last: it says the rest is ready
        for a in rel.get("assets", []):
            if a["name"] in names:
                requests.delete(f"{api}/releases/assets/{a['id']}", headers=h).raise_for_status()
        up = rel["upload_url"].split("{")[0]
        for n in names:
            with open(self.path("quest", n), "rb") as fh:
                requests.post(f"{up}?name={n}", headers={**h, "Content-Type": "application/octet-stream"},
                              data=fh).raise_for_status()
            self.log(f"[publish] {n} enviado")
        url = f"https://github.com/{repo}/releases/tag/{tag}"
        self.sub.event("PUBLISHED", self.furnace.champion["brain"], url=url)
        self.log(f"[publish] pronto: o app no Quest baixa sozinho na próxima vez que abrir ({url})")
        return url

    def ask(self, question, n=300):
        """What the headset does for a question about the world: retrieve, then let the champion speak."""
        c = self.furnace.champion
        hit = subprocess.run([CLI, "search", os.path.join(self.scratch, "know.mlgk"), question, "1"],
                             capture_output=True, text=True).stdout.strip()
        ctx = ""
        if hit.startswith("[") and float(hit[1:hit.index("]")]) >= 4:  # weak matches are not evidence
            ctx = hit.split(" | ", 1)[1]
        prompt = (f"C: {ctx}\n" if ctx else "") + f"U: {question}\nM:"
        out = B.generate(c["model"], c["g"], prompt, self.body.device, n=n, temperature=0.6, top_p=0.9, seed=0)
        return out[len(prompt):].split("\n")[0].strip()

    def serve(self, hours=1.0):
        if self.lnk and self.furnace and self.furnace.champion:
            self.lnk.commit(self.furnace.champion)
            self.lnk.serve(hours * 3600)
