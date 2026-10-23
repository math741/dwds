"""MLG Cognitive Substrate on FalkorDB.

The manifesto already describes a graph: every MLG OBJECT has ID, TYPE,
VERSION, HASH, EVIDENCE, DEPENDENCIES and ORIGIN (IX); intelligence is
event-sourced (X); nothing is promoted without evidence (LXXII). Here that
graph is literal. Bodies, experiments, measurements, traits of the body,
brain genomes, trained brains, evaluations and decisions are nodes of one
FalkorDB graph, and every "WHY?" is a Cypher query over it.

Epistemic levels follow core/kg.c and are never mixed: a number the driver
REPORTED is not a number the MLG MEASURED, and a simulated body is ASSUMED.
"""
import hashlib
import json
import os
import time

# Labels and relation types are interpolated into Cypher, so they are
# restricted to this vocabulary; everything else goes through parameters.
TYPES = {
    "Body", "Experiment", "Measurement", "Trait", "Genome", "Brain",
    "Evaluation", "Decision", "Event", "Suite", "Dataset",
}
RELS = {
    "HAS_TRAIT", "PART_OF", "MEASURED_ON", "PROVEN_BY", "PARENT_OF",
    "EXPRESSES", "TRAINED_ON", "EVALUATED_BY", "ON_SUITE", "SHAPED_BY",
    "JUSTIFIED_BY", "CHOSE", "KILLED", "CHALLENGED", "SUPERSEDES",
    "ABOUT", "NEXT", "FED_BY", "LIVES_IN", "PREDICTED_BY", "INHERITED_FROM",
}
LEVELS = {
    "KNOWN", "OBSERVED", "MEASURED", "REPORTED", "RETRIEVED", "INFERRED",
    "PREDICTED", "ASSUMED", "EXPERIMENTAL", "TESTED", "PROVEN",
    "CONTRADICTED", "UNKNOWN",
}


def _flat(props):
    """FalkorDB properties are scalars or lists of scalars."""
    out = {}
    for k, v in (props or {}).items():
        if v is None:
            continue
        if isinstance(v, (dict, tuple)) or (
            isinstance(v, list) and any(isinstance(x, (dict, list)) for x in v)
        ):
            v = json.dumps(v, sort_keys=True, ensure_ascii=False)
        out[k] = v
    return out


def _hash(obj):
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class Substrate:
    def __init__(self, graph, mode):
        self.g = graph
        self.mode = mode
        for label, key in (("Object", "id"), ("Event", "id"), ("Event", "seq")):
            try:
                self.g.query(f"CREATE INDEX FOR (n:{label}) ON (n.{key})")
            except Exception:
                pass  # index already exists
        rows = self.q("MATCH (e:Event) RETURN e.seq, e.hash ORDER BY e.seq DESC LIMIT 1")
        self.seq, self.head = (rows[0][0], rows[0][1]) if rows else (0, "0" * 64)

    def q(self, cypher, **params):
        return self.g.query(cypher, params).result_set

    # ---- MLG OBJECT UNIVERSE -------------------------------------------
    def obj(self, type_, id_, origin, epistemic="OBSERVED", **props):
        assert type_ in TYPES, type_
        assert epistemic in LEVELS, epistemic
        body = _flat(props)
        content_hash = _hash({"type": type_, "id": id_, **body})
        rows = self.q(
            f"MERGE (o:Object:{type_} {{id:$id}}) "
            "ON CREATE SET o.version = 0, o.created = $now "
            "SET o += $props, o.type = $type, o.origin = $origin, "
            "o.epistemic = $ep, o.hash = $hash, o.version = o.version + 1, "
            "o.updated = $now RETURN o.version",
            id=id_, props=body, type=type_, origin=origin, ep=epistemic,
            hash=content_hash, now=time.time(),
        )
        return rows[0][0]

    def link(self, src, rel, dst, **props):
        assert rel in RELS, rel
        rows = self.q(
            "MATCH (a:Object {id:$a}), (b:Object {id:$b}) "
            f"MERGE (a)-[r:{rel}]->(b) SET r += $props RETURN count(r)",
            a=src, b=dst, props=_flat(props),
        )
        if not rows or rows[0][0] == 0:
            raise KeyError(f"link {src} -[{rel}]-> {dst}: endpoint missing")

    # ---- EVENT-SOURCED INTELLIGENCE (X) --------------------------------
    def event(self, kind, about=None, /, **detail):
        """Append to a SHA-256 chained log, like core/store.c."""
        self.seq += 1
        payload = {"seq": self.seq, "kind": kind, "about": about,
                   "detail": _flat(detail), "prev": self.head}
        h = _hash(payload)
        ev = f"ev-{self.seq:06d}"
        self.q(
            "CREATE (e:Object:Event {id:$id, type:'Event', seq:$seq, kind:$kind, about:$about, "
            "hash:$hash, prev:$prev, detail:$detail, at:$now, "
            "origin:'substrate', epistemic:'OBSERVED'})",
            id=ev, seq=self.seq, kind=kind, about=about, hash=h, prev=self.head,
            detail=json.dumps(payload["detail"], ensure_ascii=False), now=time.time(),
        )
        if self.seq > 1:
            self.q(
                "MATCH (a:Event {seq:$p}), (b:Event {seq:$s}) CREATE (a)-[:NEXT]->(b)",
                p=self.seq - 1, s=self.seq,
            )
        if about:
            self.q(
                "MATCH (e:Event {id:$e}), (o:Object {id:$o}) CREATE (e)-[:ABOUT]->(o)",
                e=ev, o=about,
            )
        self.head = h
        return ev

    def verify_events(self, anchor=None):
        """Recompute the chain; any edited event breaks every later hash.
        A chain rewritten from scratch is consistent with itself, so `anchor`
        (a file written by anchor(), kept outside the graph) pins the hash the
        chain had at that point."""
        rows = self.q("MATCH (e:Event) RETURN e.seq, e.kind, e.about, e.detail, e.prev, e.hash ORDER BY e.seq")
        prev = "0" * 64
        for seq, kind, about, detail, stored_prev, stored in rows:
            payload = {"seq": seq, "kind": kind, "about": about,
                       "detail": json.loads(detail), "prev": prev}
            if stored_prev != prev or _hash(payload) != stored:
                return False, seq
            prev = stored
        if anchor is not None:
            head, seq = open(anchor).read().split()
            seq = int(seq)
            if seq > len(rows) or (seq and rows[seq - 1][5] != head):
                return False, seq
        return True, len(rows)

    def anchor(self, path):
        """Write the head of the chain outside the graph."""
        with open(path, "w") as fh:
            fh.write(f"{self.head}  {self.seq}\n")
        return self.head

    def reset(self):
        try:
            self.g.delete()
        except Exception:
            pass
        self.__init__(self.g, self.mode)


def connect(graph="mlg", path=None, fresh=False):
    """FALKORDB_HOST set -> a FalkorDB server (Cloud or Docker);
    otherwise FalkorDBLite, an embedded server inside this process."""
    host = os.environ.get("FALKORDB_HOST")
    if host:
        from falkordb import FalkorDB

        db = FalkorDB(
            host=host,
            port=int(os.environ.get("FALKORDB_PORT", "6379")),
            username=os.environ.get("FALKORDB_USERNAME") or None,
            password=os.environ.get("FALKORDB_PASSWORD") or None,
        )
        mode = f"server {host}"
    else:
        from redislite.falkordb_client import FalkorDB

        db = FalkorDB(path or os.environ.get("MLG_DB", "/tmp/mlg-substrate.db"))
        mode = "embedded FalkorDBLite"
    sub = Substrate(db.select_graph(graph), mode)
    if fresh:
        sub.reset()
    return sub
