"""SELF-EXPLANATION (manifesto LXXI-LXXII): every answer is a graph query.

Nothing here is generated text about the past; each line is read back from
the decisions, evaluations, traits and measurements stored in FalkorDB.
"""


def champion(sub, body_id):
    rows = sub.q(
        "MATCH (d:Decision {verdict:'COMMIT', body:$b})-[:CHOSE]->(br:Brain) "
        "RETURN d.id, br.id, d.reason ORDER BY d.id DESC LIMIT 1", b=body_id)
    return rows[0] if rows else None


def why_champion(sub, body_id):
    c = champion(sub, body_id)
    if not c:
        return [f"No brain has been committed for {body_id}."]
    did, bid, reason = c
    out = [f"WHY {bid} lives in {body_id}?", f"  decision {did}: {reason}"]
    g = sub.q(
        "MATCH (g:Genome)-[:EXPRESSES]->(:Brain {id:$b}) RETURN g.id, g.generation, g.d_model, "
        "g.n_layers, g.n_heads, g.n_experts, g.top_k, g.ffn_mult, g.ctx, g.dtype, g.pred_params", b=bid)[0]
    out.append(f"  genome {g[0]} (generation {g[1]}): d_model {g[2]}, {g[3]} layers, {g[4]} heads, "
               f"{g[5]} experts top-{g[6]}, ffn x{g[7]}, context {g[8]}, {g[9]}, "
               f"{g[10] / 1e6:.2f}M params")
    for suite, bpb, q, eid in sub.q(
            "MATCH (:Decision {id:$d})-[:JUSTIFIED_BY]->(e:Evaluation)-[:ON_SUITE]->(s:Suite) "
            "MATCH (br:Brain)-[:EVALUATED_BY]->(e) "
            "RETURN s.id, e.bits_per_byte, e.q_language, br.id", d=did):
        who = "this brain" if eid == bid else f"previous champion {eid}"
        out.append(f"  evidence: {who} on {suite}: {bpb:.3f} bits/byte, Q_LANGUAGE {q:.4f}")
    for name, value, unit, level, why, n in sub.q(
            "MATCH (:Brain {id:$b})-[r:SHAPED_BY]->(t:Trait) "
            "OPTIONAL MATCH (t)-[:PROVEN_BY]->(m) "
            "RETURN t.name, t.value, t.unit, t.epistemic, r.why, count(m)", b=bid):
        out.append(f"  shaped by body trait {name} = {_v(value, unit)} [{level}, {n} proofs]: {why}")
    chain = inheritance_chain(sub, bid)
    if len(chain) > 1:
        out.append(f"  mind inherited across {len(chain) - 1} generation(s), "
                   f"{chain[0][4]:.0f} s of training in total:")
        for b, mode, bpb, params, secs in chain:
            out.append(f"    {b[6:18]}  {mode or 'born from scratch':18s} "
                       f"{(bpb or float('nan')):.3f} bpb  {(params or 0) / 1e6:7.2f}M params")
    lin = lineage(sub, g[0])
    if len(lin) > 1:
        steps = " -> ".join(f"{gid[7:13]}" + (f" ({', '.join(m)})" if m else "") for gid, m in lin)
        out.append(f"  lineage: {steps}")
    for verdict, stage, n in sub.q(
            "MATCH (d:Decision {body:$b}) RETURN d.verdict, d.stage, count(*) ORDER BY count(*) DESC",
            b=body_id):
        out.append(f"  rivals: {n} x {verdict} at {stage}")
    for name, value, meaning in sub.q(
            "MATCH (:Dataset)-[:HAS_TRAIT]->(t:Trait {name:'capacity_wall'}) "
            "RETURN t.name, t.value, t.meaning"):
        out.append(f"  known wall: {meaning}")
    return out


def inheritance_chain(sub, brain_id, limit=50):
    """From a brain back to the first ancestor whose weights it still carries:
    [(brain, how it inherited, dev bpb, params, lineage seconds)]."""
    chain, bid = [], brain_id
    while bid and len(chain) < limit:
        rows = sub.q(
            "MATCH (g:Genome)-[:EXPRESSES]->(b:Brain {id:$b}) "
            "OPTIONAL MATCH (g)-[:INHERITED_FROM]->(p:Brain) "
            "RETURN b.inherited, b.dev_bpb, b.params, b.lineage_seconds, p.id", b=bid)
        if not rows:
            break
        mode, bpb, params, secs, parent = rows[0]
        chain.append((bid, mode, bpb, params, secs or 0.0))
        bid = parent
    return chain


def lineage(sub, genome_id):
    """Ancestors first: [(genome, genes mutated to reach the next one)]."""
    rows = sub.q(
        "MATCH p=(a:Genome)-[:PARENT_OF*1..20]->(g:Genome {id:$g}) "
        "WITH p ORDER BY length(p) DESC LIMIT 1 "
        "RETURN [n IN nodes(p) | n.id], [r IN relationships(p) | r.mutated]", g=genome_id)
    if not rows:
        return [(genome_id, [])]
    ids, muts = rows[0]
    muts = [m if isinstance(m, list) else [] for m in muts]
    return [(ids[0], [])] + [(ids[i + 1], muts[i]) for i in range(len(muts))]


def why_killed(sub, node_id):
    rows = sub.q(
        "MATCH (d:Decision)-[:KILLED]->(x:Object {id:$x}) "
        "OPTIONAL MATCH (d)-[:JUSTIFIED_BY]->(e) "
        "RETURN d.verdict, d.stage, d.reason, collect(e.id)", x=node_id)
    return [f"{v} at {s}: {r} (evidence: {', '.join(ev) or 'prediction'})" for v, s, r, ev in rows]


def body_report(sub, body_id):
    out = [f"BODY {body_id}"]
    for name, value, unit, level, n in sub.q(
            "MATCH (:Body {id:$b})-[:HAS_TRAIT]->(t:Trait) OPTIONAL MATCH (t)-[:PROVEN_BY]->(m) "
            "RETURN t.name, t.value, t.unit, t.epistemic, count(m) ORDER BY t.name", b=body_id):
        out.append(f"  {name:24s} {_v(value, unit):>22s}  [{level}{f', {n} proofs' if n else ''}]")
    return out


def _v(value, unit):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if unit == "B":
            for u in ("B", "KiB", "MiB", "GiB"):
                if value < 1024 or u == "GiB":
                    return f"{value:.1f} {u}"
                value /= 1024
        return f"{value:.4g} {unit}".strip()
    return f"{value} {unit}".strip()
