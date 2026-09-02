"""Construct the threat graph (Definition 14) from Threat objects.

Two builders:
  * build_threat_graph  — the STRICT core (Def 14): nodes are threats only, edges are the five
    relations (variant_of, escalates, precedes, targets_same, enables). Technique / asset / actor
    / family live as node attributes and are the justification for the edges, not separate nodes.
  * build_enriched_graph — an OPTIONAL overlay that additionally promotes Technique / Asset /
    Actor / Family to nodes (for attribution / grouping views). Clearly a separate extension.

Only attack threats are nodes (a benign flow is not a threat, Def 5).
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np

from ..novelty import KnownThreatModel
from ..similarity import WEIGHTS_FULL, SimWeights, composite_sim
from ..taxonomy import ATTACK_TECHNIQUES
from ..threats import Threat
from .schema import PFX, EdgeType, GraphParams, NodeType

TECHNIQUE_INFO: dict[str, tuple[str, str]] = {
    "T1046": ("Network Service Discovery", "Discovery"),
    "T1595": ("Active Scanning", "Reconnaissance"),
    "T1498": ("Network Denial of Service", "Impact"),
    "T1499": ("Endpoint Denial of Service", "Impact"),
    "T1110": ("Brute Force", "Credential Access"),
    "T1190": ("Exploit Public-Facing Application", "Initial Access"),
    "T1059": ("Command and Scripting Interpreter", "Execution"),
    "T1071": ("Application Layer Protocol", "Command and Control"),
    "T1203": ("Exploitation for Client Execution", "Execution"),
    "T1041": ("Exfiltration Over C2 Channel", "Exfiltration"),
    "T1212": ("Exploitation for Credential Access", "Credential Access"),
}

_MISSING = {"", "?", "nan", "none", "0.0.0.0", "0"}


def _techs(th: Threat) -> list[str]:
    return [ATTACK_TECHNIQUES[i] for i, v in enumerate(th.A) if v > 0]


def _clean(v: object) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in _MISSING else s


# ---------------------------------------------------------------------------------------
# strict core (Definition 14): nodes = threats, edges = the five relations
# ---------------------------------------------------------------------------------------

def build_threat_graph(threats: list[Threat], params: GraphParams | None = None,
                       weights: SimWeights = WEIGHTS_FULL) -> nx.MultiDiGraph:
    """Strict threat graph: nodes are threats, edges are the five Def 14 relations."""
    params = params or GraphParams()
    attacks = [t for t in threats if t.is_attack]
    if not attacks:
        raise ValueError("no attack threats to build a graph from")

    G = nx.MultiDiGraph()
    G.graph["kind"] = "threat-core"
    model = KnownThreatModel(weights, min_cluster_size=params.min_cluster_size).fit(attacks)
    labels = model.labels_

    th_ids: list[str] = []
    for i, t in enumerate(attacks):
        tid = f"{PFX[NodeType.THREAT]}:{i}"
        th_ids.append(tid)
        techs = _techs(t)
        G.add_node(
            tid, ntype=NodeType.THREAT.value, family=t.family, is_attack=True, domain=t.D,
            channel=f"{t.K[0]}/{t.K[1]}", techniques=techs, technique_str=",".join(techs),
            sigma=[float(x) for x in t.sigma], sigma_norm=float(np.linalg.norm(t.sigma)),
            novelty=float(model.novelty(t)), ts=float(t.meta.get("ts", i)),
            dst_port=t.meta.get("dst_port"),
            actor=_clean(t.events[0].values.get("src_ip")) if t.events else None,
            asset=_clean(t.Tg[1]), cluster=int(labels[i]), text=t.text[:200], phi=t.phi)

    _add_variant_of(G, attacks, th_ids, labels, params, weights)
    _add_targets_same(G, attacks, th_ids, params)
    _add_precedes_escalates_enables(G, attacks, th_ids, params)
    return G


# ---------------------------------------------------------------------------------------
# optional enrichment overlay: + Technique / Asset / Actor / Family nodes
# ---------------------------------------------------------------------------------------

def build_enriched_graph(threats: list[Threat], params: GraphParams | None = None,
                         weights: SimWeights = WEIGHTS_FULL) -> nx.MultiDiGraph:
    """Strict threat graph plus a knowledge/context overlay (Technique/Asset/Actor/Family)."""
    params = params or GraphParams()
    G = build_threat_graph(threats, params, weights)
    G.graph["kind"] = "threat-enriched"
    attacks = [t for t in threats if t.is_attack]
    th_ids = [f"{PFX[NodeType.THREAT]}:{i}" for i in range(len(attacks))]
    labels = [G.nodes[t]["cluster"] for t in th_ids]

    for i, t in enumerate(attacks):
        tid = th_ids[i]
        for tech in _techs(t):
            ntid = f"{PFX[NodeType.TECHNIQUE]}:{tech}"
            if ntid not in G:
                name, tactic = TECHNIQUE_INFO.get(tech, (tech, "Unknown"))
                G.add_node(ntid, ntype=NodeType.TECHNIQUE.value, tech_id=tech, name=name,
                           tactic=tactic)
            G.add_edge(tid, ntid, etype=EdgeType.REALIZES_TECHNIQUE.value)
        asset = G.nodes[tid].get("asset")
        if asset:
            aid = f"{PFX[NodeType.ASSET]}:{asset}"
            if aid not in G:
                G.add_node(aid, ntype=NodeType.ASSET.value, asset_id=asset)
            G.add_edge(tid, aid, etype=EdgeType.TARGETS.value)
        actor = G.nodes[tid].get("actor")
        if actor:
            acid = f"{PFX[NodeType.ACTOR]}:{actor}"
            if acid not in G:
                G.add_node(acid, ntype=NodeType.ACTOR.value, actor_id=actor)
            G.add_edge(tid, acid, etype=EdgeType.ORIGINATES_FROM.value)

    _add_family_nodes(G, attacks, th_ids, labels)
    return G


# backwards-compatible default points at the strict core
build_graph = build_threat_graph


# ---------------------------------------------------------------------------------------
# edge construction (shared by both builders; self-contained)
# ---------------------------------------------------------------------------------------

def _medoid_local(attacks, idx) -> int:
    phi = np.vstack([attacks[i].phi for i in idx])
    mean = phi.mean(axis=0)
    return idx[int(np.argmax(phi @ mean))]


def _add_variant_of(G, attacks, th_ids, labels, params, weights) -> None:
    by_cluster: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        if lab != -1:
            by_cluster.setdefault(int(lab), []).append(i)
    for idx in by_cluster.values():
        if len(idx) <= 120:
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    if composite_sim(attacks[i], attacks[j], weights) >= params.tau_var:
                        G.add_edge(th_ids[i], th_ids[j], etype=EdgeType.VARIANT_OF.value)
        else:  # large cluster: connect each member to the medoid if similar enough
            m = _medoid_local(attacks, idx)
            for i in idx:
                if i != m and composite_sim(attacks[i], attacks[m], weights) >= params.tau_var:
                    G.add_edge(th_ids[i], th_ids[m], etype=EdgeType.VARIANT_OF.value)


def _add_targets_same(G, attacks, th_ids, params) -> None:
    by_asset: dict[str, list[int]] = {}
    for i, t in enumerate(attacks):
        asset = _clean(t.Tg[1])
        if asset:
            by_asset.setdefault(asset, []).append(i)
    for idx in by_asset.values():
        if len(idx) < 2:
            continue
        if len(idx) <= params.targets_same_cap:
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    G.add_edge(th_ids[idx[a]], th_ids[idx[b]], etype=EdgeType.TARGETS_SAME.value)
        else:  # large asset group: star to the earliest threat to bound edge count
            anchor = min(idx, key=lambda i: float(attacks[i].meta.get("ts", i)))
            for i in idx:
                if i != anchor:
                    G.add_edge(th_ids[anchor], th_ids[i], etype=EdgeType.TARGETS_SAME.value)


def _add_precedes_escalates_enables(G, attacks, th_ids, params) -> None:
    by_actor: dict[str, list[int]] = {}
    for i, t in enumerate(attacks):
        actor = _clean(t.events[0].values.get("src_ip")) if t.events else None
        if actor:
            by_actor.setdefault(actor, []).append(i)
    for idx in by_actor.values():
        idx.sort(key=lambda i: float(attacks[i].meta.get("ts", i)))
        for a, b in zip(idx, idx[1:], strict=False):
            ta = float(attacks[a].meta.get("ts", a))
            tb = float(attacks[b].meta.get("ts", b))
            if tb - ta > params.precedes_window_s:
                continue
            G.add_edge(th_ids[a], th_ids[b], etype=EdgeType.PRECEDES.value)
            if np.linalg.norm(attacks[b].sigma) > np.linalg.norm(attacks[a].sigma):
                G.add_edge(th_ids[a], th_ids[b], etype=EdgeType.ESCALATES.value)
            ta_set, tb_set = set(_techs(attacks[a])), set(_techs(attacks[b]))
            if (ta_set & set(params.recon_techniques)) and (tb_set - set(params.recon_techniques)):
                G.add_edge(th_ids[a], th_ids[b], etype=EdgeType.ENABLES.value)


def _add_family_nodes(G, attacks, th_ids, labels) -> None:
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        idx = [i for i, lb in enumerate(labels) if lb == lab]
        name = Counter(attacks[i].family for i in idx).most_common(1)[0][0]
        fid = f"{PFX[NodeType.FAMILY]}:{lab}"
        G.add_node(fid, ntype=NodeType.FAMILY.value, cluster=int(lab), size=len(idx),
                   dominant_family=name, medoid=th_ids[_medoid_local(attacks, idx)])
        for i in idx:
            G.add_edge(th_ids[i], fid, etype=EdgeType.MEMBER_OF.value)
