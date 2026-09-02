"""Tests for the threat graph database (Definition 14): strict core + enriched overlay."""

import numpy as np

from zeroday_verify.events import T_NET, Event
from zeroday_verify.graph import build_enriched_graph, build_threat_graph
from zeroday_verify.graph import persist as P
from zeroday_verify.graph import query as Q
from zeroday_verify.graph.schema import EdgeType, GraphParams, NodeType
from zeroday_verify.taxonomy import family_to_A, family_to_sigma
from zeroday_verify.threats import Threat


def _threat(phi, family, src, dst, ts, port, dim=8):
    vec = np.zeros(dim)
    vec[: len(phi)] = phi
    ev = Event(event_type=T_NET, ts=ts, family=family, is_attack=True,
               values={"src_ip": src, "dst_ip": dst, "dst_port": port, "protocol": 6,
                       "flags": "", "length": 0, "payload_meta": 0})
    t = Threat(events=[ev], K=("transport", "TCP"), Tg=("protected_asset", dst), D="Network",
               A=family_to_A(family), sigma=family_to_sigma(family), m_vec=np.zeros(3),
               raw_vec=np.zeros(1), text=f"{family} flow", family=family, is_attack=True,
               meta={"ts": ts, "dst_port": port})
    t.phi = vec
    return t


def _synthetic():
    threats = []
    for i in range(3):
        threats.append(_threat([1.0, 0.02 * i], "WebAttack", "10.0.0.9", "10.0.0.5", 100 + i, 80))
    for i in range(3):
        threats.append(_threat([0.0, 1.0, 0.02 * i], "DoS", "10.0.0.9", "10.0.0.5", 200 + i, 80))
    return threats


def _core():
    return build_threat_graph(_synthetic(), GraphParams(tau_var=0.9, min_cluster_size=2,
                                                        precedes_window_s=10000))


def test_core_nodes_are_strictly_threats():
    G = _core()
    counts = Q.node_counts(G)
    assert counts == {NodeType.THREAT.value: 6}, f"core must be threats-only, got {counts}"


def test_core_edges_are_only_relations():
    G = _core()
    ec = Q.edge_counts(G)
    allowed = {EdgeType.VARIANT_OF.value, EdgeType.ESCALATES.value, EdgeType.PRECEDES.value,
               EdgeType.TARGETS_SAME.value, EdgeType.ENABLES.value}
    assert set(ec).issubset(allowed)
    for et in (EdgeType.VARIANT_OF, EdgeType.PRECEDES, EdgeType.TARGETS_SAME):
        assert ec.get(et.value, 0) > 0, f"missing {et.value}"


def test_enriched_adds_context_nodes_and_links():
    G = build_enriched_graph(_synthetic(), GraphParams(tau_var=0.9, min_cluster_size=2))
    counts = Q.node_counts(G)
    for nt in (NodeType.THREAT, NodeType.TECHNIQUE, NodeType.ASSET, NodeType.ACTOR, NodeType.FAMILY):
        assert counts.get(nt.value, 0) > 0, f"enriched missing {nt.value}"
    ec = Q.edge_counts(G)
    for et in (EdgeType.REALIZES_TECHNIQUE, EdgeType.MEMBER_OF, EdgeType.TARGETS,
               EdgeType.ORIGINATES_FROM):
        assert ec.get(et.value, 0) > 0, f"enriched missing {et.value}"


def test_realizes_technique_matches_A():
    G = build_enriched_graph(_synthetic(), GraphParams(tau_var=0.9, min_cluster_size=2))
    th = next(n for n, d in G.nodes(data=True)
              if d.get("ntype") == NodeType.THREAT.value and d.get("family") == "WebAttack")
    techs = {v.split(":")[1] for u, v, d in G.edges(th, data=True)
             if d.get("etype") == EdgeType.REALIZES_TECHNIQUE.value}
    assert techs == {"T1190", "T1059"}


def test_variant_of_within_family_only():
    G = _core()
    for u, v, d in G.edges(data=True):
        if d.get("etype") == EdgeType.VARIANT_OF.value:
            assert G.nodes[u]["family"] == G.nodes[v]["family"]


def test_variant_of_threshold_blocks_dissimilar():
    threats = [_threat([1.0, 0.0], "WebAttack", "10.0.0.9", "10.0.0.5", 1, 80),
               _threat([0.0, 1.0], "DoS", "10.0.0.9", "10.0.0.5", 2, 80)]
    G = build_threat_graph(threats, GraphParams(min_cluster_size=2))
    assert Q.edge_counts(G).get(EdgeType.VARIANT_OF.value, 0) == 0


def test_graphml_roundtrip(tmp_path):
    G = _core()
    p = tmp_path / "g.graphml"
    P.save_graphml(G, str(p))
    H = P.load_graphml(str(p))
    assert H.number_of_nodes() == G.number_of_nodes()
    assert H.number_of_edges() == G.number_of_edges()


def test_cypher_export_has_merges(tmp_path):
    G = build_enriched_graph(_synthetic(), GraphParams(tau_var=0.9, min_cluster_size=2))
    p = tmp_path / "g.cypher"
    P.export_cypher(G, str(p))
    text = p.read_text()
    assert "MERGE" in text and "REALIZES_TECHNIQUE" in text


def test_curated_campaign_exercises_all_def14_relations():
    """Acceptance test mirroring the document's worked example (Th3->Th1<->Th2->Th4)."""
    from zeroday_verify.graph.demo import build_curated_campaign

    g = build_curated_campaign()
    ec = Q.edge_counts(g)
    for et in (EdgeType.PRECEDES, EdgeType.ESCALATES, EdgeType.ENABLES, EdgeType.VARIANT_OF,
               EdgeType.TARGETS_SAME):
        assert ec.get(et.value, 0) > 0, f"curated campaign missing {et.value}"
