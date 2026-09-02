"""Build the threat graph database (Definition 14) end-to-end and write all artifacts.

    ./.venv/bin/python build_graph_db.py

Primary deliverable is the STRICT threat graph (nodes = threats, edges = the five relations).
An optional enrichment overlay (+ Technique/Asset/Actor/Family nodes) is also produced, clearly
separate. Outputs in results/graph/:
  threat_graph.graphml / .json / .cypher       — strict core (Gephi / Neo4j)
  threat_graph.html                             — interactive Cytoscape.js (strict core)
  enriched_graph.graphml / .cypher / .html      — optional overlay
  figures/overview.png (+ _legend), relations_small_multiples.png, campaign.png, ego.png
  GRAPH_REPORT.md
"""

from __future__ import annotations

import glob
import os

import networkx as nx

from zeroday_verify import data as D
from zeroday_verify.graph import build_enriched_graph, build_threat_graph, cyto
from zeroday_verify.graph import persist as P
from zeroday_verify.graph import query as Q
from zeroday_verify.graph import viz as V
from zeroday_verify.graph.demo import build_curated_campaign
from zeroday_verify.graph.schema import EdgeType, NodeType

OUT = "results/graph"
FIG = f"{OUT}/figures"
_BACKBONE_DROP = {EdgeType.VARIANT_OF.value, EdgeType.TARGETS_SAME.value}


def _clean_old() -> None:
    for p in glob.glob(f"{OUT}/*") + glob.glob(f"{FIG}/*"):
        if os.path.isfile(p):
            os.remove(p)


def _backbone_ego(G: nx.MultiDiGraph):
    fams = sorted(((n, d) for n, d in G.nodes(data=True) if d.get("ntype") == NodeType.FAMILY.value),
                  key=lambda nd: nd[1].get("size", 0))
    if not fams:
        return nx.MultiDiGraph(), ""
    medoid = fams[len(fams) // 2][1]["medoid"]
    H = nx.MultiDiGraph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from((u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)
                     if d.get("etype") not in _BACKBONE_DROP)
    return Q.ego_subgraph(H, medoid, radius=1), medoid


def main(per_attack_family: int = 150, seed: int = 17) -> None:
    os.makedirs(FIG, exist_ok=True)
    _clean_old()
    threats = D.build_threats(D.load_subsampled(D.discover_csvs("data/raw_full"),
                                                per_attack_family=per_attack_family,
                                                n_benign=0, seed=seed))
    core = build_threat_graph(threats)        # strict: nodes = threats only
    enriched = build_enriched_graph(threats)  # optional overlay

    # persist
    P.save_graphml(core, f"{OUT}/threat_graph.graphml")
    P.save_json(core, f"{OUT}/threat_graph.json")
    P.export_cypher(core, f"{OUT}/threat_graph.cypher")
    P.save_embeddings(core, f"{OUT}/threat_graph_phi.npy", f"{OUT}/threat_graph_phi_ids.json")
    P.save_graphml(enriched, f"{OUT}/enriched_graph.graphml")
    P.export_cypher(enriched, f"{OUT}/enriched_graph.cypher")

    # interactive (Cytoscape.js)
    cyto.interactive_html(core, f"{OUT}/threat_graph.html",
                          "Graf zagrożeń (rdzeń: węzły = zagrożenia, krawędzie = relacje)")
    cyto.interactive_html(enriched, f"{OUT}/enriched_graph.html",
                          "Graf zagrożeń — warstwa wzbogacona (z technikami / zasobami / aktorami)")

    # static figures (compute the sfdp layout once, share it)
    V.family_summary_figure(core, f"{FIG}/family_summary.png")
    V.inter_family_heatmap(core, f"{FIG}/inter_family_heatmap.png")
    pos = V.layout_threats(core)
    V.overview_figure(core, pos, f"{FIG}/overview.png")
    V.relations_small_multiples(core, pos, f"{FIG}/relations_small_multiples.png")
    V.campaign_figure(build_curated_campaign(enriched=False), f"{FIG}/campaign.png",
                      "Kampania (rdzeń: węzły = zagrożenia, krawędzie = relacje)")
    V.campaign_figure(build_curated_campaign(enriched=True), f"{FIG}/campaign_enriched.png",
                      "Ta sama kampania — warstwa wzbogacona (z technikami / zasobem / aktorem)")
    ego, medoid = _backbone_ego(enriched)
    V.ego_figure(ego, f"{FIG}/ego.png", f"Otoczenie zagrożenia {medoid}")

    nc, ec = Q.node_counts(core), Q.edge_counts(core)
    print("core nodes:", nc)
    print("core edges:", ec)
    _write_report(core, enriched, nc, ec, per_attack_family, seed)
    print(f"wrote {OUT}/GRAPH_REPORT.md + figures/")


def _write_report(core, enriched, nc, ec, per_family, seed) -> None:
    lines = [
        "# Threat graph database — build report\n",
        "**Strict core** (Definition 14): nodes are threats, edges are the five relations. An "
        "optional **enrichment overlay** additionally promotes Technique/Asset/Actor/Family to "
        "nodes. Built from CIC-IDS2017 `TrafficLabelling` flows (real IP/timestamp).\n",
        f"- Subsample: {per_family} threats/family, seed {seed}.",
        f"- Core: {core.number_of_nodes()} threat nodes, {core.number_of_edges()} relation edges.",
        f"- Enriched overlay: {enriched.number_of_nodes()} nodes, {enriched.number_of_edges()} edges.\n",
        "## Core relation edges\n",
    ]
    lines += [f"- `{k}`: {v}" for k, v in sorted(ec.items())]
    lines += [
        "\n## Notes\n",
        "- CIC-IDS2017 is a controlled testbed (~10 attacker / 11 victim hosts), so per-actor "
        "multi-stage chains are rare and `escalates`/`enables` are sparse in the bulk graph; the "
        "curated campaign (`figures/campaign.png`) exercises every relation.",
        "- `variant_of` dominates edge count and is hidden by default in the interactive view and "
        "the per-relation small-multiples make each relation legible on its own.",
        "\n## Files\n",
        "- Strict core: `threat_graph.{graphml,json,cypher,html}`.",
        "- Overlay: `enriched_graph.{graphml,cypher,html}`.",
        "- Figures: `family_summary.png`, `inter_family_heatmap.png`, "
        "`relations_small_multiples.png`, `campaign.png`, `campaign_enriched.png`, `overview.png`, "
        "`ego.png`.",
    ]
    with open(f"{OUT}/GRAPH_REPORT.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
