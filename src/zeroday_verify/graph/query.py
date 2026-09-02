"""Query helpers over the threat graph (Definition 14)."""

from __future__ import annotations

from collections import Counter

import networkx as nx

from .schema import EdgeType, NodeType


def node_counts(G: nx.MultiDiGraph) -> dict[str, int]:
    return dict(Counter(d.get("ntype") for _, d in G.nodes(data=True)))


def edge_counts(G: nx.MultiDiGraph) -> dict[str, int]:
    return dict(Counter(d.get("etype") for _, _, d in G.edges(data=True)))


def _edges_of(G: nx.MultiDiGraph, etype: EdgeType):
    return [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == etype.value]


def variants_of(G: nx.MultiDiGraph, tid: str) -> list[str]:
    """Threats linked to `tid` by variant_of (treated as symmetric)."""
    out = set()
    for u, v, d in G.edges(data=True):
        if d.get("etype") == EdgeType.VARIANT_OF.value:
            if u == tid:
                out.add(v)
            elif v == tid:
                out.add(u)
    return sorted(out)


def family_members(G: nx.MultiDiGraph, fid: str) -> list[str]:
    return sorted(u for u, v, d in G.edges(data=True)
                  if d.get("etype") == EdgeType.MEMBER_OF.value and v == fid)


def attack_chain(G: nx.MultiDiGraph, start: str) -> list[str]:
    """Follow precedes edges forward from `start` (a single time-ordered campaign path)."""
    chain, cur, seen = [start], start, {start}
    while True:
        nxt = [v for u, v, d in G.edges(cur, data=True)
               if d.get("etype") == EdgeType.PRECEDES.value and v not in seen]
        if not nxt:
            break
        cur = nxt[0]
        chain.append(cur)
        seen.add(cur)
    return chain


def campaign_of(G: nx.MultiDiGraph, actor_id: str) -> set[str]:
    """All threat nodes originating from a given actor node."""
    return {u for u, v, d in G.edges(data=True)
            if d.get("etype") == EdgeType.ORIGINATES_FROM.value and v == actor_id}


def threats_targeting(G: nx.MultiDiGraph, asset_id: str) -> set[str]:
    return {u for u, v, d in G.edges(data=True)
            if d.get("etype") == EdgeType.TARGETS.value and v == asset_id}


def ego_subgraph(G: nx.MultiDiGraph, node: str, radius: int = 1) -> nx.MultiDiGraph:
    """Undirected ego network around `node` (for focused visualisation)."""
    nodes = nx.ego_graph(G.to_undirected(as_view=True), node, radius=radius).nodes()
    return G.subgraph(nodes).copy()


def largest_campaign(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """The subgraph of the actor with the longest precedes-chain (a good demo campaign)."""
    actors = [n for n, d in G.nodes(data=True) if d.get("ntype") == NodeType.ACTOR.value]
    best_nodes: set[str] = set()
    for ac in actors:
        ths = campaign_of(G, ac)
        if len(ths) > len(best_nodes):
            best_nodes = ths
    if not best_nodes:
        return nx.MultiDiGraph()
    related = set(best_nodes)
    for t in best_nodes:
        related.update(G.successors(t))
    return G.subgraph(related).copy()
