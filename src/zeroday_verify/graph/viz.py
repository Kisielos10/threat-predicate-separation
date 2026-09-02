"""Static visualisation of the threat graph for slides/thesis.

  * graphviz_overview        — crisp sfdp layout of the whole threat graph, nodes coloured by
                               family (the "wow" overview; families emerge as colour clusters).
  * relations_small_multiples — one panel per relation type over a shared layout, so it is
                               unambiguous which edge is which (addresses "can't tell links apart").
  * campaign_figure          — temporal left-to-right campaign story (the curated example).
  * ego_figure               — one threat's neighbourhood.

Interactive viz lives in `cyto.py` (Cytoscape.js). Threats are coloured by attack family
everywhere for consistency with the interactive legend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from .schema import (  # noqa: E402
    EDGE_COLORS,
    FAMILY_COLORS,
    FAMILY_FALLBACK,
    NODE_COLORS,
    NodeType,
)

_THREAT_EDGES = ["precedes", "escalates", "enables", "variant_of", "targets_same"]


def _fill(d: dict) -> str:
    if d.get("ntype") == NodeType.THREAT.value:
        return FAMILY_COLORS.get(d.get("family"), FAMILY_FALLBACK)
    return NODE_COLORS.get(d.get("ntype"), "#888888")


# ---------------------------------------------------------------------------------------
# shared layout (graphviz sfdp) + static overview + per-relation small multiples
# ---------------------------------------------------------------------------------------

def layout_threats(G: nx.MultiDiGraph, prog: str = "sfdp") -> dict:
    """Graphviz layout of the threat nodes (families emerge as clusters via variant_of).

    Computed once and shared by the overview and the small-multiples (it is the slow step).
    """
    threats = [n for n, d in G.nodes(data=True) if d.get("ntype") == NodeType.THREAT.value]
    core = G.subgraph(threats)
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        return graphviz_layout(nx.DiGraph(core), prog=prog)
    except Exception:
        return nx.spring_layout(nx.Graph(core), seed=17, k=0.4, iterations=60)


def _fam_handles(G, threats):
    fams = sorted({G.nodes[n].get("family") for n in threats})
    return [mpatches.Patch(color=FAMILY_COLORS.get(f, FAMILY_FALLBACK), label=f) for f in fams]


def family_summary_figure(G: nx.MultiDiGraph, path: str) -> None:
    """Aggregated overview: one node per attack family (sized by count), inter-family relation
    edges weighted by count. Clean, legible 'threat landscape' summary of the whole graph."""
    from collections import Counter

    fam_of = {n: d.get("family") for n, d in G.nodes(data=True)
              if d.get("ntype") == NodeType.THREAT.value}
    fam_count = Counter(fam_of.values())
    inter: Counter = Counter()
    for u, v in G.edges():
        if u in fam_of and v in fam_of and fam_of[u] != fam_of[v]:
            inter[(fam_of[u], fam_of[v])] += 1

    fams = sorted(fam_count)
    n = len(fams)
    ang = {f: 2 * np.pi * i / n for i, f in enumerate(fams)}
    pos = {f: (np.cos(a), np.sin(a)) for f, a in ang.items()}
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    if inter:
        mx = max(inter.values())
        for (fa, fb), c in inter.items():
            x1, y1 = pos[fa]; x2, y2 = pos[fb]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color="#9aa3ab",
                                        lw=0.8 + 4.0 * c / mx, alpha=0.55,
                                        connectionstyle="arc3,rad=0.18"))
    mxc = max(fam_count.values())
    for f in fams:
        x, y = pos[f]
        ax.scatter([x], [y], s=900 + 5000 * fam_count[f] / mxc,
                   c=FAMILY_COLORS.get(f, FAMILY_FALLBACK), alpha=0.92, linewidths=0, zorder=3)
        ax.text(x, y, f"{f}\n{fam_count[f]}", ha="center", va="center", fontsize=10,
                fontweight="bold", color="#ffffff", zorder=4)
    ax.set_title("Krajobraz zagrożeń — rodziny i ich powiązania (grubość = liczba relacji)")
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def inter_family_heatmap(G: nx.MultiDiGraph, path: str) -> None:
    """Matrix of cross/intra-family relation counts (excluding variant_of, which is intra by
    definition). Quantifies which families couple via shared targets / temporal order."""
    fam_of = {n: d.get("family") for n, d in G.nodes(data=True)
              if d.get("ntype") == NodeType.THREAT.value}
    fams = sorted(set(fam_of.values()))
    idx = {f: i for i, f in enumerate(fams)}
    M = np.zeros((len(fams), len(fams)))
    for u, v, d in G.edges(data=True):
        if d.get("etype") == "variant_of":
            continue
        if u in fam_of and v in fam_of:
            M[idx[fam_of[u]], idx[fam_of[v]]] += 1
    norm = np.log1p(M) / max(np.log1p(M).max(), 1e-9)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(norm, cmap="viridis")
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=40, ha="right")
    ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams)
    for i in range(len(fams)):
        for j in range(len(fams)):
            if M[i, j] > 0:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] < 0.6 else "black")
    ax.set_xlabel("do rodziny (cel relacji)"); ax.set_ylabel("z rodziny (źródło relacji)")
    ax.set_title("Powiązania między rodzinami (relacje bez variant_of)")
    fig.colorbar(im, fraction=0.046, pad=0.04, label="log(1 + liczba relacji)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def overview_figure(G: nx.MultiDiGraph, pos: dict, path: str,
                    relations=("precedes",)) -> None:
    """Single clean panel: threat nodes coloured by family (clusters), backbone relations drawn."""
    threats = [n for n, d in G.nodes(data=True) if d.get("ntype") == NodeType.THREAT.value]
    core = G.subgraph(threats)
    fig, ax = plt.subplots(figsize=(12, 9))
    rel_handles = []
    for et in relations:
        elist = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == et
                 and u in pos and v in pos]
        if elist:
            nx.draw_networkx_edges(core, pos, edgelist=elist, edge_color=EDGE_COLORS[et],
                                   width=0.7, alpha=0.4, arrows=False, ax=ax)
            rel_handles.append(mpatches.Patch(color=EDGE_COLORS[et], label=et))
    sizes = [14 + 60 * float(G.nodes[n].get("novelty", 0.2)) for n in threats]
    ax.scatter([pos[n][0] for n in threats], [pos[n][1] for n in threats],
               s=sizes, c=[_fill(G.nodes[n]) for n in threats], alpha=0.85, linewidths=0)
    leg1 = ax.legend(handles=_fam_handles(G, threats), loc="upper left", fontsize=9,
                     title="rodzina (węzeł = zagrożenie)")
    ax.add_artist(leg1)
    if rel_handles:
        ax.legend(handles=rel_handles, loc="lower left", fontsize=9, title="relacje")
    ax.set_title("Graf zagrożeń — przegląd (węzły = zagrożenia, kolor = rodzina)")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def relations_small_multiples(G: nx.MultiDiGraph, pos: dict, path: str) -> None:
    """One panel per relation type, sharing the layout, so each edge type is unambiguous."""
    threats = [n for n, d in G.nodes(data=True) if d.get("ntype") == NodeType.THREAT.value]
    core = G.subgraph(threats)
    fills = [_fill(G.nodes[n]) for n in threats]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, et in zip(axes.ravel(), _THREAT_EDGES, strict=False):
        ax.scatter([pos[n][0] for n in threats], [pos[n][1] for n in threats],
                   s=10, c=fills, alpha=0.5, linewidths=0)
        elist = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == et
                 and u in pos and v in pos]
        nx.draw_networkx_edges(core, pos, edgelist=elist, edge_color=EDGE_COLORS[et], width=0.8,
                               alpha=0.7, arrows=False, ax=ax)
        ax.set_title(f"{et}  (n={len(elist)})", fontsize=12, color=EDGE_COLORS[et])
        ax.axis("off")
    ax = axes.ravel()[5]
    ax.legend(handles=_fam_handles(G, threats), loc="center", frameon=False,
              title="rodzina (kolor węzła)")
    ax.axis("off")
    fig.suptitle("Relacje między zagrożeniami — każdy panel to jeden typ krawędzi", fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------------------
# campaign (temporal) and ego figures
# ---------------------------------------------------------------------------------------

def campaign_figure(sub: nx.MultiDiGraph, path: str, title: str) -> None:
    """Threats left-to-right by time; threat-to-threat relations prominent, context faint."""
    from collections import defaultdict

    threats = sorted([n for n, d in sub.nodes(data=True) if d.get("ntype") == NodeType.THREAT.value],
                     key=lambda n: float(sub.nodes[n].get("ts", 0.0)))
    if not threats:
        return
    span = max((len(threats) - 1) * 2.0, 1.0)
    pos = {t: (i * 2.0, 0.0) for i, t in enumerate(threats)}
    bands = {NodeType.TECHNIQUE.value: 2.6, NodeType.ACTOR.value: 1.3,
             NodeType.ASSET.value: -1.3, NodeType.FAMILY.value: -2.6}
    by_band: dict[float, list] = defaultdict(list)
    for o, d in sub.nodes(data=True):
        if o in pos:
            continue
        nbr = [pos[t][0] for t in threats if sub.has_edge(t, o) or sub.has_edge(o, t)]
        by_band[bands.get(d.get("ntype"), 3.6)].append((float(np.mean(nbr)) if nbr else span / 2, o))
    for y, items in by_band.items():
        items.sort()
        xs = np.linspace(0, span, len(items)) if len(items) > 1 else [span / 2]
        for (_, o), x in zip(items, xs, strict=True):
            pos[o] = (float(x), y)

    fig, ax = plt.subplots(figsize=(11, 7))
    ctx = [(u, v) for u, v, d in sub.edges(data=True) if d.get("etype") not in _THREAT_EDGES]
    nx.draw_networkx_edges(sub, pos, edgelist=ctx, edge_color="#cccccc", width=0.8, alpha=0.45,
                           arrows=False, ax=ax)
    rad = {"precedes": 0.0, "escalates": 0.22, "enables": -0.22, "variant_of": 0.40,
           "targets_same": -0.40}
    for et in _THREAT_EDGES:
        elist = [(u, v) for u, v, d in sub.edges(data=True) if d.get("etype") == et]
        if elist:
            nx.draw_networkx_edges(sub, pos, edgelist=elist, edge_color=EDGE_COLORS[et], width=2.4,
                                   alpha=0.9, arrows=True, arrowsize=18,
                                   connectionstyle=f"arc3,rad={rad.get(et, 0.1)}", ax=ax)
    for n, d in sub.nodes(data=True):
        sz = 900 if d.get("ntype") == NodeType.THREAT.value else 430
        nx.draw_networkx_nodes(sub, pos, nodelist=[n], node_color=[_fill(d)], node_size=sz,
                               alpha=0.93, ax=ax)
    labels = {n: _short_label(n, d) for n, d in sub.nodes(data=True)}
    for i, t in enumerate(threats, start=1):
        labels[t] = f"{i}. {sub.nodes[t].get('family', t)}"
    nx.draw_networkx_labels(sub, pos, labels=labels, font_size=8, ax=ax)
    edge_handles = [mpatches.Patch(color=EDGE_COLORS[e], label=e) for e in _THREAT_EDGES
                    if any(d.get("etype") == e for _, _, d in sub.edges(data=True))]
    ax.legend(handles=edge_handles, loc="upper center", ncol=5, fontsize=8,
              bbox_to_anchor=(0.5, -0.02), title="relacje między zagrożeniami")
    ax.set_title(title)
    ax.margins(0.12); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def ego_figure(sub: nx.MultiDiGraph, path: str, title: str) -> None:
    if sub.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(sub, seed=17, k=0.9, iterations=120)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    seen = []
    for et, color in EDGE_COLORS.items():
        elist = [(u, v) for u, v, d in sub.edges(data=True) if d.get("etype") == et]
        if elist:
            seen.append(et)
            nx.draw_networkx_edges(sub, pos, edgelist=elist, edge_color=color, width=1.8,
                                   alpha=0.8, arrows=True, arrowsize=14,
                                   connectionstyle="arc3,rad=0.08", ax=ax)
    for n, d in sub.nodes(data=True):
        nx.draw_networkx_nodes(sub, pos, nodelist=[n], node_color=[_fill(d)], node_size=560,
                               alpha=0.92, ax=ax)
    nx.draw_networkx_labels(sub, pos, labels={n: _short_label(n, d) for n, d in sub.nodes(data=True)},
                            font_size=8, ax=ax)
    ax.legend(handles=[mpatches.Patch(color=EDGE_COLORS[e], label=e) for e in seen],
              loc="best", fontsize=8)
    ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _short_label(n: str, data: dict) -> str:
    nt = data.get("ntype")
    if nt == NodeType.THREAT.value:
        return data.get("family", n)
    if nt == NodeType.TECHNIQUE.value:
        return data.get("tech_id", n)
    if nt == NodeType.ASSET.value:
        return str(data.get("asset_id", n))
    if nt == NodeType.ACTOR.value:
        return str(data.get("actor_id", n))
    if nt == NodeType.FAMILY.value:
        return f"{data.get('dominant_family', '')} (k{data.get('cluster', '')})"
    return str(n)
