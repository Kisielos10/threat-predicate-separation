"""Schema for the threat graph database: node/edge types, parameters, and a shared palette.

The palette is shared by every visualisation route (Cytoscape.js HTML, matplotlib/graphviz
figures) so the colours mean the same thing everywhere, which matters for defence slides.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeType(str, Enum):
    THREAT = "Threat"
    TECHNIQUE = "Technique"
    ASSET = "Asset"
    ACTOR = "Actor"
    FAMILY = "Family"


class EdgeType(str, Enum):
    # inter-threat relations (Definition 14)
    VARIANT_OF = "variant_of"
    ESCALATES = "escalates"
    PRECEDES = "precedes"
    TARGETS_SAME = "targets_same"
    ENABLES = "enables"
    # instance <-> knowledge / context links
    REALIZES_TECHNIQUE = "realizes_technique"
    MEMBER_OF = "member_of"
    TARGETS = "targets"
    ORIGINATES_FROM = "originates_from"


@dataclass(frozen=True)
class GraphParams:
    """Tunable construction parameters."""

    tau_var: float = 0.90          # variant_of: composite-sim threshold
    precedes_window_s: float = 3600.0  # precedes/escalates: max gap between consecutive steps
    min_cluster_size: int = 10     # HDBSCAN min cluster size for Family nodes
    targets_same_cap: int = 30     # max threats per asset to fully connect (else star to earliest)
    # techniques considered "reconnaissance/discovery" for the enables heuristic
    recon_techniques: tuple[str, ...] = ("T1046", "T1595")


# --- shared visual style ---------------------------------------------------------------

NODE_COLORS: dict[str, str] = {
    NodeType.THREAT.value: "#e15759",     # red
    NodeType.TECHNIQUE.value: "#4e79a7",  # blue
    NodeType.ASSET.value: "#59a14f",      # green
    NodeType.ACTOR.value: "#f28e2b",      # orange
    NodeType.FAMILY.value: "#b07aa1",     # purple
}

EDGE_COLORS: dict[str, str] = {
    EdgeType.VARIANT_OF.value: "#9c755f",
    EdgeType.ESCALATES.value: "#e15759",
    EdgeType.PRECEDES.value: "#4e79a7",
    EdgeType.TARGETS_SAME.value: "#59a14f",
    EdgeType.ENABLES.value: "#edc948",
    EdgeType.REALIZES_TECHNIQUE.value: "#bab0ac",
    EdgeType.MEMBER_OF.value: "#b07aa1",
    EdgeType.TARGETS.value: "#86bcb6",
    EdgeType.ORIGINATES_FROM.value: "#f28e2b",
}

# colour per attack family — used to colour Threat nodes in the strict graph (meaningful groups)
FAMILY_COLORS: dict[str, str] = {
    "DoS": "#e15759",
    "DDoS": "#b07aa1",
    "PortScan": "#4e79a7",
    "BruteForce": "#f28e2b",
    "Bot": "#76b7b2",
    "WebAttack": "#59a14f",
    "Infiltration": "#edc948",
    "Heartbleed": "#ff9da7",
}
FAMILY_FALLBACK = "#9c755f"

# node id prefixes
PFX = {
    NodeType.THREAT: "th",
    NodeType.TECHNIQUE: "tech",
    NodeType.ASSET: "asset",
    NodeType.ACTOR: "actor",
    NodeType.FAMILY: "fam",
}
