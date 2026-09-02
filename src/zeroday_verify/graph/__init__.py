"""Threat graph database (Definition 14, extended to a labeled property graph).

Constructs a heterogeneous graph G = (V, E) from `Threat` objects:
  node types  : Threat, Technique, Asset, Actor, Family
  edge types  : variant_of, escalates, precedes, targets_same, enables
                (inter-threat, Def 14) + realizes_technique, member_of, targets,
                originates_from (instance <-> knowledge / context links).

This package only *constructs, stores, queries and visualises* the graph. Downstream tasks
(grouping / classification / prediction) build on it and are out of scope here.
"""

from .build import build_enriched_graph, build_graph, build_threat_graph
from .schema import EdgeType, GraphParams, NodeType

__all__ = ["build_graph", "build_threat_graph", "build_enriched_graph",
           "NodeType", "EdgeType", "GraphParams"]
