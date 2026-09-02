"""Persistence for the threat graph: GraphML, node-link JSON, Cypher export, optional Neo4j load.

GraphML/JSON are the reproducible canonical formats (GraphML also opens directly in Gephi).
The Cypher export drives Neo4j (Browser/Bloom) and works without a running server; `load_neo4j`
is an opt-in live load if a server is available.

The embedding `phi` is kept out of the serialized graph (GraphML cannot store arrays); use
`save_embeddings` to dump it separately when needed.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np

_SCALAR = (str, int, float, bool)


def _scalarize(value: object) -> object:
    """Convert an attribute value to something GraphML/JSON-friendly (drop arrays to strings)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    if isinstance(value, np.ndarray):
        return ",".join(str(float(x)) for x in value.ravel())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, _SCALAR):
        return value
    return str(value)


def _sanitized_copy(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Copy the graph with serialisation-safe attributes and no `phi`."""
    H = nx.MultiDiGraph()
    for n, data in G.nodes(data=True):
        H.add_node(n, **{k: _scalarize(v) for k, v in data.items() if k != "phi"})
    for u, v, data in G.edges(data=True):
        H.add_edge(u, v, **{k: _scalarize(val) for k, val in data.items()})
    return H


def save_graphml(G: nx.MultiDiGraph, path: str) -> None:
    nx.write_graphml(_sanitized_copy(G), path)


def load_graphml(path: str) -> nx.MultiDiGraph:
    return nx.read_graphml(path, force_multigraph=True)


def save_json(G: nx.MultiDiGraph, path: str) -> None:
    data = nx.node_link_data(_sanitized_copy(G), edges="links")
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def load_json(path: str) -> nx.MultiDiGraph:
    with open(path) as fh:
        data = json.load(fh)
    return nx.node_link_graph(data, multigraph=True, directed=True, edges="links")


def save_embeddings(G: nx.MultiDiGraph, npy_path: str, ids_path: str) -> None:
    """Dump the Threat-node phi vectors as a matrix + an aligned id list."""
    ids, vecs = [], []
    for n, data in G.nodes(data=True):
        if "phi" in data and data["phi"] is not None:
            ids.append(n)
            vecs.append(np.asarray(data["phi"], dtype=np.float64))
    np.save(npy_path, np.vstack(vecs) if vecs else np.empty((0, 0)))
    with open(ids_path, "w") as fh:
        json.dump(ids, fh)


# --- Neo4j ------------------------------------------------------------------------------

def _cypher_props(data: dict) -> str:
    parts = []
    for k, v in data.items():
        if k in ("ntype", "phi"):
            continue
        v = _scalarize(v)
        if isinstance(v, bool):
            parts.append(f"{k}: {str(v).lower()}")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            parts.append(f"{k}: {v}")
        else:
            s = str(v).replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'{k}: "{s}"')
    return "{" + ", ".join(parts) + "}" if parts else "{}"


def export_cypher(G: nx.MultiDiGraph, path: str) -> None:
    """Write a Neo4j Cypher script (MERGE nodes by id + label, then relationships)."""
    lines = ["// Threat graph database — Neo4j import script",
             "// run with: cypher-shell -f threat_graph.cypher  (or paste into Neo4j Browser)"]
    for n, data in G.nodes(data=True):
        label = data.get("ntype", "Node")
        nid = str(n).replace('"', '\\"')
        props = _cypher_props({"id": nid, **{k: v for k, v in data.items() if k != "ntype"}})
        lines.append(f'MERGE (n:{label} {{id: "{nid}"}}) SET n += {props};')
    for u, v, data in G.edges(data=True):
        rel = str(data.get("etype", "REL")).upper()
        uu, vv = str(u).replace('"', '\\"'), str(v).replace('"', '\\"')
        lines.append(
            f'MATCH (a {{id: "{uu}"}}), (b {{id: "{vv}"}}) MERGE (a)-[:{rel}]->(b);')
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def load_neo4j(G: nx.MultiDiGraph, uri: str, user: str, password: str) -> None:
    """Optional live load into a running Neo4j instance (no-op-safe if driver/server absent)."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for n, data in G.nodes(data=True):
            label = data.get("ntype", "Node")
            props = {k: _scalarize(v) for k, v in data.items() if k not in ("ntype", "phi")}
            props["id"] = str(n)
            session.run(f"MERGE (n:{label} {{id: $id}}) SET n += $props", id=str(n), props=props)
        for u, v, data in G.edges(data=True):
            rel = str(data.get("etype", "REL")).upper()
            session.run(
                f"MATCH (a {{id:$u}}),(b {{id:$v}}) MERGE (a)-[:{rel}]->(b)", u=str(u), v=str(v))
    driver.close()
