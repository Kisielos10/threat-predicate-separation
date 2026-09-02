"""Self-contained interactive Cytoscape.js visualisation of the threat graph.

Produces a single standalone HTML file (the cytoscape library is embedded, so it works offline
at a defence). Threat nodes are coloured by attack family, edges by relation type. The page has
a legend, per-relation toggle filters, layout buttons, and a click-to-inspect panel.
"""

from __future__ import annotations

import json
import os

import networkx as nx

from .schema import EDGE_COLORS, FAMILY_COLORS, FAMILY_FALLBACK, NODE_COLORS, EdgeType, NodeType

_ASSETS = os.path.join(os.path.dirname(__file__), "assets", "cytoscape.min.js")
# relations shown by default; the dense intra-family ones start hidden (toggle to reveal)
_DEFAULT_ON = {EdgeType.PRECEDES.value, EdgeType.ESCALATES.value, EdgeType.ENABLES.value,
               EdgeType.TARGETS_SAME.value, EdgeType.REALIZES_TECHNIQUE.value,
               EdgeType.MEMBER_OF.value, EdgeType.TARGETS.value, EdgeType.ORIGINATES_FROM.value}
_DEFAULT_OFF = {EdgeType.VARIANT_OF.value}


def _node_color(d: dict) -> str:
    if d.get("ntype") == NodeType.THREAT.value:
        return FAMILY_COLORS.get(d.get("family"), FAMILY_FALLBACK)
    return NODE_COLORS.get(d.get("ntype"), "#888888")


def _node_label(n: str, d: dict) -> str:
    nt = d.get("ntype")
    if nt == NodeType.THREAT.value:
        return d.get("family", "")
    for key in ("tech_id", "asset_id", "actor_id", "dominant_family"):
        if key in d:
            return str(d[key])
    return str(n)


def _tooltip(d: dict) -> str:
    keys = ["ntype", "family", "technique_str", "sigma_norm", "novelty", "asset", "actor",
            "tactic", "domain", "channel"]
    return " | ".join(f"{k}={d[k]}" for k in keys if k in d and d[k] not in ("", None))


def to_elements(G: nx.MultiDiGraph) -> list[dict]:
    els = []
    for n, d in G.nodes(data=True):
        nov = float(d.get("novelty", 0.0)) if d.get("ntype") == NodeType.THREAT.value else 0.4
        size = 16 + 40 * nov if d.get("ntype") == NodeType.THREAT.value else 30
        els.append({"data": {"id": str(n), "label": _node_label(n, d), "color": _node_color(d),
                             "size": size, "ntype": d.get("ntype", ""), "info": _tooltip(d)}})
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        et = d.get("etype", "rel")
        els.append({"data": {"id": f"e{i}", "source": str(u), "target": str(v), "etype": et,
                            "color": EDGE_COLORS.get(et, "#cccccc")}, "classes": et})
    return els


def interactive_html(G: nx.MultiDiGraph, path: str, title: str = "Graf zagrożeń") -> None:
    with open(_ASSETS) as fh:
        lib = fh.read()
    elements = json.dumps(to_elements(G))
    present_rel = sorted({d.get("etype") for _, _, d in G.edges(data=True)})
    fam_present = sorted({d.get("family") for _, d in G.nodes(data=True)
                          if d.get("ntype") == NodeType.THREAT.value and d.get("family")})

    rel_checks = "".join(
        f'<label class="chk"><input type="checkbox" data-rel="{r}" '
        f'{"checked" if r in _DEFAULT_ON else ""}> '
        f'<span class="sw" style="background:{EDGE_COLORS.get(r, "#ccc")}"></span>{r}</label>'
        for r in present_rel)
    fam_legend = "".join(
        f'<span class="lg"><span class="dot" style="background:{FAMILY_COLORS.get(f, FAMILY_FALLBACK)}">'
        f'</span>{f}</span>' for f in fam_present)
    off = json.dumps(sorted(_DEFAULT_OFF))

    html = _TEMPLATE.replace("__LIB__", lib).replace("__ELEMENTS__", elements) \
        .replace("__REL_CHECKS__", rel_checks).replace("__FAM_LEGEND__", fam_legend) \
        .replace("__TITLE__", title).replace("__DEFAULT_OFF__", off)
    with open(path, "w") as fh:
        fh.write(html)


_TEMPLATE = r"""<!DOCTYPE html><html lang="pl"><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { --ink:#1f2933; --line:#e3e8ee; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink); background:#fafbfc; }
  header { padding:14px 20px; border-bottom:1px solid var(--line); background:#fff; }
  header h1 { margin:0; font-size:18px; font-weight:650; }
  header p { margin:4px 0 0; font-size:12.5px; color:#66707a; }
  #wrap { display:flex; height:calc(100vh - 64px); }
  #cy { flex:1; background:radial-gradient(circle at 40% 30%, #ffffff, #f1f4f8); }
  #side { width:280px; border-left:1px solid var(--line); background:#fff; padding:16px;
          overflow:auto; font-size:13px; }
  #side h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:#8a949e;
             margin:18px 0 8px; }
  #side h2:first-child { margin-top:0; }
  .chk { display:flex; align-items:center; gap:8px; padding:3px 0; cursor:pointer; }
  .sw { width:22px; height:4px; border-radius:2px; display:inline-block; }
  .lg { display:inline-flex; align-items:center; gap:6px; margin:0 10px 6px 0; font-size:12.5px; }
  .dot { width:11px; height:11px; border-radius:50%; display:inline-block; }
  .btn { border:1px solid var(--line); background:#fff; border-radius:7px; padding:6px 10px;
         font-size:12.5px; cursor:pointer; margin:0 6px 6px 0; }
  .btn:hover { background:#f1f4f8; }
  #info { margin-top:10px; font-size:12px; color:#3a4651; background:#f7f9fb; border-radius:8px;
          padding:10px; min-height:42px; white-space:pre-wrap; word-break:break-word; }
</style></head><body>
<header><h1>__TITLE__</h1>
<p>Węzeł = zagrożenie (kolor = rodzina). Krawędź = relacja (kolor wg legendy). Kliknij węzeł, aby
zobaczyć szczegóły; użyj filtrów, aby pokazać/ukryć typy relacji.</p></header>
<div id="wrap"><div id="cy"></div>
<div id="side">
  <h2>Rodziny zagrożeń</h2><div>__FAM_LEGEND__</div>
  <h2>Relacje (filtr)</h2><div id="rels">__REL_CHECKS__</div>
  <h2>Układ</h2>
  <button class="btn" data-layout="cose">siłowy</button>
  <button class="btn" data-layout="concentric">koncentryczny</button>
  <button class="btn" data-layout="circle">okrąg</button>
  <h2>Szczegóły</h2><div id="info">— kliknij węzeł —</div>
</div></div>
<script>__LIB__</script>
<script>
const elements = __ELEMENTS__;
const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: elements,
  wheelSensitivity: 0.2,
  style: [
    { selector: 'node', style: {
        'background-color': 'data(color)', 'width': 'data(size)', 'height': 'data(size)',
        'border-width': 1, 'border-color': '#ffffff' } },
    { selector: 'node:selected', style: {
        'label': 'data(label)', 'font-size': 11, 'color': '#111', 'text-valign': 'top',
        'text-outline-color': '#fff', 'text-outline-width': 2, 'border-width': 3,
        'border-color': '#111', 'z-index': 99 } },
    { selector: 'edge', style: {
        'line-color': 'data(color)', 'width': 1.4, 'curve-style': 'bezier',
        'target-arrow-color': 'data(color)', 'target-arrow-shape': 'triangle',
        'arrow-scale': 0.7, 'opacity': 0.65 } },
    { selector: 'node:selected', style: { 'border-width': 3, 'border-color': '#111' } },
    { selector: '.hidden', style: { 'display': 'none' } }
  ],
  layout: { name: 'cose', animate: false, nodeRepulsion: 9000, idealEdgeLength: 70 }
});
// hide dense relations by default
__DEFAULT_OFF__.forEach(r => cy.edges('.' + r).addClass('hidden'));
document.querySelectorAll('#rels input').forEach(cb => {
  cb.addEventListener('change', e => {
    const rel = e.target.getAttribute('data-rel');
    cy.edges('.' + rel)[e.target.checked ? 'removeClass' : 'addClass']('hidden');
  });
});
document.querySelectorAll('[data-layout]').forEach(b => {
  b.addEventListener('click', () => cy.layout({ name: b.getAttribute('data-layout'),
      animate: true, animationDuration: 500 }).run());
});
cy.on('tap', 'node', e => {
  document.getElementById('info').textContent =
    e.target.id() + '\n' + (e.target.data('info') || '');
});
</script></body></html>
"""
