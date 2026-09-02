"""Deterministic tool layer for the agents (the "Tools" layer of the architecture document).

Agents never look at ground truth. They may only call these tools, each of which is a plain,
deterministic function over data a real SOC would have: the detector's novelty score, retrieval
of similar *known* threats, payload inspection, and ATT&CK lookup.

Scope note: cases are HTTP requests (CSIC-2010 and the mock service). The web domain is where a
language model can actually reason about the evidence (it can read `' OR '1'='1`), and where the
verified detector is strongest. Numeric flow statistics are deliberately out of scope here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import unquote_plus

import numpy as np

from ..data_text import parse_request, serialize_http
from ..embedding import embed_texts
from ..novelty import KnownThreatModel
from ..similarity import WEIGHTS_SEM_ONLY
from ..taxonomy import FAMILY_PROFILE
from ..threats import Threat

# suspicious token families used by `payload_inspect` (evidence, not a verdict)
_SIGNATURES: dict[str, tuple[str, ...]] = {
    "SQL": ("' or ", "union select", "--", "1=1", "waitfor delay", "sleep(", "/*", "0x",
            "' and ", "drop table", "insert into"),
    "XSS": ("<script", "javascript:", "onerror=", "onload=", "alert(", "<iframe",
            "document.cookie"),
    "path_traversal": ("../", "..\\", "/etc/passwd", "%2e%2e", "boot.ini", "windows/system32"),
    "command": (";cat ", "|cat ", "&&", "$(", "/bin/", "cmd.exe", "`", "nc -"),
}


@dataclass
class Case:
    """One incident presented to the system. `label`/`family` are ground truth, never shown."""

    case_id: str
    raw_request: str
    source: str = "csic"
    label: str | None = None       # "attack" | "benign"  (evaluation only)
    family: str | None = None      # heuristic attack type (evaluation only)
    meta: dict = field(default_factory=dict)

    def parsed(self) -> dict:
        return parse_request(self.raw_request)

    def summary(self) -> str:
        """Short, label-free description shown to the agents."""
        return serialize_http(self.parsed())


@dataclass
class ToolContext:
    """Reference knowledge the tools query: known-normal profile + known threat corpus + graph."""

    normal_model: KnownThreatModel
    known_threats: list[Threat]
    known_phi: np.ndarray
    graph: object | None = None          # networkx threat graph over known_threats (Def 14)
    structure: object | None = None      # StructureProfile of normal traffic (payload_inspect)
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def embed(self, text: str) -> np.ndarray:
        return embed_texts([text], model_name=self.model_name)[0]


def build_tool_context(normal_threats: list[Threat], known_threats: list[Threat],
                       min_cluster_size: int = 15, with_graph: bool = True,
                       normal_requests: list[str] | None = None) -> ToolContext:
    """Fit the known-normal profile, index known threats, and build the Def 14 threat graph."""
    model = KnownThreatModel(WEIGHTS_SEM_ONLY, min_cluster_size=min_cluster_size).fit(normal_threats)
    phi = np.vstack([t.phi for t in known_threats]) if known_threats else np.empty((0, 384))
    graph = None
    if with_graph and known_threats:
        try:
            from ..graph import build_threat_graph
            from ..graph.schema import GraphParams
            attacks = [t for t in known_threats if t.is_attack]
            if attacks:
                graph = build_threat_graph(attacks, GraphParams(tau_var=0.9, min_cluster_size=5))
        except Exception:
            graph = None    # the graph is an enrichment; its absence must not break the run
    structure = None
    if normal_requests:
        from .structure import StructureProfile
        structure = StructureProfile.fit(normal_requests)
    return ToolContext(normal_model=model, known_threats=known_threats, known_phi=phi, graph=graph,
                       structure=structure)


# ---------------------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------------------

def novelty_score(ctx: ToolContext, case: Case) -> dict:
    """Detector signal: how far this request is from the known-normal profile (Def 13)."""
    th = _case_as_threat(ctx, case)
    u = float(ctx.normal_model.novelty(th))
    if u < 0.02:
        reading = "bardzo blisko profilu normalnego"
    elif u < 0.06:
        reading = "umiarkowanie odbiega od profilu normalnego"
    else:
        reading = "wyraźnie odbiega od profilu normalnego"
    return {"novelty_u": round(u, 4), "interpretation": reading,
            "scale": "0 = identyczne z ruchem normalnym, 1 = całkowicie nowe"}


def similar_known_threats(ctx: ToolContext, case: Case, k: int = 3) -> dict:
    """Retrieve the k most similar *known, already-labelled* cases (case-based reasoning).

    The corpus deliberately contains both known attacks and known normal traffic, so a match of
    type "normal" is an informative answer rather than an impossible one.
    """
    if not ctx.known_threats:
        return {"matches": []}
    v = ctx.embed(case.summary())
    v = v / (np.linalg.norm(v) or 1.0)
    ref = ctx.known_phi / np.clip(np.linalg.norm(ctx.known_phi, axis=1, keepdims=True), 1e-9, None)
    sims = ref @ v
    idx = np.argsort(-sims)[:k]
    return {"matches": [{"known_type": ctx.known_threats[i].family,
                         "similarity": round(float(sims[i]), 3),
                         "example": ctx.known_threats[i].text[:160]} for i in idx]}


def payload_inspect(ctx: ToolContext, case: Case) -> dict:
    """Decode the request and report which suspicious token families appear (raw evidence)."""
    p = case.parsed()
    decoded = f"{unquote_plus(p['query'])} {unquote_plus(p['body'])}".strip()
    low = (decoded + " " + p["path"]).lower()
    hits = {name: [s for s in sigs if s in low] for name, sigs in _SIGNATURES.items()}
    hits = {k: v for k, v in hits.items() if v}
    # Structural evidence: how this request differs from what the page normally receives.
    # Token matching alone is nearly silent on this corpus, where most anomalies are structural.
    dev: list[str] = []
    if ctx.structure is not None:
        dev = ctx.structure.deviations(case.raw_request)
    return {"method": p["method"], "path": p["path"],
            "decoded_parameters": decoded[:400] or "(brak)",
            "suspicious_tokens": hits or "(brak dopasowań)",
            "structural_deviations": dev or "(brak: ścieżka, parametry i metoda są zgodne "
                                            "z ruchem normalnym)",
            "note": "Dopasowanie tokenów oraz odchylenia strukturalne są przesłankami, "
                    "nie dowodem. Nietypowa wartość parametru nie musi oznaczać ataku."}


def attack_technique_info(ctx: ToolContext, technique_or_family: str) -> dict:
    """Look up the ATT&CK techniques and CIA impact associated with a known attack family."""
    key = technique_or_family.strip()
    for fam, (techs, sigma) in FAMILY_PROFILE.items():
        if fam.lower() == key.lower():
            return {"family": fam, "attack_techniques": sorted(techs),
                    "cia_impact": {"confidentiality": sigma[0], "integrity": sigma[1],
                                   "availability": sigma[2]}}
    return {"error": f"nieznana rodzina '{technique_or_family}'",
            "known_families": sorted(FAMILY_PROFILE)}


def request_statistics(ctx: ToolContext, case: Case) -> dict:
    """Cheap structural statistics of the request (length, parameter count, odd characters)."""
    p = case.parsed()
    blob = p["query"] + p["body"]
    return {"n_parameters": blob.count("="), "length": len(blob),
            "special_char_ratio": round(sum(c in "'\"<>;|&%()" for c in blob) / max(len(blob), 1), 3),
            "encoded_sequences": len(re.findall(r"%[0-9a-fA-F]{2}", blob))}


def graph_context(ctx: ToolContext, case: Case) -> dict:
    """Query the threat-graph database (Def 14) around the closest known threat.

    Reports how many known variants that threat has (`variant_of`) and which other known threats
    hit the same asset (`targets_same`) - i.e. whether this looks like part of a known family or
    a known campaign against a known target.
    """
    if ctx.graph is None or not ctx.known_threats:
        return {"error": "graf zagrożeń niedostępny"}
    v = ctx.embed(case.summary())
    v = v / (np.linalg.norm(v) or 1.0)
    ref = ctx.known_phi / np.clip(np.linalg.norm(ctx.known_phi, axis=1, keepdims=True), 1e-9, None)
    sims = ref @ v
    best = int(np.argmax(sims))
    nearest = ctx.known_threats[best]

    node = None
    for n, d in ctx.graph.nodes(data=True):
        if d.get("text", "")[:80] == nearest.text[:80]:
            node = n
            break
    if node is None:
        return {"nearest_known_type": nearest.family,
                "similarity": round(float(sims[best]), 3),
                "note": "najbliższe znane zagrożenie nie występuje w grafie (ruch normalny)"}

    counts: dict[str, int] = {}
    for u, w, d in ctx.graph.edges(data=True):
        if node in (u, w):
            counts[d.get("etype", "?")] = counts.get(d.get("etype", "?"), 0) + 1
    return {"nearest_known_type": nearest.family,
            "similarity": round(float(sims[best]), 3),
            "relations_in_graph": counts or "(brak powiązań)",
            "note": "variant_of = liczba znanych wariantów, targets_same = wspólny zasób"}


TOOLS = {
    "novelty_score": (novelty_score, "Zwraca miarę nowości u dla żądania (odległość od profilu normalnego). Bez argumentów."),
    "payload_inspect": (payload_inspect, "Dekoduje żądanie i wskazuje podejrzane tokeny. Bez argumentów."),
    "similar_known_threats": (similar_known_threats, "Zwraca najbardziej podobne ZNANE przypadki wraz z ich typem. Bez argumentów."),
    "graph_context": (graph_context, "Sprawdza w grafowej bazie zagrożeń powiązania najbliższego znanego zagrożenia (warianty, wspólny zasób). Bez argumentów."),
    "request_statistics": (request_statistics, "Statystyki strukturalne żądania. Bez argumentów."),
    "attack_technique_info": (attack_technique_info, "Techniki ATT&CK i wpływ CIA dla podanej rodziny ataku. Argument: nazwa rodziny, np. WebAttack."),
}

TOOLS_WITH_ARG = {"attack_technique_info"}


def tool_catalogue() -> str:
    return "\n".join(f"- {name}: {desc}" for name, (_, desc) in TOOLS.items())


def run_tool(name: str, ctx: ToolContext, case: Case, argument: str | None = None) -> dict:
    """Dispatch a tool call by name; returns a plain dict (or an error dict)."""
    if name not in TOOLS:
        return {"error": f"nieznane narzędzie '{name}'", "available": sorted(TOOLS)}
    fn, _ = TOOLS[name]
    try:
        if name in TOOLS_WITH_ARG:
            return fn(ctx, argument or "")
        return fn(ctx, case)
    except Exception as e:  # a broken tool must not kill the run
        return {"error": f"narzędzie '{name}' zgłosiło błąd: {e}"}


def _case_as_threat(ctx: ToolContext, case: Case) -> Threat:
    text = case.summary()
    th = Threat(events=[], K=("application", "HTTP"), Tg=("protected_asset", case.parsed()["path"]),
                D="Web", A=np.zeros(11), sigma=np.zeros(3), m_vec=np.zeros(3),
                raw_vec=np.zeros(1), text=text, family="?", is_attack=False)
    th.phi = ctx.embed(text)
    return th
