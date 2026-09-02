"""Minimum evidentiary requirements (wkład W4).

Contribution W4 requires "minimalne wymagania dowodowe dla poszczególnych poziomów reakcji".
This module makes that requirement explicit and machine-checkable: before a verdict may stand,
the evidence chain must contain particular kinds of evidence.

The check is deterministic on purpose. Whether the chain is *sufficient* is a formal property of
the evidence, so it is verifiable and auditable; only the *interpretation* of the evidence is left
to the language model. When the requirement is unmet the Auditor sends the case back, which is the
cross-check loop (W2) actually being exercised rather than merely available.
"""

from __future__ import annotations

# verdict -> tools whose output must be present in the evidence chain
REQUIRED_EVIDENCE: dict[str, set[str]] = {
    # asserting an attack is the highest-consequence verdict: it needs both direct payload
    # evidence and the detector's deviation reading
    "atak": {"payload_inspect", "novelty_score"},
    # clearing traffic still requires having actually looked at the request content
    "normalne": {"payload_inspect"},
    # "unknown" is the honest fallback and carries no evidentiary floor
    "nieznane": set(),
}

HUMAN_NAMES = {
    "payload_inspect": "inspekcja ładunku żądania",
    "novelty_score": "miara nowości u z detektora",
    "similar_known_threats": "porównanie ze znanymi przypadkami",
    "graph_context": "kontekst z grafowej bazy zagrożeń",
    "request_statistics": "statystyki strukturalne żądania",
}


def evidence_gap(verdict: str, tools_used: set[str]) -> set[str]:
    """Which required evidence is missing for this verdict (empty set = requirement satisfied)."""
    required = REQUIRED_EVIDENCE.get(str(verdict).strip().lower(), set())
    return required - set(tools_used)


def describe_gap(gap: set[str]) -> str:
    return ", ".join(HUMAN_NAMES.get(g, g) for g in sorted(gap))


def requirement_text() -> str:
    """Human-readable statement of the policy, shown to the Auditor."""
    lines = []
    for verdict, req in REQUIRED_EVIDENCE.items():
        need = describe_gap(req) if req else "brak dodatkowych wymagań"
        lines.append(f"  - decyzja '{verdict}' wymaga: {need}")
    return "Minimalne wymagania dowodowe (W4):\n" + "\n".join(lines)
