"""Audit trail structures.

The architecture document requires every decision to carry a verifiable record of what each
agent thought, which tools it called and what it observed. These dataclasses are that record;
they are serialised with each result so a run can be inspected after the fact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

DECISIONS = ("atak", "normalne", "nieznane")


@dataclass
class Step:
    agent: str          # "Planista" | "Wykonawca" | "Audytor" | "Agent" | "Detektor"
    kind: str           # plan | thought | tool | observation | verdict | audit
    content: str
    tool: str | None = None
    seconds: float = 0.0


@dataclass
class Decision:
    """The method's output triple (y_k, c_k, r_k) from the contribution document."""

    decision: str                 # y_k
    confidence: float             # c_k in [0,1]
    recommendation: str = ""      # r_k

    def normalised(self) -> Decision:
        d = str(self.decision).strip().lower()
        if d not in DECISIONS:
            d = "nieznane"
        try:
            c = float(self.confidence)
        except (TypeError, ValueError):
            c = 0.0
        return Decision(d, min(max(c, 0.0), 1.0), self.recommendation)

    @property
    def is_alert(self) -> bool:
        """Treat 'atak' and 'nieznane' as raising an alert (open-world: unknown is escalated)."""
        return self.decision in ("atak", "nieznane")


@dataclass
class CaseResult:
    case_id: str
    config: str                       # detector | single_agent | mas
    decision: Decision
    seconds: float = 0.0
    llm_calls: int = 0
    label: str | None = None          # ground truth, evaluation only
    family: str | None = None
    steps: list[Step] = field(default_factory=list)
    audit_rounds: int = 0             # how many times the auditor sent it back (W4 gate)
    prelim_decision: str = ""         # executor's verdict before the audit
    evidence_gap: list[str] = field(default_factory=list)   # W4 evidence that was missing
    conditions: dict = field(default_factory=dict)          # Def 5 conditions W1/W2/W3
    run: int = 0                      # which repetition of the experiment

    @property
    def flipped(self) -> bool:
        """Did the audit change the executor's preliminary verdict?"""
        return bool(self.prelim_decision) and self.prelim_decision != self.decision.decision

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = asdict(self.decision)
        return d
