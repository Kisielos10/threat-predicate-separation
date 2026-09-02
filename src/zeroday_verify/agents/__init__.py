"""Multi-agent system (initial version) built on the verified detector and threat graph.

Three roles from the architecture document (Planista, Wykonawca, Audytor) coordinate over a
deterministic tool layer and emit the contribution document's output triple (y, c, r) together
with a full audit trail. Two baselines (bare detector, single agent) allow a like-for-like
comparison on identical cases.
"""

from .cases import load_evaluation_setup
from .llm import LLMClient
from .orchestrator import run_detector, run_mas, run_single_agent
from .tools import Case, ToolContext
from .trace import CaseResult, Decision

__all__ = [
    "LLMClient", "Case", "ToolContext", "CaseResult", "Decision",
    "load_evaluation_setup", "run_detector", "run_mas", "run_single_agent",
]
