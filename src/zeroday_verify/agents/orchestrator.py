"""Pipelines under comparison.

  run_detector     — the verified detector alone (novelty threshold, no LLM).
  run_single_agent — one LLM agent with the same tools, no roles, no cross-check.
  run_mas          — Planista -> Wykonawca -> Audytor, where the Auditor may send the case
                     back once for missing evidence (the cross-check mechanism, W2).

All three return a CaseResult with the output triple and a full audit trail, so they can be
compared on identical cases.
"""

from __future__ import annotations

import time

from .llm import LLMClient
from .policy import describe_gap, evidence_gap
from .roles import _emit, audit, execute, plan, single_agent
from .tools import Case, ToolContext, novelty_score
from .trace import CaseResult, Decision, Step


def run_detector(ctx: ToolContext, case: Case, threshold: float) -> CaseResult:
    """Baseline without any LLM: threshold the novelty measure u (Definition 13)."""
    t0 = time.perf_counter()
    res = novelty_score(ctx, case)
    u = float(res["novelty_u"])
    is_attack = u >= threshold
    # confidence grows with distance from the decision threshold, capped at 1
    conf = min(1.0, 0.5 + abs(u - threshold) / max(threshold, 1e-6) * 0.5)
    dec = Decision("atak" if is_attack else "normalne", conf,
                   "Eskalacja do analityka." if is_attack else "Brak działania.").normalised()
    step = Step("Detektor", "verdict", f"u={u:.4f} (próg {threshold:.4f}) -> {dec.decision}")
    return CaseResult(case.case_id, "detector", dec, time.perf_counter() - t0, 0,
                      case.label, case.family, [step])


def run_single_agent(llm: LLMClient, ctx: ToolContext, case: Case) -> CaseResult:
    """Baseline: a single agent with the same tools but no role separation or validation."""
    t0 = time.perf_counter()
    before = llm.stats.n_calls
    dec, steps = single_agent(llm, ctx, case)
    return CaseResult(case.case_id, "single_agent", dec, time.perf_counter() - t0,
                      llm.stats.n_calls - before, case.label, case.family, steps)


def run_mas(llm: LLMClient, ctx: ToolContext, case: Case, on_step=None,
            first_pass_tools: int = 2) -> CaseResult:
    """The multi-agent configuration: plan -> gather evidence -> audit (with one send-back).

    `on_step` receives each step as it happens, so a CLI can stream the reasoning live.
    """
    t0 = time.perf_counter()
    before = llm.stats.n_calls
    steps: list[Step] = []

    plan_text, plan_step = plan(llm, case, on_step=on_step)
    steps.append(plan_step)

    # First pass is deliberately short, so the evidence chain can genuinely fall short of the
    # W4 minimum and the cross-check loop has something to do.
    evidence, exec_steps = execute(llm, ctx, case, plan_text, max_tool_calls=first_pass_tools,
                                   on_step=on_step)
    steps.extend(exec_steps)
    prelim = str(evidence.get("wstepna_decyzja", "nieznane")).strip().lower()

    # W4 gate: is the evidence chain sufficient for the verdict the executor proposes?
    gap = evidence_gap(prelim, set(evidence.get("_tools_used", [])))
    rounds = 0
    if gap:
        rounds = 1
        missing = describe_gap(gap)
        _emit(steps, Step("Audytor", "audit",
                          f"Odesłanie (W4): decyzja '{prelim}' wymaga dowodu, którego brakuje: "
                          f"{missing}"), on_step)
        evidence2, exec_steps2 = execute(llm, ctx, case, plan_text, max_tool_calls=3,
                                         extra_request=missing, on_step=on_step)
        # the auditor must see the whole chain, both rounds
        evidence2["_observations"] = (evidence.get("_observations", [])
                                      + evidence2.get("_observations", []))
        evidence2["_tools_used"] = sorted(set(evidence.get("_tools_used", []))
                                          | set(evidence2.get("_tools_used", [])))
        steps.extend(exec_steps2)
        evidence = evidence2

    verdict, audit_step = audit(llm, case, evidence, allow_followup=False, on_step=on_step)
    steps.append(audit_step)

    dec = Decision(verdict.get("decision", "nieznane"), verdict.get("confidence", 0.0),
                   str(verdict.get("recommendation", ""))).normalised()
    return CaseResult(case.case_id, "mas", dec, time.perf_counter() - t0,
                      llm.stats.n_calls - before, case.label, case.family, steps, rounds,
                      prelim_decision=prelim,
                      evidence_gap=sorted(gap),
                      conditions={w: bool(verdict.get(w)) for w in ("W1", "W2", "W3")})
