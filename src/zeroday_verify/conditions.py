"""Independent evaluation of the Definition 5 threat-predicate conditions.

Why this module exists separately from `agents/`. In the multi-agent system the Auditor emitted
the verdict and all three conditions in a single JSON response. Those are therefore not three
measurements of three conditions; they are four fields of one judgement, and the apparent
separation between conditions may simply be one overall impression leaking into every field.
Standard practice for model-as-judge evaluation is to keep a single call to a small number of
dimensions for exactly this reason.

The design here is deliberately minimal:

  * every condition is judged in its OWN call;
  * every condition sees the SAME evidence bundle, so the only thing that varies between
    conditions is the question asked, not the information available;
  * no call is ever asked whether the request is an attack, and no call is shown any verdict or
    any other condition's answer.

The evidence itself is produced by deterministic code (structural comparison against a profile of
normal traffic, plus the novelty measure). The model is used only to interpret it, which is what
makes it an instrument here rather than a proposed detection system.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qsl, unquote_plus

from .data_text import parse_request

CONDITIONS = ("W1", "W2", "W3")

# One question per condition, phrased as closely as possible to Definition 5 and deliberately
# NOT mentioning attacks, threats, maliciousness, or any verdict.
QUESTIONS: dict[str, str] = {
    # Def 5 W1: "obserwowalny efekt odbiegający od normy". The first draft of this question was
    # read as asking only about structural conformity: the model answered "no deviation" for a
    # request whose parameter carried an SQL statement, because path and parameter names matched.
    # The wording therefore names both channels explicitly.
    "W1": ("Does this request differ from what is normally observed for this application, "
           "either in its structure (path, parameter names, method) or in the content of its "
           "parameter values?"),
    "W2": ("Does this request have the potential to violate the confidentiality, integrity or "
           "availability of the application or its data?"),
    # Def 5 W3: "pochodzenie pozanormatywne (celowe działanie, a nie awaria)". Rendering this as
    # "deliberate action" alone made it degenerate, since every HTTP request is deliberate in
    # that sense; the discriminating half is "pozanormatywne", i.e. outside sanctioned use.
    "W3": ("Does this request originate from outside the application's normal, sanctioned use, "
           "rather than from ordinary legitimate use or from a malfunction?"),
}

_SYS = ("You are assessing one specific property of an HTTP request against evidence collected "
        "by deterministic tools. Answer only the question asked. Do not decide whether the "
        "request is an attack, and do not comment on any property other than the one asked "
        "about. Reply with JSON only.")


@dataclass
class ConditionResult:
    condition: str
    holds: bool
    reason: str
    model: str
    failed: bool = False      # the model gave no usable answer; excluded from rates


def evidence_bundle(structure, raw_request: str, novelty_u: float | None = None) -> str:
    """The identical evidence every condition evaluator receives.

    `structure` is a StructureProfile fitted on the reference normal split. Deviations are
    rendered in English here so the whole evaluation is language-consistent and reportable.
    """
    p = parse_request(raw_request)
    path, method = p["path"], p["method"]
    params = parse_qsl(p["query"]) + parse_qsl(p["body"])
    decoded = "; ".join(f"{k}={unquote_plus(v)}" for k, v in params) or "(none)"

    lines = [f"Method: {method}", f"Path: {path}", f"Parameters: {decoded[:400]}"]

    if structure is not None:
        known = structure.params.get(path)
        if known is None:
            lines.append("Structure: this path does not occur in normal traffic at all.")
        else:
            unknown = [k for k, _ in params if k not in known]
            if unknown:
                lines.append(f"Structure: parameter(s) {unknown} never occur on this path. "
                             f"Known parameters here: {sorted(known) or '(none)'}.")
            else:
                lines.append("Structure: path, parameter names and method all match normal "
                             "traffic for this application.")
    if novelty_u is not None:
        lines.append(f"Distance from the normal-traffic profile: {novelty_u:.4f} "
                     f"(0 = identical to normal traffic, 1 = maximally unlike it).")
    return "\n".join(lines)


def evaluate_condition(llm, condition: str, evidence: str) -> ConditionResult:
    """Judge a single condition, blind to the verdict and to the other conditions.

    A model that returns no usable answer yields a result flagged `failed`. Such results are
    excluded from the reported rates rather than being counted as "condition does not hold",
    which would bias every rate downward by the failure rate. The count of exclusions is
    reported alongside the rates so the reader can see how often it happened.
    """
    model = getattr(llm, "model", "?")
    try:
        out = llm.chat_json(_SYS, (
            f"Evidence:\n{evidence}\n\n"
            f"Question: {QUESTIONS[condition]}\n\n"
            'Reply exactly: {"holds": true or false, "reason": "one short sentence"}'
        ))
    except Exception as exc:                                        # noqa: BLE001
        return ConditionResult(condition=condition, holds=False,
                               reason=f"[no usable answer: {type(exc).__name__}]",
                               model=model, failed=True)
    return ConditionResult(condition=condition,
                           holds=bool(out.get("holds")),
                           reason=str(out.get("reason", ""))[:200],
                           model=model)


def evaluate_all(llm, evidence: str) -> dict[str, ConditionResult]:
    """Judge every condition, each in its own call. Order is fixed for reproducibility."""
    return {c: evaluate_condition(llm, c, evidence) for c in CONDITIONS}
