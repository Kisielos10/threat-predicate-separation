"""Comparative evaluation: bare detector vs single agent vs multi-agent, on identical cases.

An "alert" is raised when the decision is `atak` or `nieznane`: in an open-world setting an
unknown-but-suspicious case is escalated, not silently dropped. Strict metrics counting only
`atak` are reported alongside, so the choice of convention is visible rather than hidden.
"""

from __future__ import annotations

import json
import os

from .llm import LLMClient
from .orchestrator import run_detector, run_mas, run_single_agent
from .tools import Case, ToolContext
from .trace import CaseResult

CONFIGS = ("detector", "single_agent", "mas")

# `mas` runs a deliberately short first pass (2 tool calls) so the evidence chain can fall short
# of the W4 minimum and the send-back loop is genuinely exercised. That thinning is a confound:
# it also starves the Auditor in the cases where the gate does *not* fire. `mas_full` gives the
# executor the same budget as the single agent (4 calls), which separates the effect of role
# separation from the effect of the thinning.
MAS_FIRST_PASS = {"mas": 2, "mas_full": 4}


def evaluate(ctx: ToolContext, cases: list[Case], threshold: float, llm: LLMClient,
             configs: tuple[str, ...] = CONFIGS, verbose: bool = True,
             run: int = 0) -> list[CaseResult]:
    """Run every case through every configuration (one repetition of the experiment)."""
    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        for cfg in configs:
            try:
                if cfg == "detector":
                    r = run_detector(ctx, case, threshold)
                elif cfg == "single_agent":
                    r = run_single_agent(llm, ctx, case)
                else:
                    r = run_mas(llm, ctx, case, first_pass_tools=MAS_FIRST_PASS.get(cfg, 2))
                    r.config = cfg
            except Exception as e:                      # noqa: BLE001
                # One unrecoverable LLM failure must not discard a multi-hour run. The case is
                # skipped and reported, so the loss is visible rather than silently averaged in.
                print(f"  [POMINIETO] {case.case_id} / {cfg}: {type(e).__name__}: {e}",
                      flush=True)
                continue
            r.run = run
            results.append(r)
            if verbose and cfg.startswith("mas"):
                ok = "OK " if _correct(r) else "BŁĄD"
                sb = f" [odesłanie: {','.join(r.evidence_gap)}]" if r.audit_rounds else ""
                print(f"  run{run} [{i}/{len(cases)}] {case.case_id:14s} -> "
                      f"{r.decision.decision:9s} ({ok}) {r.seconds:5.1f}s{sb}", flush=True)
    return results


def _correct(r: CaseResult) -> bool:
    return (r.label == "attack") == r.decision.is_alert


def metrics_by_config(results: list[CaseResult]) -> dict:
    """TPR/FPR/Precision/F1 plus cost metrics, per configuration."""
    out: dict[str, dict] = {}
    for cfg in sorted({r.config for r in results}):
        rs = [r for r in results if r.config == cfg]
        tp = sum(1 for r in rs if r.label == "attack" and r.decision.is_alert)
        fn = sum(1 for r in rs if r.label == "attack" and not r.decision.is_alert)
        fp = sum(1 for r in rs if r.label == "benign" and r.decision.is_alert)
        tn = sum(1 for r in rs if r.label == "benign" and not r.decision.is_alert)
        tpr = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0
        strict_tp = sum(1 for r in rs if r.label == "attack" and r.decision.decision == "atak")
        out[cfg] = {
            "n": len(rs), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "TPR": round(tpr, 3), "FPR": round(fpr, 3), "Precision": round(prec, 3),
            "F1": round(f1, 3), "Accuracy": round((tp + tn) / len(rs), 3) if rs else 0.0,
            "TPR_strict_atak": round(strict_tp / (tp + fn), 3) if tp + fn else 0.0,
            "unknown_rate": round(sum(1 for r in rs if r.decision.decision == "nieznane") / len(rs), 3),
            "mean_confidence": round(sum(r.decision.confidence for r in rs) / len(rs), 3),
            "mean_seconds": round(sum(r.seconds for r in rs) / len(rs), 2),
            "mean_llm_calls": round(sum(r.llm_calls for r in rs) / len(rs), 2),
            "audit_sendbacks": sum(r.audit_rounds for r in rs),
        }
    return out


def confidence_calibration(results: list[CaseResult]) -> dict:
    """Mean confidence when right vs when wrong (a well-calibrated system is less sure when wrong)."""
    out = {}
    for cfg in sorted({r.config for r in results}):
        rs = [r for r in results if r.config == cfg]
        right = [r.decision.confidence for r in rs if _correct(r)]
        wrong = [r.decision.confidence for r in rs if not _correct(r)]
        out[cfg] = {
            "mean_conf_correct": round(sum(right) / len(right), 3) if right else None,
            "mean_conf_wrong": round(sum(wrong) / len(wrong), 3) if wrong else None,
            "n_correct": len(right), "n_wrong": len(wrong),
        }
    return out


def sendback_stats(results: list[CaseResult], config: str = "mas") -> dict:
    """How often the W4 evidentiary gate fired, and what it changed."""
    rs = [r for r in results if r.config == config]
    if not rs:
        return {}
    fired = [r for r in rs if r.audit_rounds > 0]
    flipped = [r for r in fired if r.flipped]
    gaps: dict[str, int] = {}
    for r in fired:
        for g in r.evidence_gap:
            gaps[g] = gaps.get(g, 0) + 1
    # A flip whose final decision is correct is NOT evidence that the flip helped: most flips are
    # `atak` -> `nieznane`, and both count as an alert, so alert-level correctness is unchanged.
    # These two counters ask the sharper question: did the flip cross the alert boundary, and in
    # which direction?
    corrected = broke = 0
    directions: dict[str, int] = {}
    for r in flipped:
        directions[f"{r.prelim_decision} -> {r.decision.decision}"] = directions.get(
            f"{r.prelim_decision} -> {r.decision.decision}", 0) + 1
        truth = r.label == "attack"
        before = r.prelim_decision in ("atak", "nieznane")
        after = r.decision.is_alert
        if after == truth and before != truth:
            corrected += 1
        elif before == truth and after != truth:
            broke += 1
    return {
        "n_mas_cases": len(rs),
        "sendback_count": len(fired),
        "sendback_rate": round(len(fired) / len(rs), 3),
        "decision_flipped_after_sendback": len(flipped),
        "final_decision_correct_after_flip": sum(
            1 for r in flipped if (r.label == "attack") == r.decision.is_alert),
        "flip_corrected_alert_level": corrected,
        "flip_broke_alert_level": broke,
        "flip_directions": directions,
        "missing_evidence_counts": gaps,
    }


def condition_stats(results: list[CaseResult], config: str = "mas") -> dict:
    """How the Definition 5 conditions W1/W2/W3 were assessed, by ground truth."""
    rs = [r for r in results if r.config == config and r.conditions]
    out: dict[str, dict] = {}
    for lab in ("attack", "benign"):
        sub = [r for r in rs if r.label == lab]
        if sub:
            out[lab] = {w: round(sum(1 for r in sub if r.conditions.get(w)) / len(sub), 3)
                        for w in ("W1", "W2", "W3")}
            out[lab]["n"] = len(sub)
    return out


def _mean_ci(values: list[float]) -> dict:
    a = [v for v in values if v == v]
    if not a:
        return {"mean": float("nan"), "ci95": float("nan"), "n": 0}
    m = sum(a) / len(a)
    if len(a) < 2:
        return {"mean": round(m, 3), "ci95": 0.0, "n": len(a)}
    sd = (sum((x - m) ** 2 for x in a) / (len(a) - 1)) ** 0.5
    return {"mean": round(m, 3), "ci95": round(1.96 * sd / len(a) ** 0.5, 3), "n": len(a)}


def metrics_across_runs(results: list[CaseResult]) -> dict:
    """Per-run metrics aggregated to mean +/- 95% CI (the form used in the earlier reports)."""
    runs = sorted({r.run for r in results})
    per_run = {cfg: {} for cfg in sorted({r.config for r in results})}
    for run in runs:
        m = metrics_by_config([r for r in results if r.run == run])
        for cfg, d in m.items():
            for k, v in d.items():
                per_run.setdefault(cfg, {}).setdefault(k, []).append(v)
    return {cfg: {k: _mean_ci(v) for k, v in ks.items()} for cfg, ks in per_run.items()}


def save_results(results: list[CaseResult], out_dir: str = "results/agents") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as fh:
        json.dump([r.to_dict() for r in results], fh, indent=2, ensure_ascii=False)
    summary = {
        "metrics": metrics_by_config(results),                 # pooled over all runs
        "metrics_across_runs": metrics_across_runs(results),   # mean +/- 95% CI
        "calibration": confidence_calibration(results),
        "sendback": sendback_stats(results),
        "conditions_W1_W2_W3": condition_stats(results),
        "n_runs": len(sorted({r.run for r in results})),
    }
    # the same two breakdowns for every multi-agent variant present (mas, mas_full, ...)
    mas_cfgs = sorted({r.config for r in results if r.config.startswith("mas")})
    summary["sendback_by_config"] = {c: sendback_stats(results, c) for c in mas_cfgs}
    summary["conditions_by_config"] = {c: condition_stats(results, c) for c in mas_cfgs}
    with open(f"{out_dir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def load_results(path: str = "results/agents/results.json") -> list[dict]:
    with open(path) as fh:
        return json.load(fh)
