"""Evaluation on legitimate traffic that superficially resembles an attack.

The in-distribution experiment measures whether a request can be told apart from normal traffic.
A detector whose model is fitted on normal traffic does that well by construction, so the
comparison says little about the value of reasoning over the payload.

This experiment separates the two abilities. The benign half of the evaluation set contains
ordinary normal traffic and hand-written hard negatives: requests that are unusual but harmless.
The detector flags every hard negative, because they are far from the normal profile. The question
is whether a system that reads the request can clear them without losing attacks.

    ./.venv/bin/python run_hardneg_eval.py [n_attack] [n_plain] [n_hard] [n_runs]

Writes results/agents/hardneg_{results,summary}.json.
"""

from __future__ import annotations

import json
import os
import sys
import time

from zeroday_verify.agents import LLMClient
from zeroday_verify.agents.cases import load_hard_negative_setup
from zeroday_verify.agents.evaluate import evaluate, metrics_by_config
from zeroday_verify.agents.trace import CaseResult

CONFIGS = ("detector", "single_agent", "mas_full")
SEEDS = (17, 23, 42)
OUT = "results/agents"


def breakdown(results: list[CaseResult]) -> dict:
    """Alert rate per configuration, split by the kind of case."""
    out: dict[str, dict] = {}
    for cfg in sorted({r.config for r in results}):
        rs = [r for r in results if r.config == cfg]
        d = {}
        for fam in ("csic_attack", "benign_plain", "benign_hard"):
            sub = [r for r in rs if r.family == fam]
            if sub:
                d[fam] = {"n": len(sub),
                          "alert_rate": round(sum(1 for r in sub if r.decision.is_alert) / len(sub), 3),
                          "atak_rate": round(sum(1 for r in sub if r.decision.decision == "atak") / len(sub), 3)}
        out[cfg] = d
    return out


def _save(results: list[CaseResult]) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/hardneg_results.json", "w") as fh:
        json.dump([r.to_dict() for r in results], fh, indent=2, ensure_ascii=False)
    with open(f"{OUT}/hardneg_summary.json", "w") as fh:
        json.dump({"metrics": metrics_by_config(results), "breakdown": breakdown(results),
                   "n_runs": len(sorted({r.run for r in results}))}, fh, indent=2,
                  ensure_ascii=False)


def main(n_attack: int = 40, n_plain: int = 40, n_hard: int = 40, n_runs: int = 1) -> None:
    llm = LLMClient()
    if not llm.available():
        raise SystemExit(f"LLM backend '{llm.backend}' is not reachable.")
    print(f"backend={llm.backend} model={llm.model}")
    print(f"{n_runs} run(s) x ({n_attack} ataki + {n_plain} zwykle + {n_hard} trudne negatywy)")

    t0 = time.perf_counter()
    results: list[CaseResult] = []
    for run, seed in enumerate(SEEDS[:n_runs]):
        print(f"\n=== run {run} (seed {seed}) ===", flush=True)
        ctx, cases, threshold = load_hard_negative_setup(
            n_attack=n_attack, n_plain_benign=n_plain, n_hard_benign=n_hard, seed=seed)
        print(f"threshold: {threshold:.5f}", flush=True)
        results += evaluate(ctx, cases, threshold, llm, configs=CONFIGS, run=run)
        _save(results)

    print("\n=== ALERT RATE BY CASE KIND ===")
    print(json.dumps(breakdown(results), indent=2, ensure_ascii=False))
    print("\n=== OVERALL ===")
    print(json.dumps(metrics_by_config(results), indent=2, ensure_ascii=False))
    print(f"\ntotal wall time {(time.perf_counter() - t0) / 60:.0f} min")


if __name__ == "__main__":
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    c = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    r = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    main(a, b, c, r)
