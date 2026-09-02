"""Follow-up experiment: the multi-agent configuration without the thinned first pass.

The main evaluation runs the multi-agent system with a deliberately short first pass
(`first_pass_tools=2`) so that the evidence chain can fall short of the W4 minimum and the
send-back loop is actually exercised. That thinning is a confound: in the ~80% of cases where the
gate does not fire, the Auditor rules on less evidence than the single-agent baseline ever sees.

This script adds a `mas_full` configuration that gives the executor the same budget as the single
agent (4 tool calls), on the same cases and seeds. Comparing `mas` with `mas_full` separates the
effect of role separation from the effect of the thinning.

    ./.venv/bin/python run_mas_full_eval.py [n_attack] [n_benign] [n_runs]

Appends to results/agents/results.json and rewrites summary.json, so the main results are kept.
"""

from __future__ import annotations

import json
import sys
import time

from zeroday_verify.agents import LLMClient, load_evaluation_setup
from zeroday_verify.agents.evaluate import (
    evaluate,
    load_results,
    metrics_across_runs,
    save_results,
    sendback_stats,
)
from zeroday_verify.agents.trace import CaseResult, Decision, Step

SEEDS = (17, 23, 42, 101, 7)


def _revive(d: dict) -> CaseResult:
    """Rebuild a CaseResult from its serialised form so old and new results can be merged."""
    d = dict(d)
    d["decision"] = Decision(**d["decision"])
    d["steps"] = [Step(**s) for s in d.get("steps", [])]
    return CaseResult(**d)


def main(n_attack: int = 75, n_benign: int = 75, n_runs: int = 3) -> None:
    llm = LLMClient()
    if not llm.available():
        raise SystemExit(f"LLM backend '{llm.backend}' is not reachable.")

    try:
        previous = [_revive(d) for d in load_results()]
        previous = [r for r in previous if r.config != "mas_full"]   # idempotent re-runs
        print(f"loaded {len(previous)} existing results")
    except FileNotFoundError:
        previous = []
        print("no existing results found, starting fresh")

    seeds = SEEDS[:n_runs]
    print(f"backend={llm.backend} model={llm.model}")
    print(f"mas_full: {n_runs} runs x {n_attack + n_benign} cases, seeds {seeds}")

    t0 = time.perf_counter()
    new: list[CaseResult] = []
    for run, seed in enumerate(seeds):
        print(f"\n=== mas_full run {run} (seed {seed}) ===", flush=True)
        # identical setup to the main evaluation: same seed -> same cases
        ctx, cases, _ = load_evaluation_setup(
            n_eval_attack=n_attack, n_eval_benign=n_benign, seed=seed)
        new += evaluate(ctx, cases, 0.0, llm, configs=("mas_full",), run=run)
        save_results(previous + new)

    print("\n=== METRICS (mean +/- 95% CI across runs) ===")
    print(json.dumps(metrics_across_runs(previous + new), indent=2, ensure_ascii=False))
    print("\n=== SEND-BACK, mas_full ===")
    print(json.dumps(sendback_stats(previous + new, "mas_full"), indent=2, ensure_ascii=False))
    print(f"\ntotal wall time {(time.perf_counter() - t0) / 60:.0f} min")
    print("wrote results/agents/results.json and summary.json")


if __name__ == "__main__":
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 75
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 75
    r = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    main(a, b, r)
