"""Run the comparative multi-agent evaluation over several independent repetitions.

    ./.venv/bin/python run_agents_eval.py [n_attack] [n_benign] [n_runs]

Each run draws a fresh case sample with its own seed, and every case goes through all three
configurations (detector, single agent, multi-agent). Results are aggregated to mean +/- 95% CI,
matching the reporting style of the earlier verification studies.
Writes results/agents/{results,summary}.json.
"""

from __future__ import annotations

import json
import sys
import time

from zeroday_verify.agents import LLMClient, load_evaluation_setup
from zeroday_verify.agents.evaluate import (
    evaluate,
    metrics_across_runs,
    save_results,
    sendback_stats,
)

SEEDS = (17, 23, 42, 101, 7)


def main(n_attack: int = 75, n_benign: int = 75, n_runs: int = 3) -> None:
    llm = LLMClient()
    if not llm.available():
        raise SystemExit(
            f"LLM backend '{llm.backend}' is not reachable. Start it with `ollama serve` "
            f"and `ollama pull {llm.model}`.")
    seeds = SEEDS[:n_runs]
    total = (n_attack + n_benign) * n_runs
    print(f"backend={llm.backend} model={llm.model}")
    print(f"{n_runs} runs x {n_attack + n_benign} cases = {total} cases, seeds {seeds}")

    t0 = time.perf_counter()
    all_results = []
    for run, seed in enumerate(seeds):
        print(f"\n=== run {run} (seed {seed}) ===", flush=True)
        ctx, cases, threshold = load_evaluation_setup(
            n_eval_attack=n_attack, n_eval_benign=n_benign, seed=seed)
        print(f"detector threshold (calibrated on held-out split): {threshold:.5f}", flush=True)
        all_results += evaluate(ctx, cases, threshold, llm, run=run)
        save_results(all_results)      # checkpoint after each run

    print("\n=== METRICS (mean +/- 95% CI across runs) ===")
    print(json.dumps(metrics_across_runs(all_results), indent=2, ensure_ascii=False))
    print("\n=== SEND-BACK (W4 evidentiary gate) ===")
    print(json.dumps(sendback_stats(all_results), indent=2, ensure_ascii=False))
    print(f"\ntotal wall time {(time.perf_counter() - t0) / 60:.0f} min, "
          f"{llm.stats.n_calls} LLM calls")
    print("wrote results/agents/results.json and summary.json")


if __name__ == "__main__":
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 75
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 75
    r = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    main(a, b, r)
