"""Everything the paper needs, run in one sequence.

Ollama serves one generation at a time, so these run strictly one after another rather than in
parallel; running anything else against the model while this is going will contend and appear as
a stall.

Stage 1  Definition-5 conditions, judged independently, on THREE model families (qwen, llama,
         mistral). Three families rather than two so that "this is not an artefact of one model"
         is a claim the results actually support.
Stage 2  The comparative evaluation re-run with the corrected evidence tool. The published
         numbers were produced before `payload_inspect` gained the structural profile, so the
         agent configurations were working with materially less evidence than the paper describes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

MODELS = ("qwen2.5:7b", "llama3.1:8b", "mistral:7b")


def stage1(n_per_group: int = 100) -> None:
    print("=" * 70, flush=True)
    print("STAGE 1: Definition 5 conditions, independent evaluation, three models", flush=True)
    print("=" * 70, flush=True)
    subprocess.run([sys.executable, "run_conditions_check.py", str(n_per_group), *MODELS],
                   check=False)


def stage2(n_attack: int = 75, n_benign: int = 75, n_runs: int = 3) -> None:
    print("\n" + "=" * 70, flush=True)
    print("STAGE 2: comparative evaluation, corrected evidence tool", flush=True)
    print("=" * 70, flush=True)
    from zeroday_verify.agents import LLMClient, load_evaluation_setup
    from zeroday_verify.agents.evaluate import (
        evaluate,
        metrics_across_runs,
        save_results,
        sendback_stats,
    )
    llm = LLMClient()
    if not llm.available():
        print("LLM backend unreachable, skipping stage 2")
        return
    configs = ("detector", "single_agent", "mas_full")
    seeds = (17, 23, 42)[:n_runs]
    results = []
    t0 = time.perf_counter()
    for run, seed in enumerate(seeds):
        print(f"\n=== run {run} (seed {seed}) ===", flush=True)
        ctx, cases, threshold = load_evaluation_setup(
            n_eval_attack=n_attack, n_eval_benign=n_benign, seed=seed)
        print(f"threshold {threshold:.5f}; structural profile "
              f"{'present' if ctx.structure is not None else 'MISSING'}", flush=True)
        results += evaluate(ctx, cases, threshold, llm, configs=configs, run=run)
        save_results(results, out_dir="results/agents_v3")
    print("\n=== METRICS (mean +/- 95% CI) ===")
    print(json.dumps(metrics_across_runs(results), indent=2, ensure_ascii=False))
    print("\n=== SEND-BACK ===")
    print(json.dumps(sendback_stats(results, "mas_full"), indent=2, ensure_ascii=False))
    print(f"\nstage 2 wall time {(time.perf_counter() - t0) / 60:.0f} min")


if __name__ == "__main__":
    stage1(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
    stage2()
    print("\nALL EXPERIMENTS COMPLETE")
