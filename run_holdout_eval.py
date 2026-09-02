"""Leave-one-attack-type-out evaluation: the zero-day setting proper.

The in-distribution experiment (`run_agents_eval.py`) asks whether a request can be separated
from normal traffic when every attack type is already represented in the reference corpus and in
the detector's calibration split. That favours the threshold detector, which is a supervised
method tuned on the very distribution it is tested on.

Here the held-out attack type appears in NONE of: the retrieval corpus, the known-normal profile,
or the calibration split. The detector's threshold is therefore set without any knowledge of the
attacks it must catch, which is the situation the thesis actually concerns.

    ./.venv/bin/python run_holdout_eval.py [n_attack] [n_benign] [types...]

Writes results/agents/holdout_{results,summary}.json.
"""

from __future__ import annotations

import json
import sys
import time

from zeroday_verify.agents import LLMClient
from zeroday_verify.agents.cases import load_holdout_setup
from zeroday_verify.agents.evaluate import evaluate, metrics_by_config
from zeroday_verify.agents.trace import CaseResult

CONFIGS = ("detector", "single_agent", "mas_full")
DEFAULT_TYPES = ("sqli", "xss")
OUT = "results/agents"


def _save(all_results: dict[str, list[CaseResult]]) -> None:
    import os
    os.makedirs(OUT, exist_ok=True)
    flat = [dict(r.to_dict(), holdout=h) for h, rs in all_results.items() for r in rs]
    with open(f"{OUT}/holdout_results.json", "w") as fh:
        json.dump(flat, fh, indent=2, ensure_ascii=False)
    summary = {h: metrics_by_config(rs) for h, rs in all_results.items()}
    with open(f"{OUT}/holdout_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def main(n_attack: int = 60, n_benign: int = 60, types: tuple[str, ...] = DEFAULT_TYPES) -> None:
    llm = LLMClient()
    if not llm.available():
        raise SystemExit(f"LLM backend '{llm.backend}' is not reachable.")
    print(f"backend={llm.backend} model={llm.model}")
    print(f"leave-one-type-out over {list(types)}, {n_attack}+{n_benign} cases each")

    t0 = time.perf_counter()
    all_results: dict[str, list[CaseResult]] = {}
    for holdout in types:
        print(f"\n=== holdout: {holdout} ===", flush=True)
        ctx, cases, threshold = load_holdout_setup(
            holdout, n_eval_attack=n_attack, n_eval_benign=n_benign)
        print(f"threshold calibrated WITHOUT '{holdout}': {threshold:.5f}", flush=True)
        all_results[holdout] = evaluate(ctx, cases, threshold, llm, configs=CONFIGS)
        _save(all_results)
        m = metrics_by_config(all_results[holdout])
        for cfg in CONFIGS:
            d = m[cfg]
            print(f"  {cfg:13s} TPR={d['TPR']:.3f} FPR={d['FPR']:.3f} F1={d['F1']:.3f}",
                  flush=True)

    print(f"\ntotal wall time {(time.perf_counter() - t0) / 60:.0f} min")
    print(f"wrote {OUT}/holdout_results.json and holdout_summary.json")


if __name__ == "__main__":
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    t = tuple(sys.argv[3:]) or DEFAULT_TYPES
    main(a, b, t)
