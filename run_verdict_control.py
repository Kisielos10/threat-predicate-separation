"""Direct-verdict control: does decomposing into conditions add anything over asking outright?

The condition assessments never ask a model whether a request is an attack. That is right for
the blind-assessment design, but it leaves open whether the decomposition adds anything over a
single yes-or-no question put to the same model with the same evidence. This script asks that
question directly, on exactly the cases and evidence bundles of `run_conditions_check.py
--generated`, so its verdicts can be compared with each condition case by case.

The comparison that matters is with W3. If a model without statistics answers the deviation
question from semantic plausibility, as the agreement between its W1 and W3 answers indicates,
then its direct verdict should track W3 closely, and the decomposition's value is analytical
rather than operational.

    ./.venv/bin/python run_verdict_control.py [n_per_group] [model ...]

Writes results/conditions/verdict_control.json, saving after each model and setting.
"""

from __future__ import annotations

import json
import os
import sys
import time

from zeroday_verify.agents import LLMClient
from zeroday_verify.agents.cases import load_hard_negative_setup
from zeroday_verify.agents.tools import novelty_score
from zeroday_verify.conditions import evidence_bundle

OUT = "results/conditions/verdict_control.json"

# Parallel to the condition prompts in every respect except the question itself: same evidence,
# same reply format, one call per request, no condition answers visible.
_SYS = ("You are assessing an HTTP request against evidence collected by deterministic tools. "
        "Reply with JSON only.")
_QUESTION = "Is this request an attack on the application?"


def ask(llm, evidence: str) -> tuple[bool, bool]:
    """Return (verdict, failed). Unusable answers are flagged, not scored as benign."""
    try:
        out = llm.chat_json(_SYS, (
            f"Evidence:\n{evidence}\n\nQuestion: {_QUESTION}\n\n"
            'Reply exactly: {"holds": true or false, "reason": "one short sentence"}'))
    except Exception:                                                     # noqa: BLE001
        return False, True
    return bool(out.get("holds")), False


def main(n_per_group: int = 150,
         models: tuple[str, ...] = ("qwen2.5:7b", "llama3.1:8b", "mistral:7b")) -> None:
    ctx, cases, threshold = load_hard_negative_setup(
        n_attack=n_per_group, n_plain_benign=n_per_group, n_hard_benign=n_per_group,
        seed=17, generated_probes=True)
    print(f"{len(cases)} cases, detector threshold {threshold:.5f}", flush=True)

    bundles = {True: {}, False: {}}
    for c in cases:
        u = novelty_score(ctx, c)["novelty_u"]
        bundles[True][c.case_id] = evidence_bundle(ctx.structure, c.raw_request, u)
        bundles[False][c.case_id] = evidence_bundle(ctx.structure, c.raw_request, None)

    result = json.load(open(OUT)) if os.path.exists(OUT) else {"by_model": {}}
    for model in models:
        llm = LLMClient(model=model)
        if not llm.available():
            print(f"skipping {model}: backend unreachable", flush=True)
            continue
        for with_score in (False, True):
            key = f"{model}|{'with_score' if with_score else 'no_score'}"
            if key in result["by_model"]:
                print(f"{key}: already done, skipping", flush=True)
                continue
            rows, t0 = [], time.perf_counter()
            for i, c in enumerate(cases, 1):
                verdict, failed = ask(llm, bundles[with_score][c.case_id])
                rows.append({"case_id": c.case_id, "family": c.family,
                             "verdict": verdict, "failed": failed})
                if i % 50 == 0:
                    print(f"  {key} {i}/{len(cases)} ({(time.perf_counter()-t0)/60:.1f} min)",
                          flush=True)
            result["by_model"][key] = rows
            with open(OUT, "w") as fh:
                json.dump(result, fh, indent=1)
            fam = {f: [r for r in rows if r["family"] == f and not r["failed"]]
                   for f in ("csic_attack", "benign_hard", "benign_plain")}
            rate = {f: sum(r["verdict"] for r in g) / len(g) for f, g in fam.items() if g}
            print(f"{key}: attacks {rate['csic_attack']:.2f}  probes {rate['benign_hard']:.2f}  "
                  f"ordinary {rate['benign_plain']:.2f}  "
                  f"separation {rate['csic_attack'] - rate['benign_hard']:+.2f}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    n = int(argv[0]) if argv else 150
    main(n, tuple(argv[1:]) or ("qwen2.5:7b", "llama3.1:8b", "mistral:7b"))
