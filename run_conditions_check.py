"""Smoke test: do the Definition 5 conditions still separate when judged independently?

Background. In the multi-agent experiment the verdict and all three conditions came out of one
LLM call, so the observed separation (W1 0.15, W3 0.26 between real attacks and legitimate but
unusual requests) may be an artefact of a single judgement leaking across fields. This script
re-measures the same quantity with each condition judged in its own call, blind to the verdict
and to the other conditions, and repeats the whole thing on a second model from a different
family to check that the result is not model-specific.

    ./.venv/bin/python run_conditions_check.py [n_per_group] [model ...]

Writes results/conditions/check.json.
"""

from __future__ import annotations

import json
import os
import sys
import time

from zeroday_verify.agents import LLMClient
from zeroday_verify.agents.cases import load_hard_negative_setup
from zeroday_verify.agents.tools import novelty_score
from zeroday_verify.conditions import CONDITIONS, evaluate_all, evidence_bundle

GROUPS = (("csic_attack", "real attacks"),
          ("benign_hard", "legitimate but unusual"),
          ("benign_plain", "ordinary legitimate"))
OUT = "results/conditions"


def separation(rows: list[dict], cond: str) -> dict:
    """Rate at which a condition is judged to hold, per group, plus the attack/hard gap."""
    out = {}
    for fam, _ in GROUPS:
        # exclude cases where the model gave no usable answer, rather than scoring them False
        g = [r for r in rows if r["family"] == fam and not r.get("failed", {}).get(cond)]
        out[fam] = round(sum(1 for r in g if r["conditions"][cond]) / len(g), 3) if g else None
        out[f"{fam}_n"] = len(g)
    if out.get("csic_attack") is not None and out.get("benign_hard") is not None:
        out["separation"] = round(out["csic_attack"] - out["benign_hard"], 3)
    return out


def main(n_per_group: int = 40, models: tuple[str, ...] = ("qwen2.5:7b", "llama3.1:8b"),
         generated_probes: bool = False, out_name: str = "check.json",
         with_novelty: bool = True) -> None:
    """`with_novelty=False` removes the detector's novelty reading from the evidence bundle.

    This is the ablation that answers the obvious objection to the W3 result: that the model is
    simply echoing the detector's score back as a judgement about origin. If W3 still separates
    without ever seeing the novelty value, it is drawing on the request itself rather than on the
    detector, and the objection does not hold.
    """
    os.makedirs(OUT, exist_ok=True)
    ctx, cases, threshold = load_hard_negative_setup(
        n_attack=n_per_group, n_plain_benign=n_per_group, n_hard_benign=n_per_group, seed=17,
        generated_probes=generated_probes)
    print(f"{len(cases)} cases, detector threshold {threshold:.5f}")

    # Evidence is deterministic and identical for every condition and every model, so it is
    # computed once. Only the question and the model vary.
    bundles = {}
    det = {}
    for c in cases:
        u = novelty_score(ctx, c)["novelty_u"]
        det[c.case_id] = u >= threshold
        bundles[c.case_id] = evidence_bundle(ctx.structure, c.raw_request,
                                             u if with_novelty else None)

    all_rows: dict[str, list[dict]] = {}
    for model in models:
        llm = LLMClient(model=model)
        if not llm.available():
            print(f"skipping {model}: backend unreachable")
            continue
        print(f"\n=== {model} ===", flush=True)
        rows, t0 = [], time.perf_counter()
        for i, case in enumerate(cases, 1):
            res = evaluate_all(llm, bundles[case.case_id])
            rows.append({"case_id": case.case_id, "family": case.family, "label": case.label,
                         "conditions": {k: v.holds for k, v in res.items()},
                         "failed": {k: v.failed for k, v in res.items()},
                         "reasons": {k: v.reason for k, v in res.items()}})
            if i % 20 == 0:
                print(f"  {i}/{len(cases)}  ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
        all_rows[model] = rows
        with open(f"{OUT}/{out_name}", "w") as fh:
            json.dump({"threshold": threshold, "detector": det, "by_model": all_rows}, fh,
                      indent=2, ensure_ascii=False)

    print("\n=== RATE AT WHICH EACH CONDITION HOLDS ===")
    for model, rows in all_rows.items():
        nf = sum(1 for r in rows for c in CONDITIONS if r.get("failed", {}).get(c))
        print(f"\n{model}" + (f"   ({nf} condition answers unusable, excluded)" if nf else ""))
        print(f"  {'condition':12s} {'attacks':>9s} {'legit-unusual':>15s} {'plain':>8s} "
              f"{'separation':>11s}")
        for cond in CONDITIONS:
            s = separation(rows, cond)
            print(f"  {cond:12s} {s['csic_attack']:9.2f} {s['benign_hard']:15.2f} "
                  f"{s['benign_plain']:8.2f} {s.get('separation', float('nan')):11.2f}")
        # the detector is the reference point: it implements W1 and nothing else
        # divide by the ACTUAL size of each group, never by the requested n: the hand-written
        # probe set is smaller than n_per_group, and dividing by n understated its rate badly
        ga = [r for r in rows if r["family"] == "csic_attack"]
        gh = [r for r in rows if r["family"] == "benign_hard"]
        da = sum(1 for r in ga if det[r["case_id"]]) / max(len(ga), 1)
        dh = sum(1 for r in gh if det[r["case_id"]]) / max(len(gh), 1)
        print(f"  {'detector':12s} {da:9.2f} {dh:15.2f} {'':>8s} {da - dh:11.2f}")
    print(f"\nwrote {OUT}/{out_name}")


if __name__ == "__main__":
    gen = "--generated" in sys.argv
    no_nov = "--no-novelty" in sys.argv
    argv = [a for a in sys.argv[1:] if a not in ("--generated", "--no-novelty")]
    n = int(argv[0]) if argv else 40
    ms = tuple(argv[1:]) or ("qwen2.5:7b", "llama3.1:8b")
    name = "check_generated" if gen else "check"
    if no_nov:
        name += "_no_novelty"
    # Include the models in the filename. Without this, a run with a different model silently
    # overwrote a completed run's results, because the output name depended only on the flags.
    tag = "_".join(m.split(":")[0].replace(".", "").replace("/", "") for m in ms)
    name += f"__{tag}"
    main(n, ms, generated_probes=gen, out_name=f"{name}.json", with_novelty=not no_nov)
