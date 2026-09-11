"""Collect every number the paper reports into one file, computed from the result artefacts.

Written so that no figure in the manuscript is transcribed by hand. The paper's numbers are
generated from here; if an experiment is re-run, this is the only place that needs re-running.
"""
from __future__ import annotations

import json
import pathlib

R = pathlib.Path("results")
OUT = pathlib.Path("report/final_numbers.json")
FAM = ("csic_attack", "benign_hard", "benign_plain")


def load(p):
    q = R / p
    return json.load(open(q)) if q.exists() else None


def cond_rates(rows, cond_set):
    """Rate at which all conditions in `cond_set` hold, per group, excluding unusable answers."""
    out = {}
    for fam in FAM:
        g = [x for x in rows if x["family"] == fam
             and not any(x.get("failed", {}).get(c) for c in cond_set)]
        out[fam] = round(sum(1 for x in g if all(x["conditions"][c] for c in cond_set)) / len(g), 3) \
            if g else None
        out[f"{fam}_n"] = len(g)
    if out["csic_attack"] is not None and out["benign_hard"] is not None:
        out["separation"] = round(out["csic_attack"] - out["benign_hard"], 3)
    return out


def detector_rates(rows, det):
    out = {}
    for fam in FAM:
        g = [x for x in rows if x["family"] == fam]
        out[fam] = round(sum(1 for x in g if det[x["case_id"]]) / len(g), 3) if g else None
    out["separation"] = round(out["csic_attack"] - out["benign_hard"], 3)
    return out


res = {}

# --- comparative evaluation (corrected evidence tool) --------------------------------------
s2 = load("agents_v3/summary.json")
if s2:
    res["comparative"] = {
        "n_runs": s2["n_runs"],
        "across_runs": {c: {k: s2["metrics_across_runs"][c][k]
                            for k in ("TPR", "FPR", "Precision", "F1")}
                        for c in s2["metrics_across_runs"]},
        "pooled": {c: {k: s2["metrics"][c][k] for k in
                       ("n", "TP", "FP", "TN", "FN", "unknown_rate",
                        "TPR_strict_atak", "mean_seconds")}
                   for c in s2["metrics"]},
        "sendback_mas_full": s2["sendback_by_config"].get("mas_full"),
        "calibration": s2["calibration"],
    }

# --- Definition 5 conditions, generated probes ----------------------------------------------
combos = {"W1": ("W1",), "W2": ("W2",), "W3": ("W3",),
          "W1_and_W2": ("W1", "W2"), "W1_and_W3": ("W1", "W3"),
          "W2_and_W3": ("W2", "W3"), "T_S": ("W1", "W2", "W3")}
conds = {}
# The runs at 150 per group made before the case-ID fix merged 50 probe/ordinary pairs, which
# then shared one evidence bundle and one detector flag. They are superseded by the re-runs below.
# The hosted model's run with the score was one of them and cannot be repeated, so it is dropped;
# its run without the score used 50 per group, where the old numbering never collided.
for tag, path in (("with_novelty_local", "conditions/check_generated__qwen25_llama31_mistral.json"),
                  ("no_novelty_local",
                   "conditions/check_generated_no_novelty__qwen25_llama31_mistral.json"),
                  ("no_novelty_sonnet",
                   "conditions/check_generated_no_novelty__claude-sonnet-5.json")):
    d = load(path)
    if not d:
        continue
    for model, rows in d["by_model"].items():
        keys = [(r["case_id"], r["family"]) for r in rows]
        ids = [r["case_id"] for r in rows]
        assert len(set(ids)) == len(ids), f"{path} [{model}]: case IDs collide; refusing to use it"
        assert len(set(keys)) == len(keys), f"{path} [{model}]: duplicate cases"
        conds.setdefault(tag, {})[model] = {
            "criteria": {name: cond_rates(rows, cs) for name, cs in combos.items()},
            "detector": detector_rates(rows, d["detector"]),
            "unusable": sum(1 for r in rows for c in ("W1", "W2", "W3")
                            if r.get("failed", {}).get(c)),
            "n_calls": len(rows) * 3,
            # how often two conditions receive the same answer on the same case: the evidence that
            # a model without statistics answers the deviation question as a question of origin
            "agreement": {f"{a}={b}": round(
                sum(r["conditions"][a] == r["conditions"][b] for r in ok) / len(ok), 3)
                for a, b in (("W1", "W3"), ("W1", "W2"), ("W2", "W3"))
                for ok in [[r for r in rows if not any(r.get("failed", {}).values())]]},
        }
res["conditions"] = conds

# --- hand-written probe set (for the probe-construction comparison) -------------------------
d = load("conditions/check.json")
if d:
    res["conditions_handwritten"] = {
        m: {"criteria": {name: cond_rates(rows, cs) for name, cs in combos.items()},
            "detector": detector_rates(rows, d["detector"])}
        for m, rows in d["by_model"].items()}

# --- published baselines, graph-based W2, flow domain ---------------------------------------
for key, path in (("published_baselines", "paper/main_experiment.json"),
                  ("robustness", "conditions/robustness_check.json"),
                  ("graph_w2", "conditions/w2_check.json"),
                  ("flow_domain", "conditions/flow_check.json")):
    d = load(path)
    if d:
        res[key] = d.get("results") or d.get("detectors") or d.get("stats") or d

OUT.parent.mkdir(exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
for k, v in res.items():
    print(f"  {k}: {len(v) if isinstance(v, dict) else '?'} entries")
