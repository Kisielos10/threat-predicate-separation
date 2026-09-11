"""Compare the direct verdict with each condition, case by case, on identical cases and evidence.

Reads results/conditions/verdict_control.json and the condition runs it mirrors, and reports for
each model and setting: the direct verdict's separation, its separation on injection attacks, and
how often it agrees with each condition. Writes results/conditions/verdict_analysis.json.

    ./.venv/bin/python analyse_verdict_control.py
"""

from __future__ import annotations

import json

from run_subtype_check import subtype

from zeroday_verify.agents.cases import load_hard_negative_setup

VERDICT = "results/conditions/verdict_control.json"
CONDITIONS = {"with_score": "results/conditions/check_generated__qwen25_llama31_mistral.json",
              "no_score": "results/conditions/check_generated_no_novelty__qwen25_llama31_mistral.json"}
OUT = "results/conditions/verdict_analysis.json"


def rate(rows, key, fam, keep=lambda r: True):
    g = [r for r in rows if r["family"] == fam and keep(r)]
    return sum(key(r) for r in g) / len(g), len(g)


def main() -> None:
    ctx, cases, _ = load_hard_negative_setup(n_attack=150, n_plain_benign=150, n_hard_benign=150,
                                             seed=17, generated_probes=True)
    st = {c.case_id: subtype(c.raw_request, ctx.structure.params, None) for c in cases}
    verdicts = json.load(open(VERDICT))["by_model"]
    out = {}
    print(f"{'model':13s} {'setting':10s} {'sep':>6s} {'sep inj':>8s} "
          f"{'=W1':>5s} {'=W2':>5s} {'=W3':>5s}   W1/W2/W3 separation")
    for key, vrows in verdicts.items():
        model, setting = key.split("|")
        crow = {r["case_id"]: r for r in json.load(open(CONDITIONS[setting]))["by_model"][model]}
        ok = [r for r in vrows if not r["failed"]]
        v = {r["case_id"]: r["verdict"] for r in ok}
        pa, _ = rate(ok, lambda r: r["verdict"], "csic_attack")
        pp, _ = rate(ok, lambda r: r["verdict"], "benign_hard")
        ia, n_inj = rate(ok, lambda r: r["verdict"], "csic_attack",
                         lambda r: st[r["case_id"]] == "injection")
        agree = {c: sum(v[i] == crow[i]["conditions"][c] for i in v
                        if not crow[i].get("failed", {}).get(c)) / len(v)
                 for c in ("W1", "W2", "W3")}
        csep = {}
        for c in ("W1", "W2", "W3"):
            rows = list(crow.values())
            a, _ = rate(rows, lambda r, c=c: r["conditions"][c], "csic_attack")
            p, _ = rate(rows, lambda r, c=c: r["conditions"][c], "benign_hard")
            csep[c] = round(a - p, 3)
        out[key] = {"separation": round(pa - pp, 3), "attack_rate": round(pa, 3),
                    "probe_rate": round(pp, 3), "injection_separation": round(ia - pp, 3),
                    "n_injection": n_inj, "agreement": {c: round(x, 3) for c, x in agree.items()},
                    "condition_separation": csep}
        print(f"{model[:13]:13s} {setting:10s} {pa-pp:+6.2f} {ia-pp:+8.2f} "
              f"{agree['W1']:5.2f} {agree['W2']:5.2f} {agree['W3']:5.2f}   "
              f"{csep['W1']:+.2f} / {csep['W2']:+.2f} / {csep['W3']:+.2f}")
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
