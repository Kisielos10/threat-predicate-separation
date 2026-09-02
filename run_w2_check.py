"""Check 3a: can the threat graph supply condition W2 (potential CIA violation)?

This is the constructive half of the paper and it is untested. The idea: for a request under
assessment, retrieve similar threats already in the graph database and inherit their impact
vector sigma, giving an estimate of W2 without ever touching the label of the case being judged.

The obvious risk, and the reason this must be checked before writing: retrieval is driven by
phi-similarity, and phi-similarity is exactly what fails on legitimate-but-unusual requests,
which land near attacks in embedding space. If retrieval hands them attack-level sigma, then W2
fails in the same way W1 does and the constructive half of the paper collapses.

    ./.venv/bin/python run_w2_check.py [n_per_group]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from zeroday_verify.agents.cases import load_hard_negative_setup
from zeroday_verify.agents.tools import _case_as_threat, novelty_score

OUT = "results/conditions"
GROUPS = (("csic_attack", "attacks"), ("benign_hard", "legit-unusual"),
          ("benign_plain", "ordinary benign"))


def sigma_by_retrieval(ctx, case, k: int = 5, weighted: bool = True) -> tuple[float, dict]:
    """Estimate sigma for a case by retrieving its k nearest known threats.

    Returns (magnitude of the estimated sigma, diagnostic info). The reference corpus contains
    both known attacks (sigma > 0) and known normal traffic (sigma = 0), so a request whose
    neighbours are normal inherits a low estimate. No label of the case itself is used.
    """
    th = _case_as_threat(ctx, case)
    phi = th.phi / (np.linalg.norm(th.phi) + 1e-12)
    ref = ctx.known_phi / (np.linalg.norm(ctx.known_phi, axis=1, keepdims=True) + 1e-12)
    sims = ref @ phi
    idx = np.argsort(-sims)[:k]
    sig = np.vstack([ctx.known_threats[i].sigma for i in idx])
    w = np.clip(sims[idx], 0, None) if weighted else np.ones(len(idx))
    w = w / (w.sum() + 1e-12)
    est = (sig * w[:, None]).sum(axis=0)
    n_atk = sum(1 for i in idx if ctx.known_threats[i].is_attack)
    return float(np.linalg.norm(est)), {"neighbours_attack": int(n_atk), "k": k,
                                        "top_sim": round(float(sims[idx[0]]), 3)}


def main(n: int = 40) -> None:
    os.makedirs(OUT, exist_ok=True)
    ctx, cases, thr = load_hard_negative_setup(n_attack=n, n_plain_benign=n, n_hard_benign=n,
                                               seed=17)
    print(f"{len(cases)} cases; reference corpus {len(ctx.known_threats)} known threats "
          f"({sum(1 for t in ctx.known_threats if t.is_attack)} attacks)")

    rows = []
    for c in cases:
        mag, info = sigma_by_retrieval(ctx, c)
        rows.append({"case_id": c.case_id, "family": c.family,
                     "sigma_mag": mag, "u": novelty_score(ctx, c)["novelty_u"],
                     "neighbours_attack": info["neighbours_attack"]})

    print("\n=== estimated |sigma| by group (W2 by retrieval from the threat graph) ===")
    print(f"{'group':18s} {'n':>4s} {'median |sigma|':>15s} {'mean nbr-attacks/5':>20s}")
    stats = {}
    for fam, lab in GROUPS:
        g = [r for r in rows if r["family"] == fam]
        med = float(np.median([r["sigma_mag"] for r in g]))
        nbr = float(np.mean([r["neighbours_attack"] for r in g]))
        stats[fam] = {"median_sigma": round(med, 4), "mean_attack_neighbours": round(nbr, 2)}
        print(f"{lab:18s} {len(g):4d} {med:15.4f} {nbr:20.2f}")

    # Sweep a threshold on |sigma| and report separation attacks vs legitimate-but-unusual
    print("\n=== separation at various |sigma| thresholds ===")
    print(f"{'threshold':>10s} {'attacks':>9s} {'legit-unusual':>15s} {'ordinary':>10s} "
          f"{'separation':>11s}")
    best = None
    for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        r = {}
        for fam, _ in GROUPS:
            g = [x for x in rows if x["family"] == fam]
            r[fam] = sum(1 for x in g if x["sigma_mag"] >= t) / len(g)
        sep = r["csic_attack"] - r["benign_hard"]
        print(f"{t:10.2f} {r['csic_attack']:9.2f} {r['benign_hard']:15.2f} "
              f"{r['benign_plain']:10.2f} {sep:11.2f}")
        if best is None or sep > best[1]:
            best = (t, sep)
    print(f"\nbest separation {best[1]:.2f} at threshold {best[0]:.2f}")
    print("(for reference: distance detector separation on these groups is 0.00)")

    with open(f"{OUT}/w2_check.json", "w") as fh:
        json.dump({"stats": stats, "best": {"threshold": best[0], "separation": best[1]},
                   "rows": rows}, fh, indent=2)
    print(f"\nwrote {OUT}/w2_check.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
