"""Check 3: do PUBLISHED / standard anomaly detectors fail the same way ours does?

The paper claims something about a class of methods, not about one detector we wrote. That claim
is only credible if standard approaches from the literature reproduce the failure. All detectors
here are unsupervised or one-class, fitted on normal traffic only, and each is given its best
case: the decision threshold is calibrated on a labelled held-out split (Youden), which is a
supervised advantage the method would not have in deployment.

Detectors:
  * character-distribution model in the style of Kruegel & Vigna (2003), the canonical web
    anomaly-detection approach: per-request character frequencies sorted descending, scored by
    chi-square distance from the profile learned on normal traffic;
  * One-Class SVM on character n-gram TF-IDF;
  * Isolation Forest on character n-gram TF-IDF;
  * PCA reconstruction error, the standard linear stand-in for an autoencoder.

    ./.venv/bin/python run_baseline_check.py [n_per_group]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from zeroday_verify.agents.cases import load_hard_negative_setup
from zeroday_verify.data_text import parse_request, serialize_http

OUT = "results/conditions"
GROUPS = (("csic_attack", "attacks"), ("benign_hard", "legit-unusual"),
          ("benign_plain", "ordinary benign"))


def icd(text: str, n: int = 32) -> np.ndarray:
    """Idealized character distribution: sorted, normalised character frequencies (Kruegel/Vigna)."""
    if not text:
        return np.zeros(n)
    counts = np.zeros(256)
    for ch in text.encode("utf-8", "ignore"):
        counts[ch] += 1
    counts = np.sort(counts)[::-1][:n]
    total = counts.sum()
    return counts / total if total else counts


def main(n: int = 40) -> None:
    os.makedirs(OUT, exist_ok=True)
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import roc_curve
    from sklearn.svm import OneClassSVM

    ctx, cases, _ = load_hard_negative_setup(n_attack=n, n_plain_benign=n, n_hard_benign=n,
                                             seed=17)
    txt = lambda raw: serialize_http(parse_request(raw))                       # noqa: E731
    ref_norm = [t.text for t in ctx.known_threats if not t.is_attack]
    ref_atk = [t.text for t in ctx.known_threats if t.is_attack]
    # calibration split, disjoint from the evaluation cases (these are reference-corpus items)
    cal_n, cal_a = ref_norm[-150:], ref_atk[-150:]
    fit_n = ref_norm[:-150]
    print(f"fit on {len(fit_n)} normal; calibrate on {len(cal_n)}+{len(cal_a)}; "
          f"evaluate {len(cases)} cases")

    grp = {g: [txt(c.raw_request) for c in cases if c.family == g] for g, _ in GROUPS}

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=3000)
    Xfit = vec.fit_transform(fit_n).toarray()
    enc = lambda ts: vec.transform(ts).toarray()                               # noqa: E731

    Ficd = np.vstack([icd(t) for t in fit_n])
    icd_mu = Ficd.mean(axis=0)

    def chi2_icd(ts):
        F = np.vstack([icd(t) for t in ts])
        return (((F - icd_mu) ** 2) / (icd_mu + 1e-6)).sum(axis=1)

    ocs = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Xfit)
    iso = IsolationForest(n_estimators=200, random_state=17).fit(Xfit)
    pca = PCA(n_components=32, random_state=17).fit(Xfit)

    def pca_err(ts):
        X = enc(ts)
        return np.linalg.norm(X - pca.inverse_transform(pca.transform(X)), axis=1)

    detectors = {
        "char-distribution (Kruegel/Vigna)": chi2_icd,
        "one-class SVM (char n-gram)": lambda ts: -ocs.decision_function(enc(ts)),
        "isolation forest (char n-gram)": lambda ts: -iso.score_samples(enc(ts)),
        "PCA reconstruction error": pca_err,
    }

    out = {}
    print(f"\n{'detector':36s} {'attacks':>8s} {'legit-unusual':>14s} {'ordinary':>9s} "
          f"{'separation':>11s}  inversion")
    for name, fn in detectors.items():
        s_cn, s_ca = fn(cal_n), fn(cal_a)
        fpr, tpr, th = roc_curve([0] * len(s_cn) + [1] * len(s_ca),
                                 np.concatenate([s_cn, s_ca]))
        t = float(th[int(np.argmax(tpr - fpr))])
        rates, meds = {}, {}
        for g, _ in GROUPS:
            s = fn(grp[g])
            rates[g] = float((s >= t).mean())
            meds[g] = float(np.median(s))
        sep = rates["csic_attack"] - rates["benign_hard"]
        inv = "INVERTED" if meds["benign_hard"] > meds["csic_attack"] else "ordered"
        out[name] = {"rates": rates, "medians": meds, "separation": round(sep, 3),
                     "inverted": inv == "INVERTED", "threshold": t}
        print(f"{name:36s} {rates['csic_attack']:8.2f} {rates['benign_hard']:14.2f} "
              f"{rates['benign_plain']:9.2f} {sep:11.2f}  {inv}")

    with open(f"{OUT}/baseline_check.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT}/baseline_check.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
