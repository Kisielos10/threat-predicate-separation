"""Is the detector inversion an artefact of how attacks and probes differ from normal traffic?

The obvious objection to the main result. Probes are value-only by construction: they change one
free-text parameter value and leave path, method and parameter names identical to a request the
corpus labels normal. CSIC attacks are not like that. Many of them are structural, using a path
that never occurs in normal traffic or renaming a parameter (`id` to `idA`, `B1` to `B1A`). If a
detector keyed on character content reacts more to a changed value than to a renamed parameter,
the inversion would follow from that asymmetry alone and would say nothing about whether anomalous
traffic is harmful.

The test partitions the attack group by how each attack departs from normal traffic, and reports
separation against the same probe group for each partition. Detectors, fitting set, calibration
split and thresholds are exactly those of `run_scaled_experiment.py`; only the attack subset
varies. If the inversion survives on the value-only attacks, which differ from normal traffic in
the same way probes do, the asymmetry is not what produces it.

    ./.venv/bin/python run_robustness_check.py [n_group] [n_fit]

Writes results/conditions/robustness_check.json.
"""

from __future__ import annotations

import json
import os
import sys
from urllib.parse import parse_qsl

import numpy as np

from zeroday_verify.baselines import PAYL, KruegelVignaICD
from zeroday_verify.data_text import load_csic, parse_request, serialize_http
from zeroday_verify.probes import generate

OUT = "results/conditions"


def structural_profile(requests: list[str]) -> dict[str, set[str]]:
    """Parameter names observed on each path in normal traffic."""
    prof: dict[str, set[str]] = {}
    for r in requests:
        p = parse_request(r)
        prof.setdefault(p["path"], set()).update(
            k for k, _ in parse_qsl(p["query"]) + parse_qsl(p["body"]))
    return prof


def departure(raw: str, prof: dict[str, set[str]]) -> str:
    """How this request departs from normal traffic, before any detector sees it."""
    p = parse_request(raw)
    known = prof.get(p["path"])
    if known is None:
        return "unknown_path"
    params = parse_qsl(p["query"]) + parse_qsl(p["body"])
    if [k for k, _ in params if k not in known]:
        return "unknown_parameter"
    return "value_only"


def main(n_group: int = 300, n_fit: int = 2000) -> None:
    os.makedirs(OUT, exist_ok=True)
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import roc_curve
    from sklearn.svm import OneClassSVM

    need = n_fit + 3 * n_group + 1500
    df = load_csic("data/raw_text/csic_all.parquet", n_normal=need,
                   n_anomalous=2 * n_group + 1500, seed=17)
    norm = [str(r) for r in df[df["label"] == 0]["requests"]]
    atk = [str(r) for r in df[df["label"] == 1]["requests"]]
    rng = np.random.RandomState(17)
    rng.shuffle(norm); rng.shuffle(atk)

    fit_n = norm[:n_fit]
    cal_n = norm[n_fit:n_fit + 500]
    base_n = norm[n_fit + 500 + n_group:]
    cal_a, ev_a = atk[:500], atk[500:500 + n_group]
    probes = [p.raw_request for p in generate(base_n, n_group)]

    prof = structural_profile(fit_n)
    parts: dict[str, list[str]] = {"value_only": [], "unknown_parameter": [], "unknown_path": []}
    for r in ev_a:
        parts[departure(r, prof)].append(r)
    # probes are value-only by construction; this asserts it rather than assuming it
    bad = [r for r in probes if departure(r, prof) != "value_only"]
    assert not bad, f"{len(bad)} probes are not value-only; the comparison would be unsound"

    print(f"attacks {len(ev_a)}: " + ", ".join(f"{k} {len(v)}" for k, v in parts.items()))
    print(f"probes {len(probes)}: value_only {len(probes)} (asserted)")

    txt = lambda r: serialize_http(parse_request(r))                          # noqa: E731
    RAW = {"fit": fit_n, "cal_n": cal_n, "cal_a": cal_a, "probe": probes,
           "attack_all": ev_a, **{f"attack_{k}": v for k, v in parts.items()}}
    T = {k: [txt(r) for r in v] for k, v in RAW.items()}

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=3000)
    Xfit = vec.fit_transform(T["fit"]).toarray()
    enc = lambda ts: vec.transform(ts).toarray()                              # noqa: E731
    kv = KruegelVignaICD().fit(RAW["fit"])
    payl = PAYL().fit(RAW["fit"])
    sub = Xfit[rng.choice(len(Xfit), min(1500, len(Xfit)), replace=False)]
    ocs = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(sub)
    iso = IsolationForest(n_estimators=200, random_state=17).fit(Xfit)
    pca = PCA(n_components=32, random_state=17).fit(Xfit)

    detectors = {
        "char-distribution (Kruegel & Vigna 2003)": ("raw", kv.score),
        "PAYL byte frequency (Wang & Stolfo 2004)": ("raw", payl.score),
        "one-class SVM (char n-gram)": ("txt", lambda ts: -ocs.decision_function(enc(ts))),
        "isolation forest (char n-gram)": ("txt", lambda ts: -iso.score_samples(enc(ts))),
        "PCA reconstruction error":
            ("txt", lambda ts: np.linalg.norm(
                enc(ts) - pca.inverse_transform(pca.transform(enc(ts))), axis=1)),
    }

    groups = ("attack_all", "attack_value_only", "attack_unknown_parameter", "attack_unknown_path")
    out = {"n": {k: len(v) for k, v in RAW.items() if k != "fit"}, "detectors": {}}
    print(f"\n{'detector':42s} {'probe':>7s} " +
          " ".join(f"{g.replace('attack_', ''):>18s}" for g in groups))
    for name, (kind, fn) in detectors.items():
        src = RAW if kind == "raw" else T
        s_cn, s_ca = fn(src["cal_n"]), fn(src["cal_a"])
        fpr, tpr, th = roc_curve([0] * len(s_cn) + [1] * len(s_ca),
                                 np.concatenate([s_cn, s_ca]))
        t = float(th[int(np.argmax(tpr - fpr))])
        rp = float((fn(src["probe"]) >= t).mean())
        row = {"probe_rate": round(rp, 3), "threshold": t}
        cells = []
        for g in groups:
            if not src[g]:
                row[g] = None; cells.append(f"{'n/a':>18s}"); continue
            r = float((fn(src[g]) >= t).mean())
            row[g] = {"rate": round(r, 3), "separation": round(r - rp, 3), "n": len(src[g])}
            cells.append(f"{r:8.2f} {r - rp:+9.2f}")
        out["detectors"][name] = row
        print(f"{name:42s} {rp:7.2f} " + " ".join(cells))

    with open(f"{OUT}/robustness_check.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT}/robustness_check.json")


if __name__ == "__main__":
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    main(a, b)
