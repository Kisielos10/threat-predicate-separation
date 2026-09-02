"""Paired version of the main experiment, removing the confounds found in validation.

The unpaired comparison was confounded twice. Probes can only be generated from requests that
carry a free-text parameter, so they are drawn from the parameter-rich part of the corpus, while
"ordinary" traffic includes many requests with no parameters at all (median payload length 0
against 137). And percent-encoding differs: ordinary CSIC traffic contains essentially no '%',
while any substituted value introduces some.

The paired design removes both. For every probe we keep the exact request it was derived from as
its own control. The pair is identical in method, path, parameter names, parameter order and all
other values; it differs only in the content of one free-text field. Any difference in the
detector's verdict is therefore attributable to that content and to nothing else.

    ./.venv/bin/python run_paired_experiment.py [n]
"""
from __future__ import annotations

import json
import os
import sys
from urllib.parse import parse_qsl, quote_plus

import numpy as np

from zeroday_verify.baselines import PAYL, KruegelVignaICD
from zeroday_verify.data_text import load_csic, parse_request
from zeroday_verify.probes import _rebuild, generate

OUT = "results/paper"


def rebuilt_identity(raw: str) -> str:
    """The base request put through the same rebuild path, values unchanged."""
    p = parse_request(raw)
    params = parse_qsl(p["query"]) + parse_qsl(p["body"])
    return _rebuild(p, [(k, quote_plus(v)) for k, v in params])


def main(n: int = 300) -> None:
    os.makedirs(OUT, exist_ok=True)
    from sklearn.metrics import roc_curve

    df = load_csic("data/raw_text/csic_all.parquet", n_normal=3200, n_anomalous=1200, seed=17)
    norm = [str(r) for r in df[df["label"] == 0]["requests"]]
    atk = [str(r) for r in df[df["label"] == 1]["requests"]]
    rng = np.random.RandomState(17); rng.shuffle(norm); rng.shuffle(atk)
    fit, cal_n, base = norm[:2000], norm[2000:2500], norm[2500:]
    cal_a, ev_a = atk[:500], atk[500:500 + n]

    # generate probes and recover, for each, the exact request it came from
    probes = generate(base, n)
    pairs = []
    for p in probes:
        pr = parse_request(p.raw_request)
        keys = [k for k, _ in parse_qsl(pr["query"]) + parse_qsl(pr["body"])]
        for b in base:
            pb = parse_request(b)
            if pb["path"] != pr["path"]:
                continue
            kb = [k for k, _ in parse_qsl(pb["query"]) + parse_qsl(pb["body"])]
            if kb != keys:
                continue
            vb = dict(parse_qsl(pb["query"]) + parse_qsl(pb["body"]))
            vr = dict(parse_qsl(pr["query"]) + parse_qsl(pr["body"]))
            diff = [k for k in keys if vb.get(k) != vr.get(k)]
            if diff == [p.parameter]:
                pairs.append((rebuilt_identity(b), p.raw_request, p.category))
                break
    print(f"matched {len(pairs)}/{len(probes)} probes to their exact base request")
    if len(pairs) < 50:
        print("too few pairs to conclude"); return

    ctrl = [c for c, _, _ in pairs]
    prob = [p for _, p, _ in pairs]
    cats = [c for _, _, c in pairs]

    out = {}
    print(f"\n{'detector':40s} {'control':>8s} {'probe':>7s} {'flipped':>8s} "
          f"{'attacks':>8s} {'sep(atk-probe)':>15s}")
    for name, m in (("char-distribution (Kruegel & Vigna 2003)", KruegelVignaICD().fit(fit)),
                    ("PAYL byte frequency (Wang & Stolfo 2004)", PAYL().fit(fit))):
        s_cn, s_ca = m.score(cal_n), m.score(cal_a)
        fpr, tpr, th = roc_curve([0] * len(s_cn) + [1] * len(s_ca),
                                 np.concatenate([s_cn, s_ca]))
        t = float(th[int(np.argmax(tpr - fpr))])
        a_c, a_p, a_a = m.score(ctrl) >= t, m.score(prob) >= t, m.score(ev_a) >= t
        flipped = float((~a_c & a_p).mean())          # control passed, probe flagged
        out[name] = {"control": float(a_c.mean()), "probe": float(a_p.mean()),
                     "flipped_by_content": flipped, "attacks": float(a_a.mean()),
                     "separation": float(a_a.mean() - a_p.mean()),
                     "per_category": {c: float(a_p[[i for i, x in enumerate(cats) if x == c]].mean())
                                      for c in sorted(set(cats))}}
        o = out[name]
        print(f"{name:40s} {o['control']:8.2f} {o['probe']:7.2f} {flipped:8.2f} "
              f"{o['attacks']:8.2f} {o['separation']:15.2f}")

    print("\nflipped = the control request passed, and substituting one legitimate value")
    print("          into it was enough to make the detector raise an alert")
    with open(f"{OUT}/paired_experiment.json", "w") as fh:
        json.dump({"n_pairs": len(pairs), "detectors": out}, fh, indent=2)
    print(f"\nwrote {OUT}/paired_experiment.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
