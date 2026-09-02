"""Validation of the probe generator, before any result that depends on it is believed.

Four controls:

  1. IDENTITY. Pass real normal requests through the same parse-and-rebuild pipeline the probes go
     through, substituting each value with ITSELF. If these are flagged at anything like the rate
     of the probes, the detectors are reacting to the reconstruction rather than to the content,
     and every number in the main table is an artefact.
  2. LENGTH. Compare payload-length distributions across groups. If probes are simply much longer
     than attacks, "more anomalous" is trivial and uninteresting.
  3. ENCODING. Check that the percent-encoding convention in generated requests matches the
     corpus, since a mismatch would be a give-away unrelated to content.
  4. EYEBALL. Print real normal, identity-rebuilt, and probe requests side by side.
"""
from __future__ import annotations

from urllib.parse import parse_qsl

import numpy as np

from zeroday_verify.baselines import PAYL, KruegelVignaICD
from zeroday_verify.data_text import load_csic, parse_request
from zeroday_verify.probes import _rebuild, generate


def identity_rebuild(raw: str) -> str:
    """Same pipeline as a probe, but every value is replaced by itself."""
    p = parse_request(raw)
    params = parse_qsl(p["query"]) + parse_qsl(p["body"])
    from urllib.parse import quote_plus
    return _rebuild(p, [(k, quote_plus(v)) for k, v in params])


def payload_len(raw: str) -> int:
    p = parse_request(raw)
    return len(p["query"]) + len(p["body"])


def main(n: int = 300) -> None:
    df = load_csic("data/raw_text/csic_all.parquet", n_normal=3200, n_anomalous=1200, seed=17)
    norm = [str(r) for r in df[df["label"] == 0]["requests"]]
    atk = [str(r) for r in df[df["label"] == 1]["requests"]]
    rng = np.random.RandomState(17); rng.shuffle(norm); rng.shuffle(atk)
    fit, cal_n, ordinary, base = norm[:2000], norm[2000:2500], norm[2500:2500+n], norm[2500+n:]
    cal_a, ev_a = atk[:500], atk[500:500+n]

    probes = [p.raw_request for p in generate(base, n)]
    identity = [identity_rebuild(r) for r in ordinary]

    print("=== CONTROL 1: identity rebuild (values unchanged, same pipeline) ===")
    from sklearn.metrics import roc_curve
    for name, m in (("Kruegel & Vigna", KruegelVignaICD().fit(fit)), ("PAYL", PAYL().fit(fit))):
        s_cn, s_ca = m.score(cal_n), m.score(cal_a)
        fpr, tpr, th = roc_curve([0]*len(s_cn)+[1]*len(s_ca), np.concatenate([s_cn, s_ca]))
        t = float(th[int(np.argmax(tpr-fpr))])
        r = {g: float((m.score(v) >= t).mean())
             for g, v in (("ordinary(raw)", ordinary), ("ordinary(rebuilt)", identity),
                          ("probes", probes), ("attacks", ev_a))}
        print(f"  {name:16s} " + "  ".join(f"{k} {v:.2f}" for k, v in r.items()))
        art = r["ordinary(rebuilt)"] - r["ordinary(raw)"]
        print(f"{'':18s} rebuild artefact = {art:+.2f}"
              + ("   <-- PIPELINE ARTEFACT, results unsafe" if art > 0.10 else "   (acceptable)"))

    print("\n=== CONTROL 2: payload length ===")
    for lab, v in (("ordinary", ordinary), ("rebuilt", identity), ("probes", probes),
                   ("attacks", ev_a)):
        ls = [payload_len(x) for x in v]
        print(f"  {lab:10s} median {np.median(ls):6.0f}  mean {np.mean(ls):6.0f}  "
              f"p90 {np.percentile(ls,90):6.0f}")

    print("\n=== CONTROL 3: encoding convention ===")
    def enc_stats(v):
        j = "".join(parse_request(x)["query"] + parse_request(x)["body"] for x in v)
        return {"%": j.count("%") / max(len(v), 1), "+": j.count("+") / max(len(v), 1)}
    for lab, v in (("ordinary", ordinary), ("probes", probes), ("attacks", ev_a)):
        e = enc_stats(v)
        print(f"  {lab:10s} '%' per request {e['%']:.2f}   '+' per request {e['+']:.2f}")

    print("\n=== CONTROL 4: side by side ===")
    for lab, v in (("REAL NORMAL ", ordinary), ("REBUILT     ", identity), ("PROBE       ", probes)):
        for x in v[:2]:
            p = parse_request(x)
            print(f"  {lab} {p['method']:5s} {p['path'][:38]:38s} {(p['query'] or p['body'])[:70]}")


if __name__ == "__main__":
    main()
