"""Does CSIC's anomalous class, which includes non-malicious requests, distort the separations?

CSIC-2010 describes three kinds of anomalous request: static attacks (requests for hidden or
non-existent resources), dynamic attacks (valid arguments modified with SQL injection, CRLF
injection, cross-site scripting, buffer overflows and the like), and unintentionally illegal
requests, which "do not have malicious intention" but do not follow the application's normal
behaviour. The binary label used throughout this work calls all three anomalous. If a large share
of the evaluation "attacks" are unintentionally illegal, every attack rate is diluted by requests
that a correct semantic assessor should call legitimate.

The distributed corpus carries no subtype label, so subtypes are inferred here from how each
request departs from normal traffic:

  static        the path never occurs in normal traffic
  tampering     a parameter name never occurs on that path (the corpus lists parameter
                tampering among its attacks)
  injection     every name is known, and a value unseen in normal traffic carries an explicit
                attack signature
  no_signature  every name is known, and the unseen values carry no attack signature: the
                candidates for unintentionally illegal requests

`injection` is the strictest subset: each member carries positive evidence of malicious content.
If separation survives there, the dilution does not explain the result.

    ./.venv/bin/python run_subtype_check.py

Writes results/conditions/subtype_check.json. Makes no model calls; condition answers are read
from the per-case records the condition runs already saved.
"""

from __future__ import annotations

import json
import re
from urllib.parse import parse_qsl, unquote_plus

import numpy as np

from zeroday_verify.data_text import load_csic, parse_request, serialize_http

OUT = "results/conditions/subtype_check.json"

SIGNATURE = re.compile(
    r"""(
        ['"`]                          # quote characters that open or close a context
      | --|/\*|\*/|;                   # SQL comment and statement separators
      | \b(or|and)\b\s*[\w'"(]*\s*=    # boolean tautologies
      | \b(union|select|insert|delete|drop|update|exec|sleep|benchmark|waitfor)\b
      | <\s*/?\s*[a-z!]                # markup, including <script and SSI <!--#
      | javascript:|on\w+\s*=|alert\s*\(
      | \.\./|\.\.\\                   # path traversal
      | [\x00-\x08\x0b\x0c\x0e-\x1f]   # control characters, the null byte among them
      | [\r\n]                         # CRLF injection
      | set-cookie|content-type:       # header injection
      | \$\(|\|\||&&                   # shell command substitution and chaining
      | %[sndx]\b                      # format strings
    )""", re.I | re.X)
LONG_VALUE = 200                        # buffer-overflow attempts are long by construction


def profile(normal: list[str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    names: dict[str, set[str]] = {}
    values: dict[str, set[str]] = {}
    for r in normal:
        p = parse_request(r)
        for k, v in parse_qsl(p["query"]) + parse_qsl(p["body"]):
            names.setdefault(p["path"], set()).add(k)
            values.setdefault(k, set()).add(unquote_plus(v))
    return names, values


def subtype(raw: str, names: dict[str, set[str]], values: dict[str, set[str]] | None) -> str:
    """`values=None` checks every value for a signature rather than only unseen ones."""
    p = parse_request(raw)
    known = names.get(p["path"])
    if known is None:
        return "static"
    params = parse_qsl(p["query"]) + parse_qsl(p["body"])
    if any(k not in known for k, _ in params):
        return "tampering"
    novel = [unquote_plus(v) for k, v in params
             if values is None or unquote_plus(v) not in values.get(k, set())]
    if any(SIGNATURE.search(v) or len(v) > LONG_VALUE for v in novel):
        return "injection"
    return "no_signature"


def condition_rates(path: str, cases_by_id: dict, names, values) -> dict:
    """Per-subtype condition rates from the saved per-case records of a condition run."""
    d = json.load(open(path))
    out = {}
    for model, rows in d["by_model"].items():
        probes = [r for r in rows if r["family"] == "benign_hard"]
        m = {}
        for cond in ("W1", "W2", "W3"):
            pr = [r for r in probes if not r.get("failed", {}).get(cond)]
            p_rate = sum(r["conditions"][cond] for r in pr) / len(pr)
            m[cond] = {"probe_rate": round(p_rate, 3)}
            for st in ("all", "static", "tampering", "injection", "no_signature"):
                g = [r for r in rows if r["family"] == "csic_attack"
                     and not r.get("failed", {}).get(cond)
                     and (st == "all" or subtype(cases_by_id[r["case_id"]], names, values) == st)]
                if g:
                    rate = sum(r["conditions"][cond] for r in g) / len(g)
                    m[cond][st] = {"rate": round(rate, 3), "separation": round(rate - p_rate, 3),
                                   "n": len(g)}
        out[model] = m
    return out


def main() -> None:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import roc_curve
    from sklearn.svm import OneClassSVM

    from zeroday_verify.agents.cases import load_hard_negative_setup
    from zeroday_verify.baselines import PAYL, KruegelVignaICD
    from zeroday_verify.probes import generate

    result: dict = {}

    # --- part 1: the detector experiment, on its own sample (run_scaled_experiment.py) ---
    n_fit, n_group = 2000, 300
    df = load_csic("data/raw_text/csic_all.parquet", n_normal=n_fit + 3 * n_group + 1500,
                   n_anomalous=2 * n_group + 1500, seed=17)
    norm = [str(r) for r in df[df["label"] == 0]["requests"]]
    atk = [str(r) for r in df[df["label"] == 1]["requests"]]
    rng = np.random.RandomState(17)
    rng.shuffle(norm); rng.shuffle(atk)
    fit_n, cal_n = norm[:n_fit], norm[n_fit:n_fit + 500]
    base_n = norm[n_fit + 500 + n_group:]
    cal_a, ev_a = atk[:500], atk[500:500 + n_group]
    probes = [p.raw_request for p in generate(base_n, n_group)]
    names, values = profile(fit_n)
    groups = {st: [r for r in ev_a if subtype(r, names, values) == st]
              for st in ("static", "tampering", "injection", "no_signature")}
    comp = {k: len(v) for k, v in groups.items()}
    print("detector-experiment attacks by subtype:", comp)
    result["detector_experiment"] = {"composition": comp, "detectors": {}}

    txt = lambda r: serialize_http(parse_request(r))                          # noqa: E731
    RAW = {"fit": fit_n, "cal_n": cal_n, "cal_a": cal_a, "probe": probes, "all": ev_a, **groups}
    T = {k: [txt(r) for r in v] for k, v in RAW.items()}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=3000)
    Xfit = vec.fit_transform(T["fit"]).toarray()
    enc = lambda ts: vec.transform(ts).toarray()                              # noqa: E731
    kv, payl = KruegelVignaICD().fit(RAW["fit"]), PAYL().fit(RAW["fit"])
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
    print(f"\n{'detector':42s} {'probe':>6s} " +
          " ".join(f"{k:>14s}" for k in ("all", "injection", "no_signature")))
    for name, (kind, fn) in detectors.items():
        src = RAW if kind == "raw" else T
        s_cn, s_ca = fn(src["cal_n"]), fn(src["cal_a"])
        fpr, tpr, th = roc_curve([0] * len(s_cn) + [1] * len(s_ca), np.concatenate([s_cn, s_ca]))
        t = float(th[int(np.argmax(tpr - fpr))])
        rp = float((fn(src["probe"]) >= t).mean())
        row = {"probe_rate": round(rp, 3)}
        for st in ("all", "static", "tampering", "injection", "no_signature"):
            if src[st]:
                r = float((fn(src[st]) >= t).mean())
                row[st] = {"rate": round(r, 3), "separation": round(r - rp, 3), "n": len(src[st])}
        result["detector_experiment"]["detectors"][name] = row
        print(f"{name:42s} {rp:6.2f} " + " ".join(
            f"{row[s]['rate']:5.2f} {row[s]['separation']:+7.2f}" for s in
            ("all", "injection", "no_signature")))

    # --- part 2: the condition experiment, from its saved per-case answers ---
    ctx, cases, _ = load_hard_negative_setup(n_attack=150, n_plain_benign=150, n_hard_benign=150,
                                             seed=17, generated_probes=True)
    # The structural profile the evidence bundles were built from, used verbatim so the subtype
    # of each case matches what the model was told about its structure.
    cnames, cvalues = ctx.structure.params, None
    by_id = {c.case_id: c.raw_request for c in cases}
    catk = [c for c in cases if c.family == "csic_attack"]
    ccomp = {st: sum(1 for c in catk if subtype(c.raw_request, cnames, cvalues) == st)
             for st in ("static", "tampering", "injection", "no_signature")}
    print("\ncondition-experiment attacks by subtype:", ccomp)
    result["condition_experiment"] = {"composition": ccomp, "by_setting": {}}
    for label, path in (("with_score", "results/conditions/check_generated__qwen25_llama31_mistral.json"),
                        ("no_score", "results/conditions/check_generated_no_novelty__qwen25_llama31_mistral.json")):
        result["condition_experiment"]["by_setting"][label] = condition_rates(
            path, by_id, cnames, cvalues)

    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
