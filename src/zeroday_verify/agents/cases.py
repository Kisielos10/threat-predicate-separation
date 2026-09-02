"""Building the evaluation setup: reference knowledge vs. evaluation cases.

Critical discipline: the corpus the agents may retrieve from (`known_threats`, and the
known-normal profile behind the novelty tool) is DISJOINT from the cases they are asked to
judge. Otherwise `similar_known_threats` would hand the agent the answer, exactly the kind of
leakage we controlled for in the detector study.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..data_text import build_text_threats, load_csic
from .tools import Case, ToolContext, build_tool_context, novelty_score


def load_evaluation_setup(
    n_ref_normal: int = 1500,
    n_ref_attack: int = 400,
    n_eval_attack: int = 15,
    n_eval_benign: int = 15,
    n_calib: int = 200,
    seed: int = 17,
    csic_path: str = "data/raw_text/csic_all.parquet",
) -> tuple[ToolContext, list[Case], float]:
    """Return (tool context, evaluation cases, calibrated detector threshold)."""
    rng = np.random.RandomState(seed)
    total_norm = n_ref_normal + n_eval_benign + n_calib
    total_atk = n_ref_attack + n_eval_attack + n_calib
    df = load_csic(csic_path, n_normal=total_norm, n_anomalous=total_atk, seed=seed)

    normal_df = df[df["label"] == 0].reset_index(drop=True)
    attack_df = df[df["label"] == 1].reset_index(drop=True)
    norm_idx = rng.permutation(len(normal_df))
    atk_idx = rng.permutation(len(attack_df))

    # disjoint slices: reference | calibration | evaluation
    ref_norm = normal_df.iloc[norm_idx[:n_ref_normal]]
    cal_norm = normal_df.iloc[norm_idx[n_ref_normal:n_ref_normal + n_calib]]
    ev_norm = normal_df.iloc[norm_idx[n_ref_normal + n_calib:
                                      n_ref_normal + n_calib + n_eval_benign]]
    ref_atk = attack_df.iloc[atk_idx[:n_ref_attack]]
    cal_atk = attack_df.iloc[atk_idx[n_ref_attack:n_ref_attack + n_calib]]
    ev_atk = attack_df.iloc[atk_idx[n_ref_attack + n_calib:
                                    n_ref_attack + n_calib + n_eval_attack]]

    normal_threats = build_text_threats(ref_norm)
    attack_threats = build_text_threats(ref_atk)
    # The retrieval corpus must contain BOTH known attacks and known normal traffic. With an
    # attack-only corpus the "most similar known case" is always an attack, which pushes every
    # verdict towards a false positive regardless of the evidence.
    known_threats = attack_threats + normal_threats[:len(attack_threats)]
    ctx = build_tool_context(normal_threats, known_threats,
                             normal_requests=ref_norm["requests"].tolist())

    cases = ([_case(r, i, "attack") for i, r in enumerate(ev_atk["requests"])]
             + [_case(r, i, "benign") for i, r in enumerate(ev_norm["requests"])])
    rng.shuffle(cases)

    threshold = calibrate_threshold(ctx, cal_norm["requests"].tolist(),
                                    cal_atk["requests"].tolist())
    return ctx, cases, threshold


def load_holdout_setup(
    holdout: str,
    n_ref_normal: int = 1500,
    n_ref_attack: int = 400,
    n_eval_attack: int = 60,
    n_eval_benign: int = 60,
    n_calib: int = 200,
    seed: int = 17,
    csic_path: str = "data/raw_text/csic_all.parquet",
) -> tuple[ToolContext, list[Case], float]:
    """Leave-one-attack-type-out: the held-out type is absent from everything the system knows.

    This is the zero-day setting proper. The retrieval corpus, the known-normal profile and the
    detector's calibration split contain every attack type EXCEPT `holdout`; the evaluation cases
    are attacks of exactly that type. The detector's threshold is therefore tuned on a
    distribution that does not contain the attacks it will be asked to catch, which removes the
    advantage it enjoys in the in-distribution experiment.
    """
    from ..data_text import attack_type, parse_request

    rng = np.random.RandomState(seed)
    # ask for plenty of attacks: the held-out type may be a small share of the corpus
    df = load_csic(csic_path, n_normal=n_ref_normal + n_eval_benign + n_calib,
                   n_anomalous=12000, seed=seed)

    normal_df = df[df["label"] == 0].reset_index(drop=True)
    attack_df = df[df["label"] == 1].reset_index(drop=True).copy()
    attack_df["atype"] = [attack_type(parse_request(r)) for r in attack_df["requests"]]

    held = attack_df[attack_df["atype"] == holdout].reset_index(drop=True)
    rest = attack_df[attack_df["atype"] != holdout].reset_index(drop=True)
    if len(held) < n_eval_attack:
        raise ValueError(f"only {len(held)} '{holdout}' attacks available, need {n_eval_attack}")

    norm_idx = rng.permutation(len(normal_df))
    rest_idx = rng.permutation(len(rest))
    held_idx = rng.permutation(len(held))

    ref_norm = normal_df.iloc[norm_idx[:n_ref_normal]]
    cal_norm = normal_df.iloc[norm_idx[n_ref_normal:n_ref_normal + n_calib]]
    ev_norm = normal_df.iloc[norm_idx[n_ref_normal + n_calib:
                                      n_ref_normal + n_calib + n_eval_benign]]
    # reference and calibration attacks: every type except the held-out one
    ref_atk = rest.iloc[rest_idx[:n_ref_attack]]
    cal_atk = rest.iloc[rest_idx[n_ref_attack:n_ref_attack + n_calib]]
    # evaluation attacks: only the held-out type
    ev_atk = held.iloc[held_idx[:n_eval_attack]]

    normal_threats = build_text_threats(ref_norm)
    attack_threats = build_text_threats(ref_atk)
    known_threats = attack_threats + normal_threats[:len(attack_threats)]
    ctx = build_tool_context(normal_threats, known_threats,
                             normal_requests=ref_norm["requests"].tolist())

    cases = ([_case(r, i, "attack") for i, r in enumerate(ev_atk["requests"])]
             + [_case(r, i, "benign") for i, r in enumerate(ev_norm["requests"])])
    for c in cases:
        if c.label == "attack":
            c.family = holdout
    rng.shuffle(cases)

    threshold = calibrate_threshold(ctx, cal_norm["requests"].tolist(),
                                    cal_atk["requests"].tolist())
    return ctx, cases, threshold


def load_hard_negative_setup(
    n_attack: int = 40,
    n_plain_benign: int = 40,
    n_hard_benign: int = 40,
    n_ref_normal: int = 1500,
    n_ref_attack: int = 400,
    n_calib: int = 200,
    seed: int = 17,
    csic_path: str = "data/raw_text/csic_all.parquet",
    generated_probes: bool = False,
) -> tuple[ToolContext, list[Case], float]:
    """Evaluation set containing legitimate traffic that superficially resembles an attack.

    With `generated_probes=True` the legitimate-but-alarming group is produced by the documented
    procedure in `probes.py` from held-out normal traffic, which both scales the group beyond the
    forty hand-written cases and removes the objection that the cases were invented by hand.

    The benign half is split into ordinary normal traffic and hand-written "hard negatives"
    (apostrophes in surnames, SQL words used as vocabulary, encoded diacritics, file paths in
    parameters, markup in a review body). Both are benign; only the second looks suspicious.

    This separates two abilities that the in-distribution experiment conflates: noticing that a
    request is unusual, which a distance from the normal profile already does, and judging whether
    an unusual request is actually harmful, which requires reading it.
    """
    from ..probes import generate as generate_probes
    from .hard_negatives import hard_negative_requests

    rng = np.random.RandomState(seed)
    # extra normal traffic beyond the four disjoint slices, used ONLY as the pool that generated
    # probes are derived from, so that no probe's base request was seen while fitting
    n_probe_pool = 1200 if generated_probes else 0
    df = load_csic(csic_path, n_normal=n_ref_normal + n_plain_benign + n_calib + n_probe_pool,
                   n_anomalous=n_ref_attack + n_attack + n_calib, seed=seed)
    normal_df = df[df["label"] == 0].reset_index(drop=True)
    attack_df = df[df["label"] == 1].reset_index(drop=True)
    norm_idx = rng.permutation(len(normal_df))
    atk_idx = rng.permutation(len(attack_df))

    ref_norm = normal_df.iloc[norm_idx[:n_ref_normal]]
    cal_norm = normal_df.iloc[norm_idx[n_ref_normal:n_ref_normal + n_calib]]
    ev_norm = normal_df.iloc[norm_idx[n_ref_normal + n_calib:
                                      n_ref_normal + n_calib + n_plain_benign]]
    ref_atk = attack_df.iloc[atk_idx[:n_ref_attack]]
    cal_atk = attack_df.iloc[atk_idx[n_ref_attack:n_ref_attack + n_calib]]
    ev_atk = attack_df.iloc[atk_idx[n_ref_attack + n_calib:n_ref_attack + n_calib + n_attack]]

    normal_threats = build_text_threats(ref_norm)
    attack_threats = build_text_threats(ref_atk)
    known_threats = attack_threats + normal_threats[:len(attack_threats)]
    ctx = build_tool_context(normal_threats, known_threats,
                             normal_requests=ref_norm["requests"].tolist())

    if generated_probes:
        # probes derived from held-out normal traffic by the documented procedure in probes.py;
        # the base requests sit outside every fitted slice
        pool = normal_df.iloc[norm_idx[n_ref_normal + n_calib + n_plain_benign:]]
        hard = [pr.raw_request
                for pr in generate_probes(pool["requests"].astype(str).tolist(), n_hard_benign,
                                          seed=seed)]
    else:
        hard = hard_negative_requests()[:n_hard_benign]
    cases = [_case(r, i, "attack") for i, r in enumerate(ev_atk["requests"])]
    for c in cases:
        c.family = "csic_attack"
    plain = [_case(r, i, "benign") for i, r in enumerate(ev_norm["requests"])]
    for c in plain:
        c.family = "benign_plain"
    hardc = [_case(r, 100 + i, "benign") for i, r in enumerate(hard)]
    for c in hardc:
        c.family = "benign_hard"
    cases = cases + plain + hardc
    rng.shuffle(cases)

    threshold = calibrate_threshold(ctx, cal_norm["requests"].tolist(),
                                    cal_atk["requests"].tolist())
    return ctx, cases, threshold


def calibrate_threshold(ctx: ToolContext, benign_reqs: list[str], attack_reqs: list[str]) -> float:
    """Youden-optimal novelty threshold on a held-out calibration split (never the eval cases)."""
    from sklearn.metrics import roc_curve

    scores, labels = [], []
    for req, lab in [(r, 0) for r in benign_reqs] + [(r, 1) for r in attack_reqs]:
        scores.append(novelty_score(ctx, Case("calib", req))["novelty_u"])
        labels.append(lab)
    fpr, tpr, thr = roc_curve(labels, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def _case(raw: str, i: int, label: str) -> Case:
    return Case(case_id=f"{label}-{i:03d}", raw_request=str(raw), source="csic", label=label)


def load_cases_from_log(log_path: str = "env/logs/requests.jsonl",
                        truth_path: str = "env/logs/ground_truth.json") -> list[Case]:
    """Build cases from the mock environment's request log (live traffic instead of CSIC).

    The ground-truth label lives only in the side file; the agents see nothing but the
    reconstructed request line, query and body.
    """
    truth: dict[str, dict] = {}
    if os.path.exists(truth_path):
        with open(truth_path) as fh:
            truth = {t["case_id"]: t for t in json.load(fh)}

    cases: list[Case] = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            url = rec["path"] + (f"?{rec['query']}" if rec.get("query") else "")
            raw = (f"{rec['method']} {url} HTTP/1.1\n"
                   f"User-Agent: {rec.get('user_agent', '')}\nHost: mockshop\n\n"
                   f"{rec.get('body', '')}")
            t = truth.get(rec.get("case_id", ""), {})
            cases.append(Case(case_id=rec.get("case_id") or f"log-{len(cases):03d}",
                              raw_request=raw, source="live", label=t.get("label"),
                              family=t.get("family"),
                              meta={"ts": rec.get("ts"), "client": rec.get("client")}))
    return cases
