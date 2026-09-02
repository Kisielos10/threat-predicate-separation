"""Experiment orchestration: Exp 0 (sanity), Exp A (threat vs normal), Exp B (zero-day LOO).

Design notes on fairness / leakage:
  * A and sigma are derived from the attack *family* (taxonomy), i.e. from the label. So for
    Exp A ("is an event a threat?") using them would be circular — Exp A therefore scores
    threats with the SEMANTIC-ONLY similarity (phi of the flow text, which never sees the
    label) plus, as a reference, a raw-feature baseline.
  * For Exp B ("is a threat a zero-day?") the held-out family is treated as unseen. We fit
    the known model on a TRAIN split of the *other* families and test on a disjoint split,
    so held-in ("known") test threats are not their own centroids (no in-sample optimism).
    The semantic-only variant uses no label-derived info for test threats and is the
    conservative/clean zero-day detector; the composite variant shows what adding documented
    domain knowledge (A/sigma) does — it can raise OR lower novelty, so it is a real test.
"""

from __future__ import annotations

import numpy as np

from .metrics import auroc, evaluate_scores
from .novelty import KnownThreatModel
from .similarity import (
    WEIGHTS_FULL,
    WEIGHTS_NO_SEM,
    WEIGHTS_SEM_M,
    WEIGHTS_SEM_ONLY,
    SimWeights,
    composite_sim,
)
from .threats import Threat

# Similarity-variant presets compared in Exp B, grouped by deployability.
#   deployable  = uses only observable components (phi, metadata M) — valid zero-day detector
#   oracle      = uses A/sigma derived from the family label — UPPER BOUND, not deployable
SIM_VARIANTS: dict[str, SimWeights] = {
    "semantic_only": WEIGHTS_SEM_ONLY,            # deployable
    "semantic+metadata": WEIGHTS_SEM_M,           # deployable
    "composite (oracle A,sigma)": WEIGHTS_FULL,   # oracle upper bound
    "no_semantic (oracle A,sigma,M)": WEIGHTS_NO_SEM,  # oracle upper bound
}
DEPLOYABLE_VARIANTS = {"semantic_only", "semantic+metadata", "raw_1nn", "isolation_forest"}
ORACLE_VARIANTS = {"composite (oracle A,sigma)", "no_semantic (oracle A,sigma,M)"}


# ----------------------------------------------------------------------------------------
# Exp 0 — reproduce the document's SQL-injection worked example (sim sanity check)
# ----------------------------------------------------------------------------------------

def _manual_threat(text: str, techniques: set[str], sigma: tuple[float, float, float],
                   m_vec: list[float], family: str) -> Threat:
    from .taxonomy import ATTACK_TECHNIQUES

    A = np.zeros(len(ATTACK_TECHNIQUES))
    idx = {t: i for i, t in enumerate(ATTACK_TECHNIQUES)}
    for t in techniques:
        if t in idx:
            A[idx[t]] = 1.0
    return Threat(
        events=[], K=("application", "HTTP"), Tg=("protected_asset", "auth-service"),
        D="Web", A=A, sigma=np.asarray(sigma), m_vec=np.asarray(m_vec, dtype=float),
        raw_vec=np.zeros(1), text=text, family=family, is_attack=True,
    )


def exp0_sql_injection(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> dict:
    """Build Th1/Th2/Th3 from the doc and check sim(Th1,Th2) >> sim(Th1,Th3).

    The doc reports raw semantic cosine sim(Th1,Th2) ~ 0.91 and sim(Th1,Th3) ~ 0.18, plus
    high/low composite sim. We report both raw cosine and composite to verify ordering and
    rough magnitude.
    """
    from .embedding import embed_texts
    from .similarity import _cosine

    th1 = _manual_threat(
        "HTTP POST request to /login endpoint with body payload \"' OR '1'='1\". "
        "SQL injection attempt against the authentication service.",
        {"T1190"}, (0.7, 0.3, 0.0), [80.0, 6.0, 14.0], "WebAttack")
    th2 = _manual_threat(
        "HTTP GET request to /search endpoint with query \"q=' UNION SELECT username,password\". "
        "SQL injection attempt extracting database contents.",
        {"T1190"}, (0.8, 0.2, 0.0), [80.0, 6.0, 16.0], "WebAttack")
    th3 = _manual_threat(
        "Sequence of TCP SYN probes scanning many destination ports on a host. "
        "Network port-scan reconnaissance activity.",
        {"T1046"}, (0.1, 0.0, 0.0), [0.0, 6.0, 14.0], "PortScan")

    texts = [th1.text, th2.text, th3.text]
    emb = embed_texts(texts, model_name=model_name)
    # standardise the tiny metadata vectors among the three so sim_M is comparable
    M = np.vstack([th1.m_vec, th2.m_vec, th3.m_vec])
    sd = M.std(axis=0)
    sd[sd == 0] = 1.0
    mu = M.mean(axis=0)
    for t, e, m in zip((th1, th2, th3), emb, M, strict=True):
        t.phi = e
        t.m_vec = (m - mu) / sd

    cos_12 = _cosine(th1.phi, th2.phi)
    cos_13 = _cosine(th1.phi, th3.phi)
    return {
        "cosine_sem_Th1_Th2": cos_12,
        "cosine_sem_Th1_Th3": cos_13,
        "composite_Th1_Th2": composite_sim(th1, th2, WEIGHTS_FULL),
        "composite_Th1_Th3": composite_sim(th1, th3, WEIGHTS_FULL),
    }


# ----------------------------------------------------------------------------------------
# Exp A — threat vs normal  (Def 5 precondition: attacks separable from benign)
# ----------------------------------------------------------------------------------------

def exp_a_threat_vs_normal(
    threats: list[Threat], seed: int = 17, benign_train_frac: float = 0.6,
    weights: SimWeights = WEIGHTS_SEM_ONLY, min_cluster_size: int = 10,
) -> dict:
    """Score every threat by novelty u relative to a model of NORMAL (benign) behaviour.

    Benign profile is clustered from a benign-train split; the detector flags high-u events
    as threats. Evaluated on benign-test + all attacks. Semantic-only by design (no leakage).
    """
    rng = np.random.RandomState(seed)
    benign = [t for t in threats if not t.is_attack]
    attacks = [t for t in threats if t.is_attack]
    perm = rng.permutation(len(benign))
    cut = int(len(benign) * benign_train_frac)
    benign_train = [benign[i] for i in perm[:cut]]
    benign_test = [benign[i] for i in perm[cut:]]

    model = KnownThreatModel(weights, min_cluster_size=min_cluster_size).fit(benign_train)
    test = benign_test + attacks
    scores = model.novelty_batch(test)
    y = np.array([t.is_attack for t in test], dtype=int)

    det = evaluate_scores(scores, y)
    per_family = _per_family_mean(test, scores)
    # non-leaky raw-feature reference: distance to nearest benign-train flow
    raw_scores = _raw_knn_novelty(benign_train, test)
    return {"metrics": det.as_dict(), "per_family_mean_u": per_family,
            "raw_baseline_auroc": auroc(raw_scores, y),
            "n_benign_train": len(benign_train), "n_test": len(test)}


# ----------------------------------------------------------------------------------------
# Exp B — zero-day, leave-one-attack-family-out  (Def 13)
# ----------------------------------------------------------------------------------------

def exp_b_leave_one_family_out(
    threats: list[Threat], seed: int = 17, known_train_frac: float = 0.7,
    min_cluster_size: int = 10,
) -> dict:
    """For each attack family f: hold it out as 'unseen', fit known model on a train split of
    the other families, and measure AUROC of novelty u separating f (novel) from held-in
    families (known). Runs all sim variants + raw-feature and IsolationForest baselines.
    """
    attacks = [t for t in threats if t.is_attack]
    families = sorted({t.family for t in attacks})

    rows: list[dict] = []
    for f in families:
        fold = fold_scores_for_family(threats, f, seed, known_train_frac, min_cluster_size)
        if fold is None:
            continue
        y = fold["y"]
        for variant, scores in fold["scores"].items():
            rows.append({"held_out": f, "variant": variant, "auroc": auroc(scores, y),
                         "n_novel": int(y.sum()), "n_known_test": int((y == 0).sum())})

    aggregated = _aggregate_auroc(rows)
    return {"rows": rows, "aggregated": aggregated, "families": families}


def fold_scores_for_family(
    threats: list[Threat], held_out: str, seed: int = 17,
    known_train_frac: float = 0.7, min_cluster_size: int = 10,
) -> dict | None:
    """Novelty scores for one leave-one-family-out fold (all variants + baselines).

    Returns {"y": labels, "scores": {variant: np.ndarray}} where label 1 = held-out (novel)
    family, 0 = held-in (known) family. Returns None if the fold is too small. Used both by
    the aggregate Exp B sweep and by the plotting code (ROC curves of a representative fold).
    """
    attacks = [t for t in threats if t.is_attack]
    novel = [t for t in attacks if t.family == held_out]
    others = [t for t in attacks if t.family != held_out]
    if len(novel) < 20 or len(others) < 50:
        return None

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(others))
    cut = int(len(others) * known_train_frac)
    known_train = [others[i] for i in perm[:cut]]
    known_test = [others[i] for i in perm[cut:]]
    test = known_test + novel
    y = np.array([0] * len(known_test) + [1] * len(novel), dtype=int)

    scores: dict[str, np.ndarray] = {}
    for name, w in SIM_VARIANTS.items():
        model = KnownThreatModel(w, min_cluster_size=min_cluster_size).fit(known_train)
        scores[name] = model.novelty_batch(test)
    scores["raw_1nn"] = _raw_knn_novelty(known_train, test)
    scores["isolation_forest"] = _iforest_novelty(known_train, test, seed)
    return {"y": y, "scores": scores, "held_out": held_out}


# ----------------------------------------------------------------------------------------
# Baselines on raw flow features
# ----------------------------------------------------------------------------------------

def _raw_matrix(threats: list[Threat]) -> np.ndarray:
    return np.vstack([t.raw_vec for t in threats])


def _raw_knn_novelty(known_train: list[Threat], test: list[Threat]) -> np.ndarray:
    """Novelty = distance to nearest known-train flow in standardised raw-feature space."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(_raw_matrix(known_train))
    Xtr = scaler.transform(_raw_matrix(known_train))
    Xte = scaler.transform(_raw_matrix(test))
    nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
    dist, _ = nn.kneighbors(Xte)
    return dist.ravel()


def _iforest_novelty(known_train: list[Threat], test: list[Threat], seed: int) -> np.ndarray:
    """Classic open-set baseline: IsolationForest anomaly score (higher = more novel)."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(_raw_matrix(known_train))
    Xtr = scaler.transform(_raw_matrix(known_train))
    Xte = scaler.transform(_raw_matrix(test))
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr)
    return -iso.score_samples(Xte)  # negate: larger => more anomalous


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------

def _per_family_mean(threats: list[Threat], scores: np.ndarray) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for t, s in zip(threats, scores, strict=True):
        out.setdefault(t.family, []).append(float(s))
    return {k: float(np.mean(v)) for k, v in sorted(out.items())}


def _aggregate_auroc(rows: list[dict]) -> dict[str, float]:
    by_variant: dict[str, list[float]] = {}
    for r in rows:
        if not np.isnan(r["auroc"]):
            by_variant.setdefault(r["variant"], []).append(r["auroc"])
    return {k: float(np.mean(v)) for k, v in by_variant.items()}
