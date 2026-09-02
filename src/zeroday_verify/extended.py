"""Extended verification analyses for the supervisor report.

(1) multi_seed_*        — robustness: mean +/- std / 95% CI over many subsamples & splits.
(2) technique_inference — replace oracle A/sigma with values *predicted* from observation,
                          measuring how much of the oracle gap is realistically recoverable.
(3) weight_sensitivity  — sweep the deployable weights (w_sem vs w_M) for Exp B.

A single large CIC threat pool is built once (embedded once) and subsampled per seed, so the
expensive CSV read + embedding happen only once.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import data as D
from .experiments import exp_a_threat_vs_normal, exp_b_leave_one_family_out
from .metrics import auroc
from .novelty import KnownThreatModel
from .similarity import WEIGHTS_FULL, WEIGHTS_SEM_ONLY, SimWeights
from .threats import Threat

# ---------------------------------------------------------------------------------------
# shared pool
# ---------------------------------------------------------------------------------------

def build_pool(raw_dir: str = "data/raw", per_attack_family: int = 1200,
               n_benign: int = 6000, seed: int = 17) -> list[Threat]:
    """Load + embed a large CIC threat pool once; subsample per seed downstream."""
    paths = D.discover_csvs(raw_dir)
    df = D.load_subsampled(paths, per_attack_family=per_attack_family, n_benign=n_benign, seed=seed)
    return D.build_threats(df)


def subsample_pool(pool: list[Threat], per_attack_family: int = 600, n_benign: int = 3000,
                   seed: int = 17) -> list[Threat]:
    rng = np.random.RandomState(seed)
    by_fam: dict[str, list[Threat]] = {}
    for t in pool:
        by_fam.setdefault(t.family, []).append(t)
    out: list[Threat] = []
    for fam, items in by_fam.items():
        cap = n_benign if fam == "BENIGN" else per_attack_family
        if len(items) <= cap:
            out.extend(items)
        else:
            idx = rng.choice(len(items), size=cap, replace=False)
            out.extend(items[i] for i in idx)
    return out


# ---------------------------------------------------------------------------------------
# (1) multi-seed robustness
# ---------------------------------------------------------------------------------------

def multi_seed(pool: list[Threat], seeds: list[int], per_attack_family: int = 600,
               n_benign: int = 3000) -> dict:
    """Run Exp A and Exp B across seeds; return per-seed values + summary stats."""
    expA_auroc: list[float] = []
    expB_rows: list[dict] = []          # (seed, variant, auroc) aggregated per seed
    expB_per_family: list[dict] = []    # (seed, held_out, variant, auroc)

    for s in seeds:
        sub = subsample_pool(pool, per_attack_family, n_benign, seed=s)
        a = exp_a_threat_vs_normal(sub, seed=s)
        expA_auroc.append(a["metrics"]["AUROC"])
        b = exp_b_leave_one_family_out(sub, seed=s)
        for variant, val in b["aggregated"].items():
            expB_rows.append({"seed": s, "variant": variant, "auroc": val})
        for r in b["rows"]:
            expB_per_family.append({"seed": s, **r})

    return {
        "seeds": seeds,
        "expA_auroc": expA_auroc,
        "expA_summary": _summary(expA_auroc),
        "expB_rows": expB_rows,
        "expB_summary": _summary_by(expB_rows, "variant"),
        "expB_per_family": expB_per_family,
        "expB_per_family_summary": _summary_by(expB_per_family, ("held_out", "variant")),
    }


def _summary(vals: list[float]) -> dict:
    a = np.asarray([v for v in vals if not np.isnan(v)], dtype=float)
    if len(a) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    mean, std = float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
    ci = 1.96 * std / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": float(ci), "n": int(len(a))}


def _summary_by(rows: list[dict], key) -> dict:
    groups: dict = {}
    for r in rows:
        k = r[key] if isinstance(key, str) else tuple(r[kk] for kk in key)
        groups.setdefault(k, []).append(r["auroc"])
    return {k: _summary(v) for k, v in groups.items()}


# ---------------------------------------------------------------------------------------
# (2) technique-inference probe (realistic, non-oracle A/sigma)
# ---------------------------------------------------------------------------------------

def technique_inference(pool: list[Threat], seed: int = 17, per_attack_family: int = 600,
                        known_train_frac: float = 0.7) -> dict:
    """Compare semantic-only vs predicted-composite vs oracle-composite per held-out family.

    For each fold, a classifier (A) and regressor (sigma) are trained on the KNOWN families'
    (phi -> A/sigma) and used to *predict* A/sigma for the test threats — i.e. A/sigma are
    estimated from observation, not read from the label. Known centroids keep their true
    A/sigma (the known families really are known). This estimates how much of the oracle
    composite is recoverable without the label.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputRegressor

    sub = subsample_pool(pool, per_attack_family, n_benign=0, seed=seed)
    attacks = [t for t in sub if t.is_attack]
    families = sorted({t.family for t in attacks})
    rows: list[dict] = []

    for f in families:
        novel = [t for t in attacks if t.family == f]
        others = [t for t in attacks if t.family != f]
        if len(novel) < 20 or len(others) < 50:
            continue
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(others))
        cut = int(len(others) * known_train_frac)
        known_train = [others[i] for i in perm[:cut]]
        known_test = [others[i] for i in perm[cut:]]
        test = known_test + novel
        y = np.array([0] * len(known_test) + [1] * len(novel), dtype=int)

        # train attribution models on known_train (phi -> A bits / sigma)
        Xtr = np.vstack([t.phi for t in known_train])
        A_tr = np.vstack([t.A for t in known_train])
        sig_tr = np.vstack([t.sigma for t in known_train])
        Xte = np.vstack([t.phi for t in test])
        # predict each technique bit independently, guarding constant-class columns
        A_hat = np.zeros((len(test), A_tr.shape[1]))
        for j in range(A_tr.shape[1]):
            col = A_tr[:, j]
            if col.max() == col.min():           # technique absent (or constant) in training
                A_hat[:, j] = col.max()          # predict that constant value
            else:
                lr = LogisticRegression(max_iter=400).fit(Xtr, col)
                A_hat[:, j] = lr.predict(Xte)
        reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=120, random_state=seed))
        reg.fit(Xtr, sig_tr)
        sig_hat = np.clip(reg.predict(Xte), 0.0, 1.0)

        # known centroids keep TRUE A/sigma; test threats get oracle vs predicted A/sigma
        model = KnownThreatModel(WEIGHTS_FULL, min_cluster_size=10).fit(known_train)
        sem = KnownThreatModel(WEIGHTS_SEM_ONLY, min_cluster_size=10).fit(known_train)
        oracle_scores = model.novelty_batch(test)
        pred_test = [replace(t, A=A_hat[i], sigma=sig_hat[i]) for i, t in enumerate(test)]
        pred_scores = model.novelty_batch(pred_test)
        sem_scores = sem.novelty_batch(test)

        rows.append({"held_out": f, "semantic_only": auroc(sem_scores, y),
                     "predicted_composite": auroc(pred_scores, y),
                     "oracle_composite": auroc(oracle_scores, y)})

    agg = {k: float(np.nanmean([r[k] for r in rows]))
           for k in ("semantic_only", "predicted_composite", "oracle_composite")} if rows else {}
    return {"rows": rows, "aggregated": agg}


# ---------------------------------------------------------------------------------------
# (3) weight sensitivity (deployable: w_sem vs w_M)
# ---------------------------------------------------------------------------------------

def weight_sensitivity(pool: list[Threat], seed: int = 17, per_attack_family: int = 600,
                       grid: list[float] | None = None) -> dict:
    """Sweep w_sem in [0,1] with w_M = 1 - w_sem (w_A = w_sigma = 0). Mean LOO AUROC each."""
    if grid is None:
        grid = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sub = subsample_pool(pool, per_attack_family, n_benign=0, seed=seed)
    attacks = [t for t in sub if t.is_attack]
    families = sorted({t.family for t in attacks})

    curve: list[dict] = []
    for w_sem in grid:
        w_m = round(1.0 - w_sem, 3)
        weights = SimWeights(w_sem, 0.0, 0.0, w_m)
        aurocs = []
        for f in families:
            res = _loo_fold_auroc(attacks, f, weights, seed)
            if res is not None:
                aurocs.append(res)
        curve.append({"w_sem": w_sem, "w_M": w_m,
                      "mean_auroc": float(np.nanmean(aurocs)) if aurocs else float("nan")})
    best = max((c for c in curve if not np.isnan(c["mean_auroc"])),
               key=lambda c: c["mean_auroc"], default=None)
    return {"curve": curve, "best": best}


def _loo_fold_auroc(attacks: list[Threat], held_out: str, weights: SimWeights,
                    seed: int, known_train_frac: float = 0.7) -> float | None:
    novel = [t for t in attacks if t.family == held_out]
    others = [t for t in attacks if t.family != held_out]
    if len(novel) < 20 or len(others) < 50:
        return None
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(others))
    cut = int(len(others) * known_train_frac)
    known_train = [others[i] for i in perm[:cut]]
    test = [others[i] for i in perm[cut:]] + novel
    y = np.array([0] * (len(others) - cut) + [1] * len(novel), dtype=int)
    model = KnownThreatModel(weights, min_cluster_size=10).fit(known_train)
    return auroc(model.novelty_batch(test), y)
