"""Web-domain experiments on CSIC-2010 HTTP requests.

Exp A-text : anomaly vs normal — does phi separate web attacks from benign when it can see
             the payload? (Contrast with the flow-domain WebAttack failure.)
Exp B-text : leave-one-attack-type-out novelty across heuristic web-attack types.

Deployable methods only (semantic phi, and a char-ngram TF-IDF + 1NN baseline). No oracle
A/sigma here — for text the type itself is heuristic, so a composite would be circular.
"""

from __future__ import annotations

import numpy as np

from .metrics import auroc, evaluate_scores
from .novelty import KnownThreatModel
from .similarity import WEIGHTS_SEM_ONLY
from .threats import Threat


def _tfidf_novelty(train_texts: list[str], test_texts: list[str]) -> np.ndarray:
    """Char n-gram TF-IDF + nearest-neighbour cosine distance to the known/normal set."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=20000)
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(Xtr)
    dist, _ = nn.kneighbors(Xte)
    return dist.ravel()


def expA_text(threats: list[Threat], seed: int = 17, train_frac: float = 0.6) -> dict:
    """Anomaly (threat) vs normal in the web domain. Semantic novelty + TF-IDF baseline."""
    rng = np.random.RandomState(seed)
    normal = [t for t in threats if not t.is_attack]
    attacks = [t for t in threats if t.is_attack]
    perm = rng.permutation(len(normal))
    cut = int(len(normal) * train_frac)
    normal_train = [normal[i] for i in perm[:cut]]
    test = [normal[i] for i in perm[cut:]] + attacks
    y = np.array([t.is_attack for t in test], dtype=int)

    model = KnownThreatModel(WEIGHTS_SEM_ONLY, min_cluster_size=15).fit(normal_train)
    sem_scores = model.novelty_batch(test)
    tfidf_scores = _tfidf_novelty([t.text for t in normal_train], [t.text for t in test])

    return {
        "semantic": evaluate_scores(sem_scores, y).as_dict(),
        "tfidf_auroc": auroc(tfidf_scores, y),
        "n_normal_train": len(normal_train), "n_test": len(test),
    }


def expB_text(threats: list[Threat], seed: int = 17, known_train_frac: float = 0.7,
              min_type_size: int = 50) -> dict:
    """Leave-one-attack-type-out novelty across heuristic web-attack types."""
    attacks = [t for t in threats if t.is_attack]
    types = sorted({t.family for t in attacks})
    rows: list[dict] = []
    for ty in types:
        novel = [t for t in attacks if t.family == ty]
        others = [t for t in attacks if t.family != ty]
        if len(novel) < min_type_size or len(others) < min_type_size:
            continue
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(others))
        cut = int(len(others) * known_train_frac)
        known_train = [others[i] for i in perm[:cut]]
        known_test = [others[i] for i in perm[cut:]]
        test = known_test + novel
        y = np.array([0] * len(known_test) + [1] * len(novel), dtype=int)

        model = KnownThreatModel(WEIGHTS_SEM_ONLY, min_cluster_size=10).fit(known_train)
        rows.append({"held_out": ty, "variant": "semantic_only",
                     "auroc": auroc(model.novelty_batch(test), y), "n_novel": len(novel)})
        rows.append({"held_out": ty, "variant": "tfidf_1nn",
                     "auroc": auroc(_tfidf_novelty([t.text for t in known_train],
                                                   [t.text for t in test]), y),
                     "n_novel": len(novel)})

    agg: dict[str, list[float]] = {}
    for r in rows:
        if not np.isnan(r["auroc"]):
            agg.setdefault(r["variant"], []).append(r["auroc"])
    aggregated = {k: float(np.mean(v)) for k, v in agg.items()}
    type_counts = {ty: sum(1 for t in attacks if t.family == ty) for ty in types}
    return {"rows": rows, "aggregated": aggregated, "type_counts": type_counts}
