"""Novelty measure u — Definition 13.

  u(Th) = 1 - max_k sim(Th, mu_k)

where mu_k are centroids of clusters of *known* threats in embedding space phi. u -> 1 means
the threat is far from every known cluster (a zero-day candidate); u -> 0 means it is a
variant of a known attack.

Clustering uses HDBSCAN over L2-normalised phi (DBSCAN optional). Each dense cluster is
represented by its medoid (the actual known threat closest to the cluster mean) so the
composite similarity sim (Def 12), which needs full threat fields, can be evaluated against
it. Known points HDBSCAN labels as noise are kept as singleton centroids, so a known-but-
sparse threat is still recognised as known (avoids spurious novelty).
"""

from __future__ import annotations

import numpy as np

from .similarity import SimWeights, sim_to_centroids
from .threats import Threat


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class KnownThreatModel:
    """Model of known threats: clusters of phi with medoid centroids mu_k (Def 13)."""

    def __init__(
        self,
        weights: SimWeights,
        algorithm: str = "hdbscan",
        min_cluster_size: int = 10,
        eps: float = 0.5,
        keep_noise_singletons: bool = True,
    ) -> None:
        self.weights = weights
        self.algorithm = algorithm
        self.min_cluster_size = min_cluster_size
        self.eps = eps
        self.keep_noise_singletons = keep_noise_singletons
        self.centroids: list[Threat] = []
        self.labels_: np.ndarray | None = None
        self.n_clusters_: int = 0

    def fit(self, known: list[Threat]) -> KnownThreatModel:
        if not known:
            raise ValueError("cannot fit KnownThreatModel on empty known set")
        phi = _l2_normalize(np.vstack([t.phi for t in known]))
        labels = self._cluster(phi)
        self.labels_ = labels

        centroids: list[Threat] = []
        for lab in sorted(set(labels)):
            idx = np.where(labels == lab)[0]
            if lab == -1:
                if self.keep_noise_singletons:
                    centroids.extend(known[i] for i in idx)
                continue
            # medoid: known threat closest to the cluster mean in phi space
            mean = phi[idx].mean(axis=0)
            medoid = idx[int(np.argmax(phi[idx] @ mean))]
            centroids.append(known[medoid])

        self.n_clusters_ = len([lab for lab in set(labels) if lab != -1])
        if not centroids:  # degenerate: everything was noise and singletons disabled
            centroids = list(known)
        self.centroids = centroids
        return self

    def _cluster(self, phi: np.ndarray) -> np.ndarray:
        from sklearn.cluster import DBSCAN, HDBSCAN

        if self.algorithm == "hdbscan":
            model = HDBSCAN(min_cluster_size=self.min_cluster_size, metric="euclidean")
        elif self.algorithm == "dbscan":
            model = DBSCAN(eps=self.eps, min_samples=self.min_cluster_size, metric="euclidean")
        else:
            raise ValueError(f"unknown clustering algorithm: {self.algorithm}")
        return model.fit_predict(phi)

    def novelty(self, th: Threat) -> float:
        """u(Th) = 1 - max_k sim(Th, mu_k) (Def 13)."""
        sims = sim_to_centroids(th, self.centroids, self.weights)
        return float(1.0 - sims.max())

    def novelty_batch(self, threats: list[Threat]) -> np.ndarray:
        return np.array([self.novelty(t) for t in threats], dtype=np.float64)


def novelty_u(th: Threat, centroids: list[Threat], weights: SimWeights) -> float:
    """Standalone novelty u given explicit centroids (Def 13)."""
    sims = sim_to_centroids(th, centroids, weights)
    return float(1.0 - sims.max())
