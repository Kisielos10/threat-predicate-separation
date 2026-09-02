"""Composite similarity sim — Definition 12.

  sim(Th_a, Th_b) = w_sem*sim_sem + w_A*sim_A + w_sigma*sim_sigma + w_M*sim_M

Each partial similarity is normalised to [0,1]; weights are non-negative and sum to 1, so
sim in [0,1] (1 = identical, 0 = completely different).

  sim_sem   cosine of embeddings phi, mapped to [0,1] via (1+cos)/2
  sim_A     Jaccard overlap of ATT&CK technique sets
  sim_sigma 1 - normalised L2 distance of CIA vectors sigma in [0,1]^3
  sim_M     metadata closeness, 1/(1+normalised L2) on standardised metadata vectors

NOTE: metadata vectors `m_vec` must be standardised across the population before use (the
data pipeline does this), so sim_M is comparable across threats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .threats import Threat

_SQRT3 = np.sqrt(3.0)


@dataclass(frozen=True)
class SimWeights:
    """Weights for the composite similarity (Def 12). Must be >=0 and sum to 1."""

    w_sem: float = 0.55
    w_A: float = 0.20
    w_sigma: float = 0.10
    w_M: float = 0.15

    def __post_init__(self) -> None:
        total = self.w_sem + self.w_A + self.w_sigma + self.w_M
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {total}")
        if min(self.w_sem, self.w_A, self.w_sigma, self.w_M) < 0:
            raise ValueError("weights must be non-negative")

    def as_array(self) -> np.ndarray:
        return np.array([self.w_sem, self.w_A, self.w_sigma, self.w_M])


# Common weight presets used by Exp B (full composite + ablation variants).
# NOTE on deployability: A and sigma are derived from the attack family (the label), so any
# variant with w_A>0 or w_sigma>0 uses information unavailable for a *truly unseen* zero-day
# at test time — those are ORACLE upper bounds. Variants using only phi and metadata M
# (w_sem, w_M) are DEPLOYABLE (observable without the label).
WEIGHTS_FULL = SimWeights(0.55, 0.20, 0.10, 0.15)          # oracle (uses A, sigma)
WEIGHTS_SEM_ONLY = SimWeights(1.0, 0.0, 0.0, 0.0)          # deployable
WEIGHTS_SEM_M = SimWeights(0.786, 0.0, 0.0, 0.214)         # deployable (phi + metadata)
WEIGHTS_NO_SEM = SimWeights(0.0, 0.50, 0.25, 0.25)         # oracle (A + sigma + M)


# --- partial similarities (each -> [0,1]) ---

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sim_sem(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
    return (1.0 + _cosine(phi_a, phi_b)) / 2.0


def sim_A(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.sum((a > 0) & (b > 0)))
    union = float(np.sum((a > 0) | (b > 0)))
    if union == 0:  # both have no techniques (e.g. benign vs benign)
        return 1.0
    return inter / union


def sim_sigma(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.linalg.norm(a - b) / _SQRT3)


def sim_M(a: np.ndarray, b: np.ndarray) -> float:
    dim = max(len(a), 1)
    d = float(np.linalg.norm(a - b)) / np.sqrt(dim)
    return 1.0 / (1.0 + d)


def composite_sim(th_a: Threat, th_b: Threat, weights: SimWeights = WEIGHTS_FULL) -> float:
    """sim(Th_a, Th_b) per Def 12. Both threats must have phi populated."""
    if th_a.phi is None or th_b.phi is None:
        raise ValueError("composite_sim requires phi to be populated on both threats")
    s_sem = sim_sem(th_a.phi, th_b.phi) if weights.w_sem else 0.0
    s_A = sim_A(th_a.A, th_b.A) if weights.w_A else 0.0
    s_sig = sim_sigma(th_a.sigma, th_b.sigma) if weights.w_sigma else 0.0
    s_M = sim_M(th_a.m_vec, th_b.m_vec) if weights.w_M else 0.0
    return (
        weights.w_sem * s_sem
        + weights.w_A * s_A
        + weights.w_sigma * s_sig
        + weights.w_M * s_M
    )


def sim_to_centroids(th: Threat, centroids: list[Threat], weights: SimWeights) -> np.ndarray:
    """Vectorised composite sim of one threat to a list of centroid threats (for novelty)."""
    return np.array([composite_sim(th, c, weights) for c in centroids], dtype=np.float64)
