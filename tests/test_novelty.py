"""Tests for the novelty measure u and the known-threat clustering model (Def 13)."""

import numpy as np

from zeroday_verify.novelty import KnownThreatModel, novelty_u
from zeroday_verify.similarity import WEIGHTS_SEM_ONLY
from zeroday_verify.threats import Threat


def _threat(phi, family="known"):
    phi = np.asarray(phi, float)
    return Threat(events=[], K=("t", "TCP"), Tg=("a", "x"), D="Network",
                  A=np.zeros(3), sigma=np.zeros(3), m_vec=np.zeros(2),
                  raw_vec=np.zeros(1), text="", phi=phi, family=family, is_attack=True)


def _cluster(center, n, spread, rng, dim=8):
    base = np.zeros(dim); base[: len(center)] = center
    return [_threat(base + spread * rng.randn(dim)) for _ in range(n)]


def test_known_point_has_low_novelty_far_point_high():
    rng = np.random.RandomState(0)
    # two tight known clusters far apart in phi space
    known = _cluster([5, 5], 40, 0.05, rng) + _cluster([-5, -5], 40, 0.05, rng)
    model = KnownThreatModel(WEIGHTS_SEM_ONLY, min_cluster_size=5).fit(known)

    near = _threat(np.r_[5.0, 5.0, np.zeros(6)])      # sits on a known cluster
    far = _threat(np.r_[5.0, -5.0, np.zeros(6)])      # orthogonal direction, unseen
    u_near = model.novelty(near)
    u_far = model.novelty(far)
    assert 0.0 <= u_near <= 1.0 and 0.0 <= u_far <= 1.0
    assert u_far > u_near
    assert u_near < 0.05  # essentially identical to a known cluster


def test_novelty_separates_held_out_cluster():
    rng = np.random.RandomState(1)
    known = _cluster([8, 0], 50, 0.1, rng)
    model = KnownThreatModel(WEIGHTS_SEM_ONLY, min_cluster_size=5).fit(known)
    novel = _cluster([0, 8], 30, 0.1, rng)          # different direction => novel
    u_known = np.mean(model.novelty_batch(_cluster([8, 0], 20, 0.1, rng)))
    u_novel = np.mean(model.novelty_batch(novel))
    assert u_novel > u_known


def test_novelty_u_standalone_matches_model():
    rng = np.random.RandomState(2)
    known = _cluster([3, 3], 30, 0.05, rng)
    model = KnownThreatModel(WEIGHTS_SEM_ONLY, min_cluster_size=5).fit(known)
    probe = _threat(np.r_[3.0, 3.0, np.zeros(6)])
    assert novelty_u(probe, model.centroids, WEIGHTS_SEM_ONLY) == model.novelty(probe)
