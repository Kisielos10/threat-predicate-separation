"""Tests for the composite similarity (Def 12) + the SQL-injection sanity check (Exp 0)."""

import numpy as np
import pytest

from zeroday_verify.experiments import exp0_sql_injection
from zeroday_verify.similarity import (
    WEIGHTS_FULL,
    SimWeights,
    composite_sim,
    sim_A,
    sim_M,
    sim_sem,
    sim_sigma,
)
from zeroday_verify.threats import Threat


def _threat(phi, A, sigma, m_vec):
    return Threat(events=[], K=("t", "TCP"), Tg=("a", "x"), D="Network",
                  A=np.asarray(A, float), sigma=np.asarray(sigma, float),
                  m_vec=np.asarray(m_vec, float), raw_vec=np.zeros(1), text="",
                  phi=np.asarray(phi, float))


def test_partial_similarities_bounds_and_identity():
    assert sim_sem(np.array([1.0, 0]), np.array([1.0, 0])) == pytest.approx(1.0)
    assert 0.0 <= sim_sem(np.array([1.0, 0]), np.array([-1.0, 0])) <= 1.0
    assert sim_A(np.array([1, 0, 1]), np.array([1, 0, 1])) == pytest.approx(1.0)
    assert sim_A(np.array([1, 0, 0]), np.array([0, 1, 0])) == pytest.approx(0.0)
    assert sim_A(np.array([1, 1, 0]), np.array([1, 0, 0])) == pytest.approx(0.5)
    assert sim_sigma(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])) == pytest.approx(1.0)
    assert sim_M(np.zeros(3), np.zeros(3)) == pytest.approx(1.0)


def test_empty_technique_vectors_are_identical():
    # benign vs benign: no techniques on either side -> sim_A defined as 1.0
    assert sim_A(np.zeros(5), np.zeros(5)) == pytest.approx(1.0)


def test_composite_self_similarity_is_one():
    t = _threat([1.0, 2.0, 3.0], [1, 0, 1], [0.5, 0.2, 0.0], [0.1, 0.2])
    assert composite_sim(t, t, WEIGHTS_FULL) == pytest.approx(1.0, abs=1e-9)


def test_composite_in_unit_interval():
    a = _threat([1.0, 0, 0], [1, 0, 0], [0.9, 0.1, 0.0], [1.0, 0.0])
    b = _threat([0.0, 1, 0], [0, 1, 0], [0.0, 0.0, 1.0], [-1.0, 2.0])
    s = composite_sim(a, b, WEIGHTS_FULL)
    assert 0.0 <= s <= 1.0


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        SimWeights(0.5, 0.5, 0.5, 0.5)


def test_composite_requires_phi():
    t = _threat([1.0], [1], [0.0, 0.0, 0.0], [0.0])
    t.phi = None
    with pytest.raises(ValueError):
        composite_sim(t, t, WEIGHTS_FULL)


def test_exp0_sql_injection_ordering():
    """Doc's worked example: two SQL injections similar, port scan dissimilar."""
    r = exp0_sql_injection()
    # same-attack pair must be far more similar than the cross-attack pair
    assert r["cosine_sem_Th1_Th2"] > r["cosine_sem_Th1_Th3"] + 0.3
    assert r["composite_Th1_Th2"] > r["composite_Th1_Th3"] + 0.2
    # port-scan vs SQLi should have low semantic similarity (doc reports ~0.18)
    assert r["cosine_sem_Th1_Th3"] < 0.4
