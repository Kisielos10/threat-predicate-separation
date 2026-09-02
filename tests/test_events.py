"""Tests for events + taxonomy (Def 1-4, family mapping, A/sigma vectors)."""

import numpy as np

from zeroday_verify.events import T_NET, Event
from zeroday_verify.taxonomy import (
    ATTACK_TECHNIQUES,
    family_to_A,
    family_to_sigma,
    label_to_family,
)


def test_label_to_family_mappings():
    cases = {
        "BENIGN": "BENIGN",
        "DDoS": "DDoS",
        "DoS Hulk": "DoS",
        "DoS GoldenEye": "DoS",
        "PortScan": "PortScan",
        "FTP-Patator": "BruteForce",
        "SSH-Patator": "BruteForce",
        "Web Attack \x96 Sql Injection": "WebAttack",
        "Web Attack \x96 XSS": "WebAttack",
        "Bot": "Bot",
        "Infiltration": "Infiltration",
        "Heartbleed": "Heartbleed",
    }
    for raw, fam in cases.items():
        assert label_to_family(raw) == fam, f"{raw!r} -> {label_to_family(raw)} != {fam}"


def test_ddos_vs_dos_disambiguation():
    # 'DDoS' must not be swallowed by the 'dos' rule
    assert label_to_family("DDoS") == "DDoS"
    assert label_to_family("DoS slowloris") == "DoS"


def test_A_and_sigma_shapes():
    A = family_to_A("WebAttack")
    sigma = family_to_sigma("WebAttack")
    assert A.shape == (len(ATTACK_TECHNIQUES),)
    assert sigma.shape == (3,)
    assert set(np.unique(A)).issubset({0.0, 1.0})
    assert np.all((0.0 <= sigma) & (sigma <= 1.0))


def test_benign_has_no_techniques():
    assert family_to_A("BENIGN").sum() == 0.0
    assert family_to_sigma("BENIGN").sum() == 0.0


def test_event_construction():
    e = Event(
        event_type=T_NET,
        ts=1.0,
        values={"dst_port": 80, "protocol": 6, "flags": "SYN"},
        raw_features={"Flow Duration": 1234.0},
        family="DoS",
        is_attack=True,
    )
    assert "dst_port" in e.att()  # att(T_net) includes dst_port
    assert e.get("dst_port") == 80
    assert e.is_attack
