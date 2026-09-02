"""Per-family domain knowledge: ATT&CK technique vector A and CIA impact vector sigma.

These are the `A` and `sigma` components of the threat object Th (Def 6). They are assigned
per *attack family* from documented knowledge of what that family does (e.g. a SQL injection
exploits a public-facing app and primarily threatens confidentiality) — NOT from the
per-sample label.

CAVEAT (reported in results): because A and sigma are constant within a family, sim_A and
sim_sigma partly encode family identity. The Exp B ablation isolates how much real signal
they add on top of the semantic embedding sim_sem, so the contribution is not overclaimed.

CIC-IDS2017 raw label strings are normalised to families by `label_to_family`.
"""

from __future__ import annotations

import numpy as np

# Canonical, ordered list of MITRE ATT&CK techniques referenced in this study.
# The index in this tuple defines the dimension in the binary vector A in {0,1}^|TTP|.
ATTACK_TECHNIQUES: tuple[str, ...] = (
    "T1046",  # Network Service Discovery (scanning)
    "T1595",  # Active Scanning
    "T1498",  # Network Denial of Service
    "T1499",  # Endpoint Denial of Service
    "T1110",  # Brute Force
    "T1190",  # Exploit Public-Facing Application
    "T1059",  # Command and Scripting Interpreter (e.g. XSS payloads)
    "T1071",  # Application Layer Protocol (C2)
    "T1203",  # Exploitation for Client Execution
    "T1041",  # Exfiltration Over C2 Channel
    "T1212",  # Exploitation for Credential Access (Heartbleed)
)
_TECH_INDEX = {t: i for i, t in enumerate(ATTACK_TECHNIQUES)}

# family -> (set of ATT&CK techniques, CIA impact sigma = (sigma_C, sigma_I, sigma_A))
FAMILY_PROFILE: dict[str, tuple[set[str], tuple[float, float, float]]] = {
    "BENIGN":       (set(),                       (0.0, 0.0, 0.0)),
    "DoS":          ({"T1498", "T1499"},          (0.0, 0.0, 0.9)),
    "DDoS":         ({"T1498"},                   (0.0, 0.0, 1.0)),
    "PortScan":     ({"T1046", "T1595"},          (0.2, 0.0, 0.0)),
    "BruteForce":   ({"T1110"},                   (0.7, 0.2, 0.1)),
    "WebAttack":    ({"T1190", "T1059"},          (0.7, 0.3, 0.0)),
    "Bot":          ({"T1071", "T1041"},          (0.5, 0.5, 0.3)),
    "Infiltration": ({"T1190", "T1203", "T1041"}, (0.8, 0.6, 0.2)),
    "Heartbleed":   ({"T1190", "T1212"},          (0.9, 0.1, 0.0)),
}

# Raw CIC-IDS2017 label substrings -> family (case-insensitive contains match, in order).
_LABEL_RULES: tuple[tuple[str, str], ...] = (
    ("benign", "BENIGN"),
    ("ddos", "DDoS"),
    ("dos", "DoS"),
    ("portscan", "PortScan"),
    ("port scan", "PortScan"),
    ("patator", "BruteForce"),
    ("brute", "BruteForce"),
    ("web attack", "WebAttack"),
    ("sql", "WebAttack"),
    ("xss", "WebAttack"),
    ("bot", "Bot"),
    ("infiltration", "Infiltration"),
    ("heartbleed", "Heartbleed"),
)


def label_to_family(label: str) -> str:
    """Normalise a raw CIC-IDS2017 label string to a coarse attack family."""
    low = str(label).strip().lower()
    for needle, fam in _LABEL_RULES:
        if needle in low:
            return fam
    # Brute-force web families (Web Attack ... Brute Force) are caught by 'web attack'/'brute'.
    return "BENIGN" if low in {"", "nan"} else "Unknown"


def family_to_A(family: str) -> np.ndarray:
    """Binary ATT&CK technique vector A in {0,1}^|TTP| for a family (Def 6, component A)."""
    vec = np.zeros(len(ATTACK_TECHNIQUES), dtype=np.float64)
    techniques, _ = FAMILY_PROFILE.get(family, (set(), (0.0, 0.0, 0.0)))
    for t in techniques:
        vec[_TECH_INDEX[t]] = 1.0
    return vec


def family_to_sigma(family: str) -> np.ndarray:
    """CIA impact vector sigma in [0,1]^3 for a family (Def 6, component sigma)."""
    _, sigma = FAMILY_PROFILE.get(family, (set(), (0.0, 0.0, 0.0)))
    return np.asarray(sigma, dtype=np.float64)
