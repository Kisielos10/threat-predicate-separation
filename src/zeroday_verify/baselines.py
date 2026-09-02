"""Published anomaly detectors used as baselines, implemented faithfully.

The paper's claim is about a class of methods, so the baselines have to be other people's
methods rather than variants of our own. This module implements the character-distribution model
of Kruegel and Vigna (CCS 2003), which is the canonical anomaly detector for web requests and the
one baseline in our results that does NOT invert on legitimate-but-alarming traffic. Because it
carries part of the argument, it is implemented as described rather than approximated.

Their model, as published:

  * a request attribute's character distribution is the relative frequency of each of the 256
    byte values, sorted in descending order (the "idealized character distribution", ICD);
  * during training, the ICD of an attribute is the mean of the ICDs observed for that attribute;
  * at detection time the sorted frequencies of both the learned and the observed distribution are
    grouped into six bins over the sorted positions, {0}, {1-3}, {4-6}, {7-11}, {12-15},
    {16-255}, and a chi-square statistic is computed between them.

Two properties of the original design matter for our results and are preserved here: models are
learned **per (path, attribute)** rather than for the request as a whole, and the statistic is
computed over binned sorted frequencies rather than raw character counts.

Reference: C. Kruegel and G. Vigna, "Anomaly Detection of Web-based Attacks", Proc. 10th ACM CCS,
2003, pp. 251-261.
"""

from __future__ import annotations

from collections import defaultdict
from urllib.parse import parse_qsl

import numpy as np

from .data_text import parse_request

# Bin edges over positions in the descending-sorted frequency vector, as published.
_BINS: tuple[tuple[int, int], ...] = ((0, 1), (1, 4), (4, 7), (7, 12), (12, 16), (16, 256))


def char_distribution(value: str) -> np.ndarray:
    """Relative frequencies of the 256 byte values, sorted in descending order."""
    counts = np.zeros(256, dtype=np.float64)
    for b in value.encode("utf-8", "ignore"):
        counts[b] += 1.0
    total = counts.sum()
    if total == 0:
        return counts
    return np.sort(counts / total)[::-1]


def _binned(icd: np.ndarray) -> np.ndarray:
    return np.array([icd[a:b].sum() for a, b in _BINS], dtype=np.float64)


class KruegelVignaICD:
    """Character-distribution model, learned per (path, attribute)."""

    def __init__(self, min_observations: int = 5) -> None:
        self.min_observations = min_observations
        self.models: dict[tuple[str, str], np.ndarray] = {}
        self.global_model: np.ndarray | None = None

    @staticmethod
    def _attributes(raw: str):
        p = parse_request(raw)
        for k, v in parse_qsl(p["query"]) + parse_qsl(p["body"]):
            yield (p["path"], k), v

    def fit(self, normal_requests: list[str]) -> KruegelVignaICD:
        acc: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        every: list[np.ndarray] = []
        for raw in normal_requests:
            for key, value in self._attributes(raw):
                d = char_distribution(value)
                acc[key].append(d)
                every.append(d)
        self.models = {k: np.mean(v, axis=0) for k, v in acc.items()
                       if len(v) >= self.min_observations}
        # fallback for attributes never seen in training; the published system handles those
        # with a separate attribute-presence model, which is out of scope here
        self.global_model = np.mean(every, axis=0) if every else None
        return self

    def score_one(self, raw: str) -> float:
        """Anomaly score in [0, 1]: one minus the smallest per-attribute goodness-of-fit p-value.

        The published model reports the probability that the observed distribution was drawn from
        the learned one, so that larger values mean *more* normal. The raw chi-square statistic is
        unbounded and explodes whenever a bin has near-zero expected mass, which makes scores
        incomparable across attributes; converting through the chi-square survival function
        restores the published semantics and bounds the score. We report 1 - p so that larger
        means more anomalous, consistently with the other detectors.
        """
        from scipy.stats import chi2 as chi2_dist

        best_p = 1.0
        for key, value in self._attributes(raw):
            expected = self.models.get(key, self.global_model)
            if expected is None:
                continue
            e = _binned(expected)
            o = _binned(char_distribution(value))
            stat = float((((o - e) ** 2) / (e + 1e-9)).sum())
            p = float(chi2_dist.sf(stat, df=len(_BINS) - 1))
            best_p = min(best_p, p)
        return 1.0 - best_p

    def score(self, requests: list[str]) -> np.ndarray:
        return np.array([self.score_one(r) for r in requests], dtype=np.float64)


class PAYL:
    """Payload byte-frequency model with simplified Mahalanobis distance (Wang & Stolfo, 2004).

    PAYL models the relative frequency of each of the 256 byte values in a payload and scores a
    new payload by a simplified Mahalanobis distance, using per-byte means and standard
    deviations rather than a full covariance matrix, with a smoothing term to keep bytes that
    never varied in training from dominating. Separate models are kept per payload-length range,
    which is how the original handles the fact that short and long payloads have different
    natural distributions.

    The contrast with `KruegelVignaICD` is deliberate and matters for our analysis: PAYL scores
    the byte frequencies **in byte order**, so it is sensitive to *which* characters appear, while
    the character-distribution model sorts them first and is therefore sensitive only to how
    concentrated the distribution is, whatever the characters happen to be.

    Reference: K. Wang and S. J. Stolfo, "Anomalous Payload-Based Network Intrusion Detection",
    RAID 2004, LNCS 3224, pp. 203-222.
    """

    def __init__(self, n_buckets: int = 10, smoothing: float = 0.001) -> None:
        self.n_buckets = n_buckets
        self.smoothing = smoothing
        self.mu: dict[int, np.ndarray] = {}
        self.sd: dict[int, np.ndarray] = {}
        self.global_mu: np.ndarray | None = None
        self.global_sd: np.ndarray | None = None

    @staticmethod
    def _payload(raw: str) -> str:
        p = parse_request(raw)
        return f"{p['query']}{p['body']}"

    @staticmethod
    def _freq(payload: str) -> np.ndarray:
        counts = np.zeros(256, dtype=np.float64)
        for b in payload.encode("utf-8", "ignore"):
            counts[b] += 1.0
        total = counts.sum()
        return counts / total if total else counts

    def _bucket(self, payload: str) -> int:
        return min(len(payload) // 64, self.n_buckets - 1)

    def fit(self, normal_requests: list[str]) -> PAYL:
        acc: dict[int, list[np.ndarray]] = defaultdict(list)
        every: list[np.ndarray] = []
        for raw in normal_requests:
            pl = self._payload(raw)
            f = self._freq(pl)
            acc[self._bucket(pl)].append(f)
            every.append(f)
        for b, vs in acc.items():
            if len(vs) >= 5:
                arr = np.vstack(vs)
                self.mu[b], self.sd[b] = arr.mean(axis=0), arr.std(axis=0)
        if every:
            arr = np.vstack(every)
            self.global_mu, self.global_sd = arr.mean(axis=0), arr.std(axis=0)
        return self

    def score_one(self, raw: str) -> float:
        pl = self._payload(raw)
        b = self._bucket(pl)
        mu = self.mu.get(b, self.global_mu)
        sd = self.sd.get(b, self.global_sd)
        if mu is None:
            return 0.0
        return float((np.abs(self._freq(pl) - mu) / (sd + self.smoothing)).sum())

    def score(self, requests: list[str]) -> np.ndarray:
        return np.array([self.score_one(r) for r in requests], dtype=np.float64)
