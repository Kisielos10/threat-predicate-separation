"""Threat object Th — Definition 6.

  Th = (E, M, K, Tg, D, A, sigma, phi, u)

  E     ordered event sequence
  M     contextual metadata (ts_start/end, IPs, ports, session) -> numeric vector m_vec for sim_M
  K     channel = (layer, protocol)
  Tg    target = (Tg_target, Tg_asset)
  D     domain in {Web, Network, Log}
  A     ATT&CK technique vector (from taxonomy)
  sigma CIA impact vector       (from taxonomy)
  phi   embedding (Def 7)        — filled in by the embedding step
  u     novelty                  — filled in at evaluation (Def 13)

For network flows each threat wraps a single event (|E| = 1). The metadata vector `m_vec`
is the numeric context used by sim_M; the raw CICFlowMeter feature vector is kept for the
raw-feature baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import Event
from .taxonomy import family_to_A, family_to_sigma

# Ordered metadata features used by sim_M (context M of the threat, Def 6).
_META_FEATURES: tuple[str, ...] = (
    "dst_port",
    "protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
)


@dataclass
class Threat:
    """A threat Th = (E, M, K, Tg, D, A, sigma, phi, u)."""

    events: list[Event]                       # E
    K: tuple[str, str]                        # channel (layer, protocol)
    Tg: tuple[str, str]                       # target (kind, asset)
    D: str                                    # domain
    A: np.ndarray                             # ATT&CK vector
    sigma: np.ndarray                         # CIA vector
    m_vec: np.ndarray                         # numeric metadata vector (for sim_M)
    raw_vec: np.ndarray                       # raw flow features (for baseline)
    text: str                                 # serialize(E) (for phi / debugging)
    phi: np.ndarray | None = None             # embedding (Def 7), filled later
    u: float | None = None                    # novelty (Def 13), filled later
    # evaluation label (not used by the method)
    family: str = "BENIGN"
    is_attack: bool = False
    meta: dict[str, object] = field(default_factory=dict)


def _meta_vector(event: Event) -> np.ndarray:
    vals = []
    for f in _META_FEATURES:
        x = event.values.get(f, event.raw_features.get(f))
        try:
            xf = float(x)  # type: ignore[arg-type]
            if xf != xf or xf in (float("inf"), float("-inf")):
                xf = 0.0
        except (TypeError, ValueError):
            xf = 0.0
        vals.append(xf)
    return np.asarray(vals, dtype=np.float64)


def build_threat_from_event(
    event: Event,
    serialize_fn,
    raw_feature_order: list[str] | None = None,
) -> Threat:
    """Construct a single-event Threat (|E| = 1) from a normalized flow event.

    A and sigma come from the event's family via the taxonomy (domain knowledge). phi and u
    are filled in later by the embedding / novelty steps.
    """
    family = event.family
    proto = str(event.values.get("protocol", ""))
    layer = "transport" if proto in {"6", "17", "1"} else "application"

    if raw_feature_order is None:
        raw_feature_order = sorted(event.raw_features.keys())
    raw_vec = np.asarray(
        [_safe_float(event.raw_features.get(k)) for k in raw_feature_order], dtype=np.float64
    )

    return Threat(
        events=[event],
        K=(layer, {"6": "TCP", "17": "UDP", "1": "ICMP"}.get(proto, proto or "?")),
        Tg=("protected_asset", str(event.values.get("dst_ip", "?"))),
        D="Network",
        A=family_to_A(family),
        sigma=family_to_sigma(family),
        m_vec=_meta_vector(event),
        raw_vec=raw_vec,
        text=serialize_fn(event),
        family=family,
        is_attack=event.is_attack,
        meta={"ts": event.ts, "dst_port": event.values.get("dst_port")},
    )


def _safe_float(x: object) -> float:
    try:
        f = float(x)  # type: ignore[arg-type]
        if f != f or f in (float("inf"), float("-inf")):
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0
