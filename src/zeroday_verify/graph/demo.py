"""Curated campaign demo — the document's worked example as a small, clean graph.

Builds the `Th3 (recon) -> Th1 (SQLi) <-> Th2 (SQLi variant) -> Th4 (exfil)` campaign from one
actor against one asset, so a single slide-ready figure shows every relation type
(precedes, enables, escalates, variant_of, targets_same, realizes_technique, member_of,
targets, originates_from). This is the presentation hero for the campaign story.
"""

from __future__ import annotations

import numpy as np

from ..embedding import embed_texts
from ..events import T_NET, Event
from ..taxonomy import family_to_A, family_to_sigma
from ..threats import Threat
from .build import build_enriched_graph, build_threat_graph
from .schema import GraphParams


def _threat(text: str, family: str, src: str, dst: str, ts: float, port: int,
            sigma=None) -> Threat:
    ev = Event(event_type=T_NET, ts=ts, family=family, is_attack=True,
               values={"src_ip": src, "dst_ip": dst, "dst_port": port, "protocol": 6,
                       "flags": "", "length": 0, "payload_meta": 0})
    return Threat(events=[ev], K=("application", "HTTP"), Tg=("protected_asset", dst), D="Web",
                  A=family_to_A(family),
                  sigma=family_to_sigma(family) if sigma is None else np.asarray(sigma),
                  m_vec=np.zeros(3), raw_vec=np.zeros(1), text=text, family=family,
                  is_attack=True, meta={"ts": ts, "dst_port": port})


def build_curated_campaign(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                           enriched: bool = False):
    """Return the doc's worked-example campaign as a graph.

    enriched=False -> strict core (threats only, the five relations).
    enriched=True  -> overlay (also Technique/Asset/Actor/Family nodes), for the comparison figure.
    """
    src, dst = "203.0.113.45", "10.0.0.5"
    threats = [
        _threat("Sequence of TCP SYN probes scanning many destination ports on the host. "
                "Network port-scan reconnaissance.", "PortScan", src, dst, ts=0.0, port=0),
        _threat("HTTP POST request to /login with payload \"' OR '1'='1\". SQL injection against "
                "the authentication service.", "WebAttack", src, dst, ts=1500.0, port=80),
        _threat("HTTP GET request to /search with payload \"q=' UNION SELECT username,password\". "
                "SQL injection extracting database contents.", "WebAttack", src, dst, ts=3000.0,
                port=80),
        _threat("Outbound C2 connection exfiltrating database records over an application "
                "protocol channel.", "Bot", src, dst, ts=4200.0, port=443, sigma=(0.8, 0.6, 0.3)),
    ]
    emb = embed_texts([t.text for t in threats], model_name=model_name)
    for t, e in zip(threats, emb, strict=True):
        t.phi = e
    # relaxed params so the two SQLi threats cluster (Family) and the chain stays connected
    params = GraphParams(tau_var=0.75, min_cluster_size=2, precedes_window_s=86400.0)
    builder = build_enriched_graph if enriched else build_threat_graph
    return builder(threats, params=params)
