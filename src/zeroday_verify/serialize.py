"""serialize : event -> text  (the `serialize` function of Def 7).

phi(Th) embeds a *textual* representation of the event sequence. The quality of phi depends
directly on this serialization. We produce a compact, natural-language-ish description from
the semantic attributes of the flow.

IMPORTANT: serialization must NOT leak the ground-truth label/family — only observable
attributes of the flow are included, so that any separability phi achieves is earned from
the data, not from the label.
"""

from __future__ import annotations

from .events import Event

# Map IANA protocol numbers to names for readable, semantically meaningful text.
_PROTO = {6: "TCP", 17: "UDP", 1: "ICMP", 0: "HOPOPT"}

# Well-known destination ports -> service name, to give the language model semantic hooks.
_PORTS = {
    20: "FTP-data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
    80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS", 445: "SMB", 3306: "MySQL",
    3389: "RDP", 8080: "HTTP-alt",
}


def _port_phrase(port: object) -> str:
    try:
        p = int(port)
    except (TypeError, ValueError):
        return "unknown port"
    name = _PORTS.get(p)
    return f"port {p} ({name})" if name else f"port {p}"


def _proto_phrase(proto: object) -> str:
    """Protocol name, or "" when unknown/absent (so the caller can omit the phrase)."""
    if proto is None or (isinstance(proto, str) and proto.strip().lower() in {"", "none", "nan"}):
        return ""
    try:
        p = int(float(proto))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    return _PROTO.get(p, f"protocol {p}")


def serialize(event: Event) -> str:
    """Return a short text describing the flow event for embedding by phi.

    Uses destination service, protocol, direction/volume and rate/flag descriptors derived
    from the CICFlowMeter features. Bucketised numeric phrasing keeps embeddings stable.
    """
    v = event.values
    rf = event.raw_features

    parts: list[str] = ["Network flow event."]
    proto = _proto_phrase(v.get("protocol"))
    if proto:  # omitted when the protocol field is absent (e.g. MachineLearningCVE CSVs)
        parts.append(f"Transport {proto}.")
    parts.append(f"Destination {_port_phrase(v.get('dst_port'))}.")

    dur = _as_float(rf.get("Flow Duration"))
    if dur is not None:
        parts.append(f"Duration {_bucket_duration(dur)}.")

    fwd = _as_float(rf.get("Total Fwd Packets"))
    bwd = _as_float(rf.get("Total Backward Packets"))
    if fwd is not None and bwd is not None:
        parts.append(f"{_bucket_count(fwd)} forward packets, {_bucket_count(bwd)} backward packets.")
        parts.append(f"Direction {_direction(fwd, bwd)}.")

    length = _as_float(v.get("length"))
    if length is not None:
        parts.append(f"Total payload {_bucket_bytes(length)}.")

    rate = _as_float(rf.get("Flow Bytes/s"))
    if rate is not None:
        parts.append(f"Throughput {_bucket_rate(rate)}.")

    flags = v.get("flags")
    if flags:
        parts.append(f"TCP flags observed: {flags}.")

    return " ".join(parts)


# --- helpers: coarse bucketing keeps the text (and embeddings) stable across near-duplicates ---

def _as_float(x: object) -> float | None:
    try:
        f = float(x)  # type: ignore[arg-type]
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _bucket_duration(us: float) -> str:
    ms = us / 1000.0
    if ms < 1:
        return "sub-millisecond"
    if ms < 100:
        return "very short (<100ms)"
    if ms < 1000:
        return "short (<1s)"
    if ms < 10000:
        return "moderate (1-10s)"
    return "long (>10s)"


def _bucket_count(n: float) -> str:
    if n <= 1:
        return "a single"
    if n <= 5:
        return "a few"
    if n <= 20:
        return "several"
    if n <= 100:
        return "many"
    return "a very large number of"


def _bucket_bytes(b: float) -> str:
    if b < 100:
        return "tiny (<100B)"
    if b < 1500:
        return "small (<1.5KB)"
    if b < 100_000:
        return "medium (<100KB)"
    return "large (>100KB)"


def _bucket_rate(r: float) -> str:
    if r < 1_000:
        return "low"
    if r < 100_000:
        return "moderate"
    if r < 10_000_000:
        return "high"
    return "very high"


def _direction(fwd: float, bwd: float) -> str:
    if bwd == 0 and fwd > 0:
        return "one-way outbound (no response)"
    if fwd == 0 and bwd > 0:
        return "one-way inbound"
    ratio = fwd / max(bwd, 1e-9)
    if ratio > 5:
        return "heavily outbound"
    if ratio < 0.2:
        return "heavily inbound"
    return "bidirectional"
