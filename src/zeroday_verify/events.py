"""Events — Definitions 1-4 of the contribution document.

Def 1 (event type):  tau = (name, att),  att(tau) subset of A (attribute names).
Def 2 (event):       e = (type, ts, v),  v: att(type) -> D  (attribute -> value).
Def 3 (stream):      ordered sequence by ts.
Def 4a (T_net):      att(T_net) = {src_ip, dst_ip, src_port, dst_port, protocol,
                                   length, flags, payload_meta}.

For this verification the observable unit is a *network flow* record from CIC-IDS2017
(domain T_net). Each flow is normalized to an `Event` whose `values` populate the
att(T_net) fields where available, while the full numeric CICFlowMeter feature vector is
retained separately for the raw-feature baseline (Exp B ablation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A — finite set of attribute names for the network-flow event type (Def 4a).
T_NET_ATTRS: tuple[str, ...] = (
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "length",
    "flags",
    "payload_meta",
)


@dataclass(frozen=True)
class EventType:
    """Def 1: an event type tau = (name, att)."""

    name: str
    att: tuple[str, ...]


# The single supported event type in this study (network flow, transport layer & below).
T_NET = EventType(name="NetFlow", att=T_NET_ATTRS)


@dataclass
class Event:
    """Def 2: an event e = (type, ts, v).

    `values` is the function v restricted to att(type) (human/semantic attributes used by
    serialize/phi). `raw_features` holds the full numeric flow feature vector used only by
    the raw-feature baseline. `family` / `is_attack` are ground-truth labels used for
    *evaluation only* — the method never consumes them (except A/sigma via the taxonomy,
    which encodes domain knowledge about an attack family, not the per-sample label).
    """

    event_type: EventType
    ts: float
    values: dict[str, object]
    raw_features: dict[str, float] = field(default_factory=dict)
    # evaluation labels (not used by the detection logic)
    family: str = "BENIGN"
    is_attack: bool = False

    def att(self) -> tuple[str, ...]:
        return self.event_type.att

    def get(self, attr: str, default: object = None) -> object:
        return self.values.get(attr, default)
