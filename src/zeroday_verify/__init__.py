"""zeroday_verify — empirical verification of the threat / zero-day detection method.

Implements the formalism from the PhD contribution document (Wkład rozprawy doktorskiej):
events (Def 1-3), threat object Th (Def 6), embedding phi (Def 7), composite similarity
sim (Def 12) and novelty measure u (Def 13). The package deliberately implements only the
*detection / novelty method*, not the multi-agent system.
"""

from .events import T_NET, Event, EventType
from .novelty import KnownThreatModel, novelty_u
from .similarity import SimWeights, composite_sim
from .threats import Threat, build_threat_from_event

__all__ = [
    "Event",
    "EventType",
    "T_NET",
    "Threat",
    "build_threat_from_event",
    "SimWeights",
    "composite_sim",
    "KnownThreatModel",
    "novelty_u",
]
