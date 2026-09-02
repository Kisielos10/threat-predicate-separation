"""Structural profile of normal traffic, used as evidence by `payload_inspect`.

Motivation. The first version of `payload_inspect` reported only matches against a list of
suspicious tokens (`' OR`, `<script`, `../`). Measured on CSIC-2010 that list is nearly silent:
it fires on 6 of 40 real attacks, because roughly 83 per cent of the corpus anomalies are not
textbook injections but structural manipulations of an otherwise ordinary request. The tool
therefore gave the agents almost no discriminating evidence, and their verdicts had to come from
the embedding-based tools, which measure "unusual" rather than "harmful".

What CSIC anomalies actually look like: a parameter renamed (`B1A` instead of `B1`, `errorMsgA`
instead of `errorMsg`), a page requested with a trailing slash, a parameter that never belongs to
that page. Those are visible only against a profile of what each page normally receives.

This module builds that profile from the reference normal traffic, which is the same disjoint
split the novelty model is fitted on. No evaluation case and no label is used. The profile yields
concrete, checkable statements ("parameter B1A does not occur on this path"), which is evidence an
agent can reason about, not a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import parse_qsl

from ..data_text import parse_request


@dataclass
class StructureProfile:
    """What each path normally looks like: which parameters and which methods it receives."""

    params: dict[str, set[str]] = field(default_factory=dict)
    methods: dict[str, set[str]] = field(default_factory=dict)

    @property
    def paths(self) -> set[str]:
        return set(self.params)

    @classmethod
    def fit(cls, raw_requests: list[str]) -> StructureProfile:
        prof = cls()
        for raw in raw_requests:
            p = parse_request(raw)
            path = p["path"]
            prof.params.setdefault(path, set())
            prof.methods.setdefault(path, set()).add(p["method"])
            for k, _ in parse_qsl(p["query"]) + parse_qsl(p["body"]):
                prof.params[path].add(k)
        return prof

    def deviations(self, raw_request: str) -> list[str]:
        """Concrete structural differences from the normal profile (empty list = none found)."""
        p = parse_request(raw_request)
        path = p["path"]
        out: list[str] = []
        if path not in self.params:
            near = self._nearest_path(path)
            hint = f" (najbliższa znana: {near})" if near else ""
            out.append(f"ścieżka '{path}' nie występuje w ruchu normalnym{hint}")
            return out
        keys = [k for k, _ in parse_qsl(p["query"]) + parse_qsl(p["body"])]
        unknown = [k for k in dict.fromkeys(keys) if k not in self.params[path]]
        if unknown:
            known = ", ".join(sorted(self.params[path])) or "(brak)"
            out.append(f"parametry {unknown} nie występują na tej ścieżce "
                       f"(znane parametry: {known})")
        if p["method"] not in self.methods.get(path, set()):
            out.append(f"metoda {p['method']} nie występuje na tej ścieżce "
                       f"(znane: {', '.join(sorted(self.methods[path]))})")
        return out

    def _nearest_path(self, path: str) -> str | None:
        """A known path that differs only by a trailing slash or a suffix, if one exists."""
        stripped = path.rstrip("/")
        for cand in (stripped, stripped + "/"):
            if cand in self.params and cand != path:
                return cand
        for known in self.params:
            if known.rstrip("/") == stripped and known != path:
                return known
        return None
