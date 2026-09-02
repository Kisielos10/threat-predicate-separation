"""Generating legitimate-but-alarming requests at scale, from real traffic.

The hand-written probe set was the weakest part of this work: forty cases, written by one person,
and an earlier version was invalid because it used paths absent from the corpus and was therefore
separable by path alone. This module replaces hand-writing with a documented, reproducible
procedure so the probe set can be regenerated and scaled by anyone.

The procedure. Take a request that the corpus itself labels normal. Keep its method, path and
parameter names exactly as they are, so the request remains structurally indistinguishable from
ordinary traffic for that application. Substitute the value of one text-valued parameter with a
value drawn from a documented pool of content that is legitimate but superficially alarming:
surnames containing apostrophes, ordinary words that happen to be SQL keywords, accented
characters, passwords containing punctuation, e-mail addresses with plus-addressing, and error
messages that quote a user name.

Every value in the pools is legitimate input for a Spanish online shop. None is an attack: no
value closes a quoted context, comments out the remainder of a statement, or introduces markup or
a traversal sequence. This is checked in `_assert_pool_is_benign` so the property is enforced
rather than asserted in prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, quote_plus

import numpy as np

from .data_text import parse_request

# Legitimate values that look alarming. Grouped so a paper can report per-category results.
VALUE_POOLS: dict[str, list[str]] = {
    "apostrophe_name": ["O'Brien", "D'Angelo", "O'Neill", "O'Hara Ruiz", "d'Oliva",
                        "Queso D'Or", "l'Occitane", "Sant'Angelo"],
    "sql_keyword_word": ["Union Cooperativa", "Jamon Selecto", "Gran Seleccion",
                         "Table de Quesos", "Pack Insert Premium", "Grupo Select",
                         "Orden de Compra", "Valores del Norte"],
    "accented": ["Jam%C3%B3n Ib%C3%A9rico", "Queso A%C3%B1ejo", "%C3%81ngel N%C3%BA%C3%B1ez",
                 "Bego%C3%B1a Pe%C3%B1a", "Turr%C3%B3n de Alicante", "Cami%C3%B3n de reparto"],
    "punctuated_password": ["P@ssw0rd!", "aB3-x9z", "1-2-3-4", "x.yz.9", "aB%26cD", "Qu3so!2024"],
    "plus_email": ["ana%2Bpedidos@example.com", "j.perez@sub.example.co.uk",
                   "info%2Bventas@mtos.by", "s.o%27hara@peli.com"],
    "quoted_message": ["Usuario 'admin' no existe", "La cuenta 'ana' esta bloqueada",
                       "Sesion caducada, vuelva a entrar", "Contrase%C3%B1a incorrecta (1/3)"],
    "long_text": ["Lote degustacion de quesos artesanos de oveja curados en cueva "
                  "durante veinticuatro meses",
                  "Cesta de Navidad con vino, queso, aceite, jamon, turron y dulces"],
    "separator": ["queso/vino", "oveja-cabra", "1/2 litro", "norte.sur.este"],
}

# Where each category of value may legitimately appear. This mapping is the heart of the
# procedure's validity and was added after inspecting a first version's output: substituting
# freely produced requests such as `modo=Angel Nunez`, where `modo` is a control parameter whose
# legitimate values are `registro` or `entrar`, and `B1=norte.sur.este`, where `B1` is a button
# label. Those are parameter tampering, which this corpus labels as an attack, so generating them
# and calling them benign would have silently corrupted the probe set.
#
# Substitution is therefore restricted to free-text fields, where any string the user types is
# legitimate input. Control parameters (modo, B1, B2, remember), identifiers (login, dni) and
# numeric fields (id, precio, cantidad) are never touched.
CATEGORY_TARGETS: dict[str, set[str]] = {
    "apostrophe_name":     {"nombre", "apellidos"},
    "sql_keyword_word":    {"nombre", "apellidos"},
    "accented":            {"nombre", "apellidos"},
    "punctuated_password": {"password", "pwd"},
    "plus_email":          {"email"},
    "quoted_message":      {"errorMsg"},
    "long_text":           {"nombre"},
    "separator":           {"nombre", "apellidos"},
}

# An attack, for this corpus, closes a quoted context, comments out the rest of a statement,
# injects markup, or walks the filesystem. No generated value may do any of these.
_FORBIDDEN = (re.compile(r"'\s*(or|and)\s", re.I), re.compile(r"--\s*$"), re.compile(r"/\*"),
              re.compile(r"<\s*(script|img|iframe)", re.I), re.compile(r"\.\./"),
              re.compile(r"\b(union\s+select|drop\s+table|insert\s+into)\b", re.I),
              re.compile(r"[;|]\s*(cat|ls|ping|nc)\b", re.I))


def _assert_pool_is_benign() -> None:
    for cat, vals in VALUE_POOLS.items():
        for v in vals:
            for pat in _FORBIDDEN:
                if pat.search(v):
                    raise ValueError(f"pool {cat!r} contains an attack-like value: {v!r}")


_assert_pool_is_benign()


@dataclass
class Probe:
    raw_request: str
    category: str
    parameter: str
    source_path: str


def _rebuild(p: dict, params: list[tuple[str, str]]) -> str:
    qs = "&".join(f"{k}={v}" for k, v in params)
    in_body = bool(p["body"])
    url = p["path"] + ("" if in_body else (f"?{qs}" if qs else ""))
    head = (f"{p['method']} {url} HTTP/1.1\n"
            f"User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux)\n"
            f"Host: localhost:8080\n")
    if in_body:
        head += ("Content-Type: application/x-www-form-urlencoded\n"
                 f"Content-Length: {len(qs)}\n")
    return head + "\n" + (qs if in_body else "")


def generate(normal_requests: list[str], n: int, seed: int = 17) -> list[Probe]:
    """Produce `n` legitimate-but-alarming requests derived from real normal traffic."""
    rng = np.random.RandomState(seed)
    cats = list(VALUE_POOLS)
    out: list[Probe] = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        raw = normal_requests[rng.randint(len(normal_requests))]
        p = parse_request(raw)
        params = parse_qsl(p["query"]) + parse_qsl(p["body"])
        cat = cats[rng.randint(len(cats))]
        targets = CATEGORY_TARGETS[cat]
        eligible = [i for i, (k, _) in enumerate(params) if k in targets]
        if not eligible:
            continue
        i = eligible[rng.randint(len(eligible))]
        pool = VALUE_POOLS[cat]
        value = pool[rng.randint(len(pool))]
        # values in the pools are already percent-encoded where they need to be
        new = list(params)
        new[i] = (params[i][0], value if "%" in value else quote_plus(value))
        out.append(Probe(_rebuild(p, new), cat, params[i][0], p["path"]))
    if len(out) < n:
        raise RuntimeError(f"only generated {len(out)}/{n} probes")
    return out
