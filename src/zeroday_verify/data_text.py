"""Textual HTTP dataset pipeline (CSIC-2010) — domain D = Web.

Tests the hypothesis that the embedding phi separates web-layer attacks far better when it
can see the *payload* (the actual HTTP request text) than when it only sees flow statistics
(CIC-IDS2017), where WebAttack novelty failed (AUROC ~0.35).

CSIC-2010 is labelled normal vs anomalous (binary). Anomalous requests are further bucketed
into coarse attack types by payload pattern matching (`attack_type`) so a leave-one-type-out
zero-day test is possible in the web domain. Those derived types are HEURISTIC (clearly
flagged in the report), unlike CIC-IDS2017's curated family labels.
"""

from __future__ import annotations

from urllib.parse import unquote_plus

import numpy as np
import pandas as pd

from .embedding import embed_texts
from .threats import Threat

# coarse web attack taxonomy (ATT&CK technique + CIA sigma) for the derived types
_TEXT_PROFILE = {
    "sqli":              ({"T1190"}, (0.7, 0.3, 0.0)),
    "xss":               ({"T1059"}, (0.5, 0.4, 0.0)),
    "path_traversal":    ({"T1083"}, (0.6, 0.0, 0.0)),
    "command_injection": ({"T1059"}, (0.7, 0.5, 0.3)),
    "other_anomaly":     (set(),     (0.4, 0.3, 0.1)),
}


def load_csic(path: str = "data/raw_text/csic_all.parquet",
              n_normal: int = 6000, n_anomalous: int = 6000, seed: int = 17) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rng = np.random.RandomState(seed)
    normal = df[df["label"] == 0]
    anom = df[df["label"] == 1]
    normal = normal.iloc[rng.choice(len(normal), min(n_normal, len(normal)), replace=False)]
    anom = anom.iloc[rng.choice(len(anom), min(n_anomalous, len(anom)), replace=False)]
    out = pd.concat([normal, anom], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def parse_request(req: str) -> dict:
    """Split a raw HTTP request into method, path, query and body."""
    text = str(req).replace("\\n", "\n")
    lines = text.split("\n")
    first = lines[0].split(" ") if lines else []
    method = first[0] if first else "?"
    url = first[1] if len(first) > 1 else ""
    path, _, query = url.partition("?")
    body = ""
    if "" in lines:
        body = "\n".join(lines[lines.index("") + 1:]).strip()
    elif len(lines) > 1:
        body = lines[-1].strip() if "=" in lines[-1] else ""
    return {"method": method, "path": path, "query": query, "body": body}


def serialize_http(parsed: dict) -> str:
    """Payload-focused text for phi: method, path, and the DECODED parameters/body.

    Headers (mostly constant boilerplate in CSIC) are dropped so the embedding focuses on the
    discriminative request line + payload. Percent-encoding is decoded so injections read as
    natural text (e.g. %27 -> ').
    """
    params = unquote_plus(parsed["query"]) if parsed["query"] else ""
    body = unquote_plus(parsed["body"]) if parsed["body"] else ""
    parts = [f"HTTP {parsed['method']} request to {parsed['path']}."]
    if params:
        parts.append(f"Query parameters: {params}.")
    if body:
        parts.append(f"Body: {body}.")
    return " ".join(parts)


_SIG = {
    "sqli": ("'", " or ", " or %", "union", "select", "--", "1=1", "drop ", "insert ",
             "%27", "0x", "/*", "*/", " and "),
    "xss": ("<script", "javascript:", "onerror", "onload", "alert(", "<img", "%3cscript",
            "<iframe", "document.cookie"),
    "path_traversal": ("../", "..\\", "/etc/passwd", "%2e%2e", "boot.ini", "/etc/", "..%2f",
                       "passwd", "windows/system32"),
    "command_injection": (";cat", ";ls", "|cat", "&&", "$(", "/bin/", "cmd.exe", "system(",
                          "`", "ping ", "nc -"),
}


def attack_type(parsed: dict) -> str:
    """Heuristic web-attack bucket from the decoded payload."""
    blob = (unquote_plus(parsed["query"]) + " " + unquote_plus(parsed["body"]) + " "
            + parsed["path"]).lower()
    for atype, sigs in _SIG.items():
        if any(s in blob for s in sigs):
            return atype
    return "other_anomaly"


def build_text_threats(df: pd.DataFrame,
                       model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> list[Threat]:
    """Parse -> serialize -> embed -> Threat (Web domain). family = 'normal' or attack type."""
    from .taxonomy import ATTACK_TECHNIQUES

    tech_idx = {t: i for i, t in enumerate(ATTACK_TECHNIQUES)}
    parsed = [parse_request(r) for r in df["requests"]]
    texts = [serialize_http(p) for p in parsed]
    labels = df["label"].to_numpy()

    threats: list[Threat] = []
    for p, txt, lab in zip(parsed, texts, labels, strict=True):
        is_attack = bool(lab == 1)
        if is_attack:
            fam = attack_type(p)
            techs, sigma = _TEXT_PROFILE[fam]
        else:
            fam, techs, sigma = "normal", set(), (0.0, 0.0, 0.0)
        A = np.zeros(len(ATTACK_TECHNIQUES))
        for t in techs:
            if t in tech_idx:
                A[tech_idx[t]] = 1.0
        # small observable metadata vector: request length, #params, #special chars
        blob = (p["query"] + p["body"])
        m = np.array([len(txt), blob.count("="), sum(c in "'\"<>;|&%/" for c in blob)], float)
        threats.append(Threat(
            events=[], K=("application", "HTTP"), Tg=("protected_asset", p["path"]),
            D="Web", A=A, sigma=np.asarray(sigma), m_vec=m, raw_vec=m.copy(), text=txt,
            family=fam, is_attack=is_attack,
        ))

    emb = embed_texts([t.text for t in threats], model_name=model_name)
    for t, e in zip(threats, emb, strict=True):
        t.phi = e
    _standardise_m(threats)
    return threats


def _standardise_m(threats: list[Threat]) -> None:
    M = np.vstack([t.m_vec for t in threats])
    mu, sd = M.mean(0), M.std(0)
    sd[sd == 0] = 1.0
    for t in threats:
        t.m_vec = (t.m_vec - mu) / sd
