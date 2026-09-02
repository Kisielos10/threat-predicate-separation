"""Data pipeline for CIC-IDS2017 flow CSVs.

Loads the labelled flow CSVs, normalises each flow into an `Event` (Def 2, domain T_net),
groups raw labels into attack families (taxonomy.label_to_family), subsamples a balanced
study set, builds `Threat` objects (Def 6), computes embeddings phi (Def 7) in batch, and
standardises the metadata vectors used by sim_M.

Robust to both common CIC-IDS2017 distributions:
  * MachineLearningCVE (numeric features + ' Destination Port', ' Label')
  * GeneratedLabelledFlows / TrafficLabelling (adds Source/Destination IP, Protocol, Timestamp)
Only columns that are present are used.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from .embedding import aggregate, embed_texts
from .events import T_NET, Event
from .serialize import serialize
from .taxonomy import label_to_family
from .threats import Threat, build_threat_from_event

_FLAG_COLS = {
    "FIN Flag Count": "FIN",
    "SYN Flag Count": "SYN",
    "RST Flag Count": "RST",
    "PSH Flag Count": "PSH",
    "ACK Flag Count": "ACK",
    "URG Flag Count": "URG",
    "ECE Flag Count": "ECE",
    "CWE Flag Count": "CWE",
}
# Columns that are identifiers / labels, excluded from the raw-feature baseline vector.
_NON_FEATURE = {
    "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP", "Timestamp",
    "Label", "family", "is_attack", "Source Port", "Src Port",
}


def discover_csvs(raw_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True))


def load_raw(paths: list[str]) -> pd.DataFrame:
    """Load + concatenate CSVs, strip column names, drop inf/NaN rows, add family labels."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False, encoding="latin-1")
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    label_col = _find_col(data, ["Label"])
    if label_col is None:
        raise ValueError("no 'Label' column found in CIC-IDS2017 CSVs")
    data = data.rename(columns={label_col: "Label"})

    data = data.replace([np.inf, -np.inf], np.nan)
    # numeric feature columns
    feat_cols = [c for c in data.columns if c not in _NON_FEATURE]
    for c in feat_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=[c for c in feat_cols if data[c].dtype != object])

    data["family"] = data["Label"].map(label_to_family)
    data["is_attack"] = data["family"] != "BENIGN"
    data = data[data["family"] != "Unknown"]
    return data.reset_index(drop=True)


def load_subsampled(
    paths: list[str],
    per_attack_family: int = 700,
    n_benign: int = 3500,
    seed: int = 17,
    min_family_size: int = 50,
) -> pd.DataFrame:
    """Memory-efficient load: subsample within each file before concatenating.

    Keeps all attack rows per file (capped per family) and a per-file benign sample, so peak
    memory is one day-file at a time rather than the full ~2.8M-row dataset. Families with
    fewer than `min_family_size` total flows (e.g. Infiltration, Heartbleed) are dropped as
    too rare for a fair leave-one-family-out fold.
    """
    rng = np.random.RandomState(seed)
    benign_per_file = int(np.ceil(n_benign / max(len(paths), 1)))
    attack_parts: list[pd.DataFrame] = []
    benign_parts: list[pd.DataFrame] = []

    for p in paths:
        df = pd.read_csv(p, low_memory=False, encoding="latin-1")
        df.columns = [str(c).strip() for c in df.columns]
        label_col = _find_col(df, ["Label"])
        df = df.rename(columns={label_col: "Label"})
        df = df.replace([np.inf, -np.inf], np.nan)
        df["family"] = df["Label"].map(label_to_family)
        df["is_attack"] = df["family"] != "BENIGN"

        atk = df[df["is_attack"] & (df["family"] != "Unknown")]
        if len(atk):
            attack_parts.append(atk)
        ben = df[df["family"] == "BENIGN"]
        if len(ben):
            take = min(benign_per_file, len(ben))
            benign_parts.append(ben.iloc[rng.choice(len(ben), size=take, replace=False)])
        del df

    attacks = pd.concat(attack_parts, ignore_index=True) if attack_parts else pd.DataFrame()
    # drop too-rare families, then cap each family
    capped = []
    for _fam, grp in attacks.groupby("family"):
        if len(grp) < min_family_size:
            continue
        n = min(per_attack_family, len(grp))
        capped.append(grp.iloc[rng.choice(len(grp), size=n, replace=False)])
    benign = pd.concat(benign_parts, ignore_index=True)
    if len(benign) > n_benign:
        benign = benign.iloc[rng.choice(len(benign), size=n_benign, replace=False)]

    data = pd.concat(capped + [benign], ignore_index=True)
    # coerce features to numeric and drop unusable rows
    feat_cols = [c for c in data.columns if c not in _NON_FEATURE]
    for c in feat_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=[c for c in feat_cols if data[c].dtype != object])
    return data.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def subsample(
    data: pd.DataFrame,
    per_attack_family: int = 800,
    n_benign: int = 3000,
    seed: int = 17,
    min_family_size: int = 50,
) -> pd.DataFrame:
    """Balanced subsample: up to `per_attack_family` per attack family + `n_benign` benign.

    Families with fewer than `min_family_size` samples are dropped (too few for a fair
    leave-one-family-out fold).
    """
    rng = np.random.RandomState(seed)
    parts = []
    for fam, grp in data.groupby("family"):
        if fam == "BENIGN":
            n = min(n_benign, len(grp))
        else:
            if len(grp) < min_family_size:
                continue
            n = min(per_attack_family, len(grp))
        idx = rng.choice(grp.index.values, size=n, replace=False)
        parts.append(data.loc[idx])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _flags_string(row: pd.Series) -> str:
    present = []
    for col, name in _FLAG_COLS.items():
        if col in row.index:
            try:
                if float(row[col]) > 0:
                    present.append(name)
            except (TypeError, ValueError):
                pass
    return ",".join(present)


def _length(row: pd.Series) -> float:
    for a, b in [("Total Length of Fwd Packets", "Total Length of Bwd Packets"),
                 ("Subflow Fwd Bytes", "Subflow Bwd Bytes")]:
        if a in row.index and b in row.index:
            return _f(row.get(a)) + _f(row.get(b))
    return _f(row.get("Flow Bytes/s"))


def _f(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        return 0.0 if (v != v or v in (float("inf"), float("-inf"))) else v
    except (TypeError, ValueError):
        return 0.0


def row_to_event(row: pd.Series, ts: float, raw_feature_cols: list[str]) -> Event:
    dst_port = row.get(_first_present(row, ["Destination Port", "Dst Port"]))
    src_port = row.get(_first_present(row, ["Source Port", "Src Port"]))
    proto = row.get(_first_present(row, ["Protocol"]))
    values = {
        "src_ip": row.get(_first_present(row, ["Source IP", "Src IP"])),
        "dst_ip": row.get(_first_present(row, ["Destination IP", "Dst IP"])),
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": proto,
        "length": _length(row),
        "flags": _flags_string(row),
        "payload_meta": _f(row.get("Average Packet Size")),
    }
    raw_features = {c: _f(row.get(c)) for c in raw_feature_cols}
    return Event(
        event_type=T_NET,
        ts=ts,
        values=values,
        raw_features=raw_features,
        family=str(row["family"]),
        is_attack=bool(row["is_attack"]),
    )


def _first_present(row: pd.Series, names: list[str]) -> str:
    for n in names:
        if n in row.index:
            return n
    return names[0]


def build_threats(
    data: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    agg: str = "mean",
) -> list[Threat]:
    """Normalise rows -> events -> threats, embed phi in batch, standardise sim_M vectors."""
    raw_feature_cols = [
        c for c in data.columns
        if c not in _NON_FEATURE and pd.api.types.is_numeric_dtype(data[c])
    ]
    ts_vals = _timestamps(data)
    events = [row_to_event(row, ts=ts_vals[k], raw_feature_cols=raw_feature_cols)
              for k, (_, row) in enumerate(data.iterrows())]
    threats = [build_threat_from_event(e, serialize, raw_feature_order=raw_feature_cols)
               for e in events]

    # phi (Def 7): embed all serialized texts in one batched pass
    texts = [t.text for t in threats]
    emb = embed_texts(texts, model_name=model_name)
    for t, vec in zip(threats, emb, strict=True):
        t.phi = aggregate(vec, method=agg)

    _standardise_m_vec(threats)
    return threats


def _timestamps(data: pd.DataFrame) -> list[float]:
    """Epoch seconds from a 'Timestamp' column when present (for precedes/escalates edges),
    falling back to the row index where the column is absent or unparseable."""
    if "Timestamp" not in data.columns:
        return [float(i) for i in range(len(data))]
    parsed = pd.to_datetime(data["Timestamp"], errors="coerce")
    epoch = parsed.astype("int64", copy=False).to_numpy() / 1e9  # NaT -> large negative sentinel
    out = []
    for i, e in enumerate(epoch):
        out.append(float(e) if (parsed.iloc[i] is not pd.NaT and e > 0) else float(i))
    return out


def _standardise_m_vec(threats: list[Threat]) -> None:
    M = np.vstack([t.m_vec for t in threats])
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd[sd == 0] = 1.0
    for t in threats:
        t.m_vec = (t.m_vec - mu) / sd
