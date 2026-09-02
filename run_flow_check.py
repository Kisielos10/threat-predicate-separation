"""Check 2: does the specificity failure appear in the network-flow domain as well?

The HTTP result rests on hand-written probes, which is its weakest point. Here nothing is
synthesised. The probes are flows that CIC-IDS2017 itself labels BENIGN and that are
statistically extreme on raw features: very long, very large, very many packets, or directed at
a rare port. They are real traffic, someone else labelled them, and they are unusual by
construction.

Two rules matter for this not to be circular:

  * probes are selected on RAW univariate features, never on the detector's own score. Selecting
    by "distance from normal" and then showing that distance is high would prove nothing;
  * the normality model is fitted on a disjoint slice of benign traffic, and the decision
    threshold is calibrated on a third slice with labels, so the detector gets its best case.

Two detectors are scored: a centroid-distance model matching the u = 1 - max_k sim(.,mu_k) form
used elsewhere in this project, and Isolation Forest, which is the standard unsupervised baseline
in the intrusion-detection literature.

    ./.venv/bin/python run_flow_check.py [n_probe] [n_attack] [n_normal]
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

OUT = "results/conditions"
# Raw features used ONLY to select unusual-but-benign flows. Deliberately simple and univariate.
EXTREME_FEATURES = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                    "Total Length of Fwd Packets", "Flow Bytes/s", "Flow Packets/s"]


def load_flows(max_rows_per_file: int = 120_000) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob("data/raw_full/TrafficLabelling */*.csv")):
        df = pd.read_csv(f, nrows=max_rows_per_file, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Label"] = df["Label"].astype(str).str.strip()
    return df


def numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    return x


def select_extreme(benign: pd.DataFrame, n: int, seed: int = 17) -> pd.DataFrame:
    """Benign flows that are extreme on at least one raw feature, or on a rare destination port.

    Selection never touches the detector's score, so this is not circular.
    """
    rng = np.random.RandomState(seed)
    mask = pd.Series(False, index=benign.index)
    for col in EXTREME_FEATURES:
        if col not in benign.columns:
            continue
        v = pd.to_numeric(benign[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if v.notna().sum() == 0:
            continue
        mask |= v >= v.quantile(0.999)
    if "Destination Port" in benign.columns:
        port = pd.to_numeric(benign["Destination Port"], errors="coerce")
        freq = port.map(port.value_counts())
        mask |= freq <= 3                      # a port seen at most 3 times in benign traffic
    pool = benign[mask]
    if len(pool) > n:
        pool = pool.iloc[rng.choice(len(pool), n, replace=False)]
    return pool


def main(n_probe: int = 400, n_attack: int = 400, n_normal: int = 400) -> None:
    os.makedirs(OUT, exist_ok=True)
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    print("loading flows ...", flush=True)
    df = load_flows()
    feat = [c for c in df.columns if c not in
            ("Flow ID", "Source IP", "Destination IP", "Timestamp", "Label",
             "Source Port", "Destination Port", "Protocol")]
    benign_all = df[df["Label"] == "BENIGN"]
    attacks_all = df[df["Label"] != "BENIGN"]
    print(f"benign {len(benign_all)}, attacks {len(attacks_all)}, {len(feat)} numeric features")

    rng = np.random.RandomState(17)
    idx = rng.permutation(len(benign_all))
    fit_b = benign_all.iloc[idx[:20_000]]                    # fit the normality model
    cal_b = benign_all.iloc[idx[20_000:23_000]]              # calibrate the threshold
    held_b = benign_all.iloc[idx[23_000:]]                   # probes + ordinary controls come here

    probes = select_extreme(held_b, n_probe)
    rest = held_b.drop(index=probes.index)
    normal = rest.iloc[rng.choice(len(rest), min(n_normal, len(rest)), replace=False)]
    atk = attacks_all.iloc[rng.choice(len(attacks_all), n_attack, replace=False)]
    cal_a = attacks_all.iloc[rng.choice(len(attacks_all), 3_000, replace=False)]
    print(f"probes (benign, extreme) {len(probes)}, ordinary benign {len(normal)}, "
          f"attacks {len(atk)}")

    scaler = StandardScaler().fit(numeric_matrix(fit_b, feat))
    X = {k: scaler.transform(numeric_matrix(v, feat))
         for k, v in (("fit", fit_b), ("cal_b", cal_b), ("cal_a", cal_a),
                      ("probe", probes), ("normal", normal), ("attack", atk))}

    results = {}

    # --- detector 1: centroid distance, the u = 1 - max_k sim(.,mu_k) form ---
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=8, n_init=10, random_state=17).fit(X["fit"])
    def u_score(a):                                                          # noqa: E306
        d = np.linalg.norm(a[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
        return d.min(axis=1)
    # --- detector 2: Isolation Forest, the standard unsupervised baseline ---
    iso = IsolationForest(n_estimators=200, random_state=17).fit(X["fit"])
    def iso_score(a):                                                        # noqa: E306
        return -iso.score_samples(a)

    for name, fn in (("centroid distance", u_score), ("isolation forest", iso_score)):
        from sklearn.metrics import roc_curve
        s_cb, s_ca = fn(X["cal_b"]), fn(X["cal_a"])
        fpr, tpr, thr = roc_curve([0] * len(s_cb) + [1] * len(s_ca),
                                  np.concatenate([s_cb, s_ca]))
        t = float(thr[int(np.argmax(tpr - fpr))])
        row = {}
        for grp in ("attack", "probe", "normal"):
            s = fn(X[grp])
            row[grp] = {"median": round(float(np.median(s)), 4),
                        "alert_rate": round(float((s >= t).mean()), 3)}
        row["threshold"] = round(t, 4)
        row["separation"] = round(row["attack"]["alert_rate"] - row["probe"]["alert_rate"], 3)
        results[name] = row

    print("\n=== FLOW DOMAIN: alert rate by group ===")
    print(f"{'detector':20s} {'attacks':>9s} {'benign-extreme':>16s} {'benign-ordinary':>17s} "
          f"{'separation':>11s}")
    for name, r in results.items():
        print(f"{name:20s} {r['attack']['alert_rate']:9.2f} {r['probe']['alert_rate']:16.2f} "
              f"{r['normal']['alert_rate']:17.2f} {r['separation']:11.2f}")
    print("\n=== median score (is the ordering inverted?) ===")
    for name, r in results.items():
        inv = "INVERTED" if r["probe"]["median"] > r["attack"]["median"] else "ordered"
        print(f"{name:20s} attacks {r['attack']['median']:9.4f}   "
              f"benign-extreme {r['probe']['median']:9.4f}   -> {inv}")

    with open(f"{OUT}/flow_check.json", "w") as fh:
        json.dump({"n": {"probe": len(probes), "attack": len(atk), "normal": len(normal)},
                   "results": results}, fh, indent=2)
    print(f"\nwrote {OUT}/flow_check.json")


if __name__ == "__main__":
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    c = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    main(a, b, c)
