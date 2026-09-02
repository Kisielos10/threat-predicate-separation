"""Figures for the multi-agent report."""

from __future__ import annotations

import json
import os

import matplotlib

from .policy import HUMAN_NAMES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {"detector": "detektor\n(bez LLM)", "single_agent": "pojedynczy\nagent",
          "mas": "system wieloagentowy\n(skrócony 1. przebieg)",
          "mas_full": "system\nwieloagentowy"}
COLORS = {"detector": "#4e79a7", "single_agent": "#e15759", "mas": "#8c8c8c",
          "mas_full": "#59a14f"}
ALL_CONFIGS = ("detector", "single_agent", "mas_full")


def _order(summary: dict) -> tuple[str, ...]:
    """Configurations actually present in the results, in a fixed display order."""
    present = set(summary.get("metrics", {}))
    return tuple(c for c in ALL_CONFIGS if c in present)


def _load(path: str = "results/agents/summary.json") -> dict:
    with open(path) as fh:
        return json.load(fh)


def quality_figure(summary: dict, path: str) -> None:
    """Detection quality per configuration, with 95% CI across independent runs."""
    across = summary.get("metrics_across_runs") or {}
    m = summary["metrics"]
    order = _order(summary)
    n_runs = summary.get("n_runs", 1)
    n_cases = m[order[0]]["n"]
    metrics = [("TPR", "TPR"), ("FPR", "FPR"), ("Precision", "precyzja"), ("F1", "F1")]
    x = np.arange(len(metrics))
    w = 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for i, cfg in enumerate(order):
        vals, errs = [], []
        for k, _ in metrics:
            if across.get(cfg, {}).get(k):
                vals.append(across[cfg][k]["mean"]); errs.append(across[cfg][k]["ci95"])
            else:
                vals.append(m[cfg][k]); errs.append(0.0)
        bars = ax.bar(x + (i - (len(order) - 1) / 2) * w, vals, w, yerr=errs, capsize=3,
                      label=LABELS[cfg].replace("\n", " "), color=COLORS[cfg], alpha=0.9)
        ax.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=3)
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 1.25); ax.set_ylabel("wartość")
    ax.set_title(f"Jakość decyzji w porównywanych konfiguracjach\n"
                 f"({n_runs} niezależne przebiegi, {n_cases} spraw na konfigurację, "
                 f"przedziały ufności 95%)")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def sendback_figure(summary: dict, path: str) -> None:
    """The W4 evidentiary gate: how often it fired and what evidence was missing."""
    by_cfg = summary.get("sendback_by_config") or {}
    # prefer the full-budget configuration; fall back to whatever is present
    primary = "mas_full" if by_cfg.get("mas_full") else "mas"
    sb = by_cfg.get(primary) or summary.get("sendback") or {}
    if not sb:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))

    # Rates, not raw counts: the configurations may rest on different numbers of cases,
    # so absolute bar heights would not be comparable.
    tot, fired = sb.get("n_mas_cases", 0), sb.get("sendback_count", 0)
    vals = [fired, tot - fired]
    ax1.bar(["odesłane\ndo uzupełnienia", "zatwierdzone\nod razu"], vals, 0.55,
            color=["#e15759", "#59a14f"], alpha=0.9)
    for xi, v in enumerate(vals):
        ax1.text(xi, v + tot * 0.015, f"{v}\n({v / max(tot, 1):.0%})", ha="center",
                 va="bottom", fontsize=10)
    ax1.set_ylim(0, tot * 1.2 if tot else 1)
    ax1.set_ylabel("liczba spraw")
    ax1.set_title("Uruchomienie mechanizmu odesłania (wkład W4)")
    ax1.spines[["top", "right"]].set_visible(False)

    gaps = sb.get("missing_evidence_counts") or {}
    if gaps:
        names = list(gaps)
        pretty = [HUMAN_NAMES.get(g, g).replace(" ", "\n", 1) for g in names]
        ax2.barh(pretty, [gaps[g] for g in names], color="#4e79a7", alpha=0.9)
        for i, g in enumerate(names):
            ax2.text(gaps[g], i, f" {gaps[g]}", va="center", fontsize=9)
        ax2.set_xlabel("liczba odesłań")
        ax2.set_title("Jakiego dowodu brakowało")
    else:
        ax2.text(0.5, 0.5, "brak odesłań", ha="center", va="center")
    ax2.spines[["top", "right"]].set_visible(False)
    flips = sb.get("decision_flipped_after_sendback", 0)
    dirs = sb.get("flip_directions") or {}
    top = max(dirs, key=dirs.get) if dirs else ""
    share = f"{dirs[top]} z {flips}" if top else ""
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.5, 0.025,
             f"Odesłanie zmieniło decyzję w {flips} przypadkach, w tym {share} w kierunku "
             f"„{top}”: mechanizm wymusza ostrożność, a nie zmianę klasyfikacji",
             ha="center", fontsize=10)
    fig.savefig(path, dpi=160); plt.close(fig)


def conditions_figure(summary: dict, path: str) -> None:
    """How often the Definition 5 conditions W1/W2/W3 were judged to hold."""
    by_cfg = summary.get("conditions_by_config") or {}
    # the full-budget configuration assesses the conditions on complete evidence
    primary = "mas_full" if by_cfg.get("mas_full") else "mas"
    cond = by_cfg.get(primary) or summary.get("conditions_W1_W2_W3") or {}
    if not cond:
        return
    # "warunek N" deliberately, not "WN": in the contribution document W1-W4 denote the
    # numbered contributions, so reusing that form for the Def 5 conditions would be ambiguous
    labels = {"W1": "warunek 1:\nobserwowalny efekt", "W2": "warunek 2:\npotencjał naruszenia CIA",
              "W3": "warunek 3:\npochodzenie pozanormatywne"}
    ws = ["W1", "W2", "W3"]
    x = np.arange(len(ws)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, (lab, colour, name) in enumerate([("attack", "#e15759", "sprawy będące atakiem"),
                                             ("benign", "#59a14f", "sprawy normalne")]):
        if lab not in cond:
            continue
        vals = [cond[lab].get(k, 0) for k in ws]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=f"{name} (n={cond[lab].get('n', 0)})",
                      color=colour, alpha=0.9)
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
    ax.set_xticks(x); ax.set_xticklabels([labels[k] for k in ws])
    ax.set_ylim(0, 1.15); ax.set_ylabel("odsetek spraw, w których warunek uznano za spełniony")
    ax.set_title("Ocena warunków predykatu zagrożenia $T(S)$ z definicji 5")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def confusion_figure(summary: dict, path: str) -> None:
    """Where each configuration's errors come from."""
    m = summary["metrics"]
    order = _order(summary)
    fig, axes = plt.subplots(1, len(order), figsize=(3.7 * len(order), 3.9))
    vmax = max(max(m[c]["TP"], m[c]["TN"]) for c in order)
    for ax, cfg in zip(axes, order, strict=True):
        d = m[cfg]
        mat = np.array([[d["TP"], d["FN"]], [d["FP"], d["TN"]]])
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=15,
                        color="white" if mat[i, j] > vmax * 0.55 else "#222")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["alarm", "brak alarmu"], fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["atak", "normalne"], fontsize=9)
        # keep the two-line form: the full-budget label does not fit on one line
        ax.set_title(LABELS[cfg], fontsize=9.5, color=COLORS[cfg])
    fig.suptitle("Macierze pomyłek: fałszywe alarmy to lewy dolny róg", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def cost_figure(summary: dict, path: str) -> None:
    """Quality against cost: the multi-agent gain is paid for in latency."""
    m = summary["metrics"]
    order = _order(summary)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    # points cluster tightly on both axes, so labels are placed alternately above and below
    # and anchored left/right to stop them colliding
    offsets = [(0, 24), (0, -34), (0, 24), (0, -34)]
    for i, cfg in enumerate(order):
        d = m[cfg]
        x = max(d["mean_seconds"], 0.05)
        ax.scatter(x, d["F1"], s=300, color=COLORS[cfg], alpha=0.9, zorder=3)
        ax.annotate(f"{LABELS[cfg].replace(chr(10), ' ')}\nF1={d['F1']:.2f}, {x:.1f} s",
                    (x, d["F1"]), textcoords="offset points", xytext=offsets[i % 4],
                    ha="center", fontsize=8.5, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.012, 600)
    ax.set_xlabel("średni czas decyzji [s, skala logarytmiczna]")
    ax.set_ylabel("F1")
    vals = [m[c]["F1"] for c in order]
    ax.set_ylim(min(vals) - 0.10, max(vals) + 0.09)
    ax.set_title("Jakość a koszt decyzji")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def calibration_figure(summary: dict, path: str) -> None:
    """Is the system less confident when it is wrong?"""
    cal = summary["calibration"]
    order = _order(summary)
    x = np.arange(len(order))
    w = 0.36
    fig, ax = plt.subplots(figsize=(2.4 * len(order), 4.8))
    right = [cal[c]["mean_conf_correct"] or 0 for c in order]
    wrong = [cal[c]["mean_conf_wrong"] or 0 for c in order]
    b1 = ax.bar(x - w / 2, right, w, label="decyzje poprawne", color="#59a14f", alpha=0.9)
    b2 = ax.bar(x + w / 2, wrong, w, label="decyzje błędne", color="#e15759", alpha=0.9)
    ax.bar_label(b1, fmt="%.2f", fontsize=9); ax.bar_label(b2, fmt="%.2f", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in order], fontsize=8.5)
    ax.set_ylim(0, 1.15); ax.set_ylabel("średnia deklarowana wiarygodność")
    ax.set_title("Kalibracja: czy system jest mniej pewny, gdy się myli?")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def architecture_figure(path: str, summary: dict | None = None) -> None:
    """Diagram of the implemented pipeline: roles, tool layer and the audit send-back."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.5, 5.4))

    def box(x, y, w, h, text, color, fontsize=10, bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor="none", alpha=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                color="white", fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, style="-|>", color="#555", ls="-", lw=1.6):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                                     linestyle=ls, lw=lw, mutation_scale=16,
                                     connectionstyle="arc3,rad=0"))

    box(0.2, 2.3, 1.5, 0.9, "Zdarzenie\n(żądanie HTTP)", "#9aa3ab", 9)
    box(2.3, 2.3, 1.5, 0.9, "Planista", "#4e79a7")
    box(4.4, 2.3, 1.6, 0.9, "Wykonawca", "#59a14f")
    box(6.7, 2.3, 1.5, 0.9, "Audytor", "#b07aa1")
    box(8.9, 2.15, 2.3, 1.2, "Decyzja (y)\nWiarygodność (c)\nRekomendacja (r)", "#e15759", 9)

    tools = ("Warstwa narzędzi (deterministyczna)\n"
             "novelty_score · payload_inspect · similar_known_threats\n"
             "graph_context · request_statistics · attack_technique_info")
    ax.add_patch(FancyBboxPatch((3.5, 0.5), 3.6, 1.0, boxstyle="round,pad=0.02",
                                facecolor="#f1f4f8", edgecolor="#c8d0d8"))
    ax.text(5.3, 1.0, tools, ha="center", va="center", fontsize=8.5, color="#333")

    for x1, x2 in ((1.7, 2.3), (3.8, 4.4), (6.0, 6.7), (8.2, 8.9)):
        arrow(x1, 2.75, x2, 2.75)
    arrow(5.2, 2.3, 5.2, 1.5, color="#59a14f")            # executor -> tools
    arrow(5.4, 1.5, 5.4, 2.3, color="#59a14f")            # tools -> executor
    arrow(7.2, 2.3, 5.6, 2.05, ls="--", color="#b07aa1")  # auditor send-back
    by_cfg = (summary or {}).get("sendback_by_config") or {}
    sb = by_cfg.get("mas_full") or by_cfg.get("mas") or (summary or {}).get("sendback") or {}
    rate = sb.get("sendback_rate")
    note = (f"odesłanie (wkład W4)\n({rate:.0%} spraw w eksperymencie)" if rate is not None
            else "odesłanie (wkład W4)")
    ax.text(6.4, 1.83, note, ha="center", fontsize=8, color="#b07aa1", style="italic")

    ax.add_patch(FancyBboxPatch((0.2, 3.6), 11.0, 0.7, boxstyle="round,pad=0.02",
                                facecolor="#fff8e1", edgecolor="#e8dcae"))
    ax.text(5.7, 3.95, "Ślad audytowy: każdy krok (myśl, wywołanie narzędzia, obserwacja, werdykt) "
                       "jest zapisywany", ha="center", fontsize=9, color="#6b5d2a")

    ax.set_xlim(0, 11.5); ax.set_ylim(0.2, 4.5); ax.axis("off")
    ax.set_title("Zaimplementowany system wieloagentowy (zredukowany zestaw trzech ról)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def make_all(summary_path: str = "results/agents/summary.json",
             out_dir: str = "results/agents/figures") -> None:
    os.makedirs(out_dir, exist_ok=True)
    s = _load(summary_path)
    architecture_figure(f"{out_dir}/architecture.png", s)
    quality_figure(s, f"{out_dir}/quality.png")
    confusion_figure(s, f"{out_dir}/confusion.png")
    cost_figure(s, f"{out_dir}/cost.png")
    calibration_figure(s, f"{out_dir}/calibration.png")
    sendback_figure(s, f"{out_dir}/sendback.png")
    conditions_figure(s, f"{out_dir}/conditions.png")
    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    make_all()
