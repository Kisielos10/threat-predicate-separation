"""Figures for the manuscript.

Deliberately plain: greyscale-safe, no decoration, sized for a single journal column, vector PDF so
they scale in the typeset document. Values come from report/final_numbers.json.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = pathlib.Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
NUM = json.load(open("report/final_numbers.json"))

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
GREY = ["#2b2b2b", "#6e6e6e", "#a5a5a5", "#cfcfcf"]


def fig1_detectors():
    """Separation achieved by each unsupervised detector on the web corpus."""
    d = NUM["published_baselines"]
    names = {"char-distribution (Kruegel & Vigna 2003)": "Kruegel & Vigna\n(2003)",
             "PAYL byte frequency (Wang & Stolfo 2004)": "PAYL\n(2004)",
             "one-class SVM (char n-gram)": "one-class\nSVM",
             "isolation forest (char n-gram)": "isolation\nforest",
             "PCA reconstruction error": "PCA\nreconstruction"}
    items = [(names.get(k, k), v["separation"]) for k, v in d.items()]
    items.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    y = np.arange(len(items))
    ax.barh(y, [v for _, v in items], color=GREY[1], edgecolor="black", linewidth=0.5, height=0.62)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _ in items])
    ax.set_xlabel("separation (attacks $-$ legitimate anomalies)")
    ax.set_xlim(-0.5, 0.25)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5); ax.grid(axis="y", visible=False)
    for i, (_, v) in enumerate(items):
        ax.text(v + (0.015 if v >= 0 else -0.015), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5)
    fig.savefig(OUT / "fig1_detectors.pdf"); plt.close(fig)


def fig2_conditions():
    """Separation by condition and model, against the detector reference."""
    c = NUM["conditions"]
    models = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b", "claude-sonnet-5"]
    labels = ["qwen2.5\n7B", "llama3.1\n8B", "mistral\n7B", "claude\nsonnet"]
    src = {**c.get("with_novelty_local", {}), **c.get("with_novelty_sonnet", {})}
    conds = ["W1", "W2", "W3"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.set_axisbelow(True); ax.xaxis.grid(False)
    x = np.arange(len(models)); w = 0.26
    for j, cond in enumerate(conds):
        vals = [src[m]["criteria"][cond]["separation"] for m in models]
        ax.bar(x + (j - 1) * w, vals, w, label=f"$W_{cond[1]}$",
               color=GREY[j], edgecolor="black", linewidth=0.4)
    det = src[models[0]]["detector"]["separation"]
    ax.axhline(det, color="black", linestyle="--", linewidth=0.9)
    ax.text(len(models) - 0.55, det + 0.02, f"detector ({det:+.2f})", fontsize=7.5, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("separation"); ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, ncol=3, loc="upper left", handlelength=1.2)
    fig.savefig(OUT / "fig2_conditions.pdf"); plt.close(fig)


def fig3_ablation():
    """Effect on W1 of withholding the detector's score, by model."""
    c = NUM["conditions"]
    w = {**c.get("with_novelty_local", {}), **c.get("with_novelty_sonnet", {})}
    n = {**c.get("no_novelty_local", {}), **c.get("no_novelty_sonnet", {})}
    models = ["qwen2.5:7b", "mistral:7b", "llama3.1:8b", "claude-sonnet-5"]
    labels = ["qwen2.5 7B", "mistral 7B", "llama3.1 8B", "claude sonnet"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    y = np.arange(len(models))
    a = [w[m]["criteria"]["W1"]["separation"] for m in models]
    b = [n[m]["criteria"]["W1"]["separation"] for m in models]
    ax.barh(y - 0.19, a, 0.36, label="score supplied", color=GREY[2],
            edgecolor="black", linewidth=0.4)
    ax.barh(y + 0.19, b, 0.36, label="score withheld", color=GREY[0],
            edgecolor="black", linewidth=0.4)
    for i in range(len(models)):
        ax.text(max(a[i], b[i]) + 0.02, y[i], f"{b[i]-a[i]:+.2f}", va="center", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("separation on $W_1$"); ax.set_xlim(0, 1.0)
    ax.set_axisbelow(True); ax.yaxis.grid(False)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.2)
    fig.savefig(OUT / "fig3_ablation.pdf"); plt.close(fig)


def fig4_pipeline():
    """Schematic of the experimental procedure.

    Laid out so that every arrow runs left to right without crossing: the evaluation groups are
    ordered to match the vertical position of whatever produces them. The fit and calibration
    slices configure the detector rather than supplying evaluation cases, which the caption states
    instead of the figure, since routing those edges obscured the boxes they passed over.
    """
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.set_xlim(0, 118); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, text, fill="white", fs=7.4):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor="black", linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.7, color="black", shrinkA=0, shrinkB=0))

    box(0, 17, 14, 13, "HTTP corpus")
    box(19, 30, 15, 8, "normal\npartition")
    box(19, 5, 15, 8, "attack\npartition")
    arrow(14, 26, 19, 34)
    arrow(14, 21, 19, 9)

    # disjoint slices of the normal partition, top to bottom
    box(40, 37, 16, 6.5, "fit", GREY[3])
    box(40, 29, 16, 6.5, "calibration", GREY[3])
    box(40, 21, 16, 6.5, "held out", GREY[3])
    for yy in (40, 32, 24):
        arrow(34, 34.5, 40, yy)

    box(40, 10, 16, 7.5, "probe\ngeneration")
    arrow(48, 21, 48, 17.5)

    # evaluation groups, ordered to match what produces them
    box(64, 29, 18, 7.5, "ordinary\ntraffic")
    box(64, 17, 18, 7.5, "legitimate\nanomalies", GREY[2])
    box(64, 5, 18, 7.5, "attacks")
    arrow(56, 24, 64, 32)          # from the held-out slice
    arrow(56, 14, 64, 20)          # from probe generation
    arrow(34, 9, 64, 9)            # straight across from the attack partition

    box(92, 15, 22, 14, "detector and\ncondition\nassessment")
    arrow(82, 32, 92, 26)
    arrow(82, 21, 92, 22)
    arrow(82, 9, 92, 18)
    arrow(103, 15, 103, 10)
    ax.text(103, 7, "separation", ha="center", fontsize=7.6, style="italic")
    fig.savefig(OUT / "fig4_pipeline.pdf"); plt.close(fig)


for f in (fig1_detectors, fig2_conditions, fig3_ablation, fig4_pipeline):
    f(); print(f"  {f.__name__}")
print(f"wrote {len(list(OUT.glob('*.pdf')))} figures to {OUT}")
