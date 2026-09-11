r"""Figures for the manuscript.

Sized so that the manuscript's ``\includegraphics[width=\linewidth]`` places each PDF at 1:1:
the page of every figure is exactly the text width of the typeset article (466.26 pt for
elsarticle with the ``preprint`` option), so a 9 pt label in a figure prints as 9 pt next to the
10 pt body text.  That requires a fixed page (``bbox_inches=None``) rather than a tight bounding
box, which is why the axes figures use constrained layout: it fits the labels inside the fixed
page instead of growing the page around them.

Colour is used where it carries meaning -- provenance in the pipeline diagram, sign in Fig. 1,
condition in Fig. 2, ablation arm in Fig. 3 -- and every palette is a lightness ramp, so the
figures survive greyscale printing.  Serif throughout, vector PDF, TrueType-embedded (no Type 3).

Every number is read from report/final_numbers.json; nothing numeric is written here.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

OUT = pathlib.Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
NUM = json.load(open("report/final_numbers.json"))

# Text width of the typeset manuscript, in inches.  Figures are authored at this width so that
# width=\linewidth is a 1:1 placement and font sizes are the ones set below.
FIG_W = 466.26 / 72.0

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "axes.axisbelow": True, "axes.edgecolor": "#333333", "axes.labelcolor": "#000000",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.color": "#333333",
    "ytick.color": "#333333", "xtick.labelcolor": "#000000", "ytick.labelcolor": "#000000",
    "axes.grid": True, "grid.alpha": 1.0, "grid.color": "#d7d7d7", "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "standard", "savefig.pad_inches": 0.0,
    "pdf.fonttype": 42, "pdf.compression": 6,
})

INK = "#333333"                              # rules, arrows, bar outlines
COOL, WARN = "#5b8db8", "#b05c48"            # sign of the separation in Fig. 1
SEQ = ["#c6dbef", "#6baed6", "#21618c"]      # lightness ramp for W1/W2/W3
REF = "#8c2d19"                              # detector reference line
# Provenance palette for the pipeline, on an even lightness ladder so it reads in greyscale too.
FILL = {"normal": "#dce9f7", "probe": "#eec27f", "attack": "#d5978a", "plain": "#ffffff"}
EDGE = {"normal": "#3f6f9f", "probe": "#a9761c", "attack": "#94493a", "plain": INK}


def _drawn_text(fig):
    """Every text artist that actually reaches the page (stale ticks outside the view do not)."""
    for ax in fig.axes:
        yield from (ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts)
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs(), strict=False):
                if lo <= loc <= hi:
                    yield from (tick.label1, tick.label2)
        if ax.get_legend() is not None:
            yield from ax.get_legend().texts
    yield from fig.texts
    for legend in fig.legends:
        yield from legend.texts


def _save(fig, name):
    """Write a fixed-size vector PDF, after checking that no text left the page."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    page = fig.bbox
    spill = []
    for t in _drawn_text(fig):
        if not isinstance(t, Text) or not t.get_visible() or not t.get_text().strip():
            continue
        b = t.get_window_extent(r)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > page.x1 + 1 or b.y1 > page.y1 + 1:
            spill.append(t.get_text().replace("\n", " ")[:40])
    if spill:
        raise SystemExit(f"{name}: text outside the page: {spill}")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches=None)
    plt.close(fig)


def fig1_detectors():
    """Separation achieved by each unsupervised detector on the web corpus."""
    d = NUM["published_baselines"]
    names = {"char-distribution (Kruegel & Vigna 2003)": "Kruegel & Vigna (2003)",
             "PAYL byte frequency (Wang & Stolfo 2004)": "PAYL (Wang & Stolfo, 2004)",
             "one-class SVM (char n-gram)": "one-class SVM",
             "isolation forest (char n-gram)": "isolation forest",
             "PCA reconstruction error": "PCA reconstruction error"}
    items = [(names.get(k, k), v["separation"]) for k, v in d.items()]
    # Only the five detectors of the scaled experiment belong here. The reference detector was
    # scored on the condition experiment's sample, and plotting it alongside would put two
    # samples on one axis as though they were one.
    items.sort(key=lambda x: x[1])
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(FIG_W, 2.45), layout="constrained")
    y = np.arange(len(items))
    ax.barh(y, vals, color=[WARN if v < 0 else COOL for v in vals],
            edgecolor=INK, linewidth=0.6, height=0.66, zorder=2)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _ in items])
    ax.set_xlabel("separation (attack rate $-$ legitimate-anomaly rate)")
    span = max(vals + [0.0]) - min(vals + [0.0])
    ax.set_xlim(min(vals + [0.0]) - 0.19 * span, max(vals + [0.0]) + 0.19 * span)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    ax.tick_params(axis="y", length=0)
    for i, v in enumerate(vals):
        off = 0.018 * span
        lab = "0.00" if abs(v) < 5e-3 else f"{v:+.2f}"
        ax.text(v + (off if v >= 0 else -off), i, lab, va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, zorder=4)
    _save(fig, "fig1_detectors")


def fig2_conditions():
    """Separation by condition and model, against the detector reference."""
    c = NUM["conditions"]
    # The three local models only: they are the ones re-run on identical cases after the case-ID
    # fix. The hosted model's run with the score predates the fix and is not used anywhere.
    models = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]
    labels = ["qwen2.5 7B", "llama3.1 8B", "mistral 7B"]
    src = c["with_novelty_local"]
    conds = ["W1", "W2", "W3"]

    fig, ax = plt.subplots(figsize=(FIG_W, 2.85), layout="constrained")
    ax.xaxis.grid(False)
    x = np.arange(len(models)); w = 0.24
    top = 0.0
    for j, cond in enumerate(conds):
        vals = [src[m]["criteria"][cond]["separation"] for m in models]
        top = max(top, max(vals))
        ax.bar(x + (j - 1) * w, vals, w, label=f"$W_{cond[1]}$",
               color=SEQ[j], edgecolor=INK, linewidth=0.5, zorder=2)
    det = src[models[0]]["detector"]["separation"]
    dash = (0, (5, 2.5))
    ax.axhline(det, color=REF, linestyle=dash, linewidth=1.3, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("separation"); ax.set_ylim(0, top * 1.12)
    ax.set_xlim(-0.5, len(models) - 0.5)
    handles, texts = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=REF, linestyle=dash, linewidth=1.3))
    texts.append(f"detector reference ({det:+.2f})")
    ax.legend(handles, texts, frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handlelength=1.7, columnspacing=1.8,
              handletextpad=0.6, borderpad=0.0)
    _save(fig, "fig2_conditions")


def fig3_ablation():
    """Effect on W1 of withholding the detector's score, by model."""
    c = NUM["conditions"]
    # An ablation needs both arms on the same cases; only the local models have that.
    w, n = c["with_novelty_local"], c["no_novelty_local"]
    models = ["qwen2.5:7b", "mistral:7b", "llama3.1:8b"]
    labels = ["qwen2.5 7B", "mistral 7B", "llama3.1 8B"]

    fig, ax = plt.subplots(figsize=(FIG_W, 2.45), layout="constrained")
    y = np.arange(len(models))
    a = [w[m]["criteria"]["W1"]["separation"] for m in models]
    b = [n[m]["criteria"]["W1"]["separation"] for m in models]
    ax.barh(y - 0.19, a, 0.36, label="score supplied", color=SEQ[0],
            edgecolor=INK, linewidth=0.5, zorder=2)
    ax.barh(y + 0.19, b, 0.36, label="score withheld", color=SEQ[2],
            edgecolor=INK, linewidth=0.5, zorder=2)
    ax.set_xlim(0, max(a + b) * 1.32)
    for i in range(len(models)):
        ax.text(max(a[i], b[i]) + 0.018 * max(a + b), y[i], f"{b[i] - a[i]:+.2f}",
                va="center", fontsize=8.5, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlabel("separation on $W_1$")
    ax.yaxis.grid(False); ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.7, columnspacing=1.8, handletextpad=0.6, borderpad=0.0)
    _save(fig, "fig3_ablation")


def fig4_pipeline():
    r"""Schematic of the experimental procedure.

    Laid out on a grid: five columns whose widths are measured from the rendered text (so a box
    can never be too small for its label) and separated by equal gaps, and rows chosen so that
    every stage sits at the height of whatever produces it.  Connectors are therefore orthogonal
    -- a straight horizontal run, or a horizontal run into a vertical bus and out again -- and
    never cross a box.  Fill colour marks provenance, which is what the figure exists to show:
    the three evaluation groups differ only in where their traffic came from.  The fit and
    calibration slices configure the detector rather than supplying evaluation cases, which the
    caption states instead of the figure, since routing those edges obscured the boxes they
    passed over.
    """
    fs, fig_h, margin = 8.5, 3.05, 8.0
    pad_x, pad_y = 7.0, 5.0                 # breathing space between a label and its border
    fig = plt.figure(figsize=(FIG_W, fig_h))
    ax = fig.add_axes((0, 0, 1, 1))
    W, H = FIG_W * 72.0, fig_h * 72.0
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # key -> (column, row centre in points, label, palette key)
    spec = {
        "corpus":   (0, 113, "HTTP corpus", "plain"),
        "normal":   (1, 170, "normal\npartition", "normal"),
        "attackp":  (1, 56, "attack\npartition", "attack"),
        "fit":      (2, 198, "fit", "normal"),
        "calib":    (2, 170, "calibration", "normal"),
        "held":     (2, 142, "held out", "normal"),
        "probe":    (2, 100, "probe\ngeneration", "probe"),
        "ordinary": (3, 142, "ordinary\ntraffic", "normal"),
        "legit":    (3, 100, "legitimate\nanomalies", "probe"),
        "attacks":  (3, 56, "attacks", "attack"),
        "assess":   (4, 100, "detector and\ncondition\nassessment", "plain"),
    }
    txt = {k: ax.text(0, row, label, ha="center", va="center", fontsize=fs,
                      linespacing=1.35, zorder=3)
           for k, (_, row, label, _) in spec.items()}
    fig.canvas.draw()
    inv = ax.transData.inverted()
    extent = {k: t.get_window_extent().transformed(inv) for k, t in txt.items()}

    # One width per column, taken from its widest label, and equal gaps between columns: the
    # boxes then line up on a grid and every connector between two columns is the same length.
    ncol = max(c for c, _, _, _ in spec.values()) + 1
    colw = [max(extent[k].width for k, v in spec.items() if v[0] == i) + 2 * pad_x
            for i in range(ncol)]
    gap = (W - 2 * margin - sum(colw)) / (ncol - 1)
    cx = [margin + sum(colw[:i]) + i * gap + colw[i] / 2 for i in range(ncol)]

    r = {}
    for key, (col, row, _, pal) in spec.items():
        w, h = colw[col], extent[key].height + 2 * pad_y
        r[key] = Bbox.from_bounds(cx[col] - w / 2, row - h / 2, w, h)
        ax.add_patch(FancyBboxPatch((r[key].x0, r[key].y0), w, h, mutation_scale=1,
                                    boxstyle="round,pad=0,rounding_size=3.5",
                                    facecolor=FILL[pal], edgecolor=EDGE[pal],
                                    linewidth=1.1 if pal == "plain" else 1.0, zorder=2))
        txt[key].set_x(cx[col]); txt[key].set_zorder(3)

    def mid(k):
        return (r[k].y0 + r[k].y1) / 2

    def line(pts):
        ax.add_line(Line2D([p[0] for p in pts], [p[1] for p in pts], color=INK, lw=0.85,
                           solid_joinstyle="miter", solid_capstyle="projecting", zorder=1))

    def arrow(pts):
        ax.add_patch(FancyArrowPatch(path=Path(pts), arrowstyle="-|>", mutation_scale=7.5,
                                     lw=0.85, color=INK, shrinkA=0, shrinkB=0, zorder=1,
                                     joinstyle="miter", capstyle="projecting"))

    def straight(a, b):
        if abs(mid(a) - mid(b)) > 1e-6:                      # keeps the connector horizontal
            raise SystemExit(f"fig4: {a!r} and {b!r} are not on the same row")
        arrow([(r[a].x1, mid(a)), (r[b].x0, mid(b))])

    def fan_out(a, dsts):
        ya, yd = mid(a), [mid(d) for d in dsts]
        bus = (r[a].x1 + min(r[d].x0 for d in dsts)) / 2
        line([(r[a].x1, ya), (bus, ya)])
        line([(bus, min(yd + [ya])), (bus, max(yd + [ya]))])
        for d, y in zip(dsts, yd, strict=True):
            arrow([(bus, y), (r[d].x0, y)])

    def fan_in(srcs, b):
        ys, yb = [mid(s) for s in srcs], mid(b)
        bus = (max(r[s].x1 for s in srcs) + r[b].x0) / 2
        for s, y in zip(srcs, ys, strict=True):
            line([(r[s].x1, y), (bus, y)])
        line([(bus, min(ys + [yb])), (bus, max(ys + [yb]))])
        arrow([(bus, yb), (r[b].x0, yb)])

    fan_out("corpus", ["normal", "attackp"])
    fan_out("normal", ["fit", "calib", "held"])
    arrow([(cx[2], r["held"].y0), (cx[2], r["probe"].y1)])       # held out -> probe generation
    straight("held", "ordinary")
    straight("probe", "legit")
    straight("attackp", "attacks")
    fan_in(["ordinary", "legit", "attacks"], "assess")
    out_y = r["attacks"].y1 - 6
    arrow([(cx[4], r["assess"].y0), (cx[4], out_y)])
    ax.text(cx[4], out_y - 9, "separation", ha="center", va="top", fontsize=fs, style="italic")

    ax.legend(handles=[Patch(facecolor=FILL[k], edgecolor=EDGE[k], linewidth=1.0, label=lab)
                       for k, lab in (("normal", "normal traffic"),
                                      ("probe", "generated legitimate anomalies"),
                                      ("attack", "attack traffic"))],
              loc="lower left", bbox_to_anchor=(margin / W, 5.0 / H), frameon=False, ncol=3,
              fontsize=8, handlelength=1.5, handleheight=0.95, columnspacing=1.5,
              handletextpad=0.5, borderpad=0.0, borderaxespad=0.0)

    # No box may collide with another, and every box must sit inside the page.
    keys = list(r)
    for i, a in enumerate(keys):
        if r[a].x0 < 0 or r[a].y0 < 0 or r[a].x1 > W or r[a].y1 > H:
            raise SystemExit(f"fig4: box {a!r} leaves the page")
        for b in keys[i + 1:]:
            if r[a].overlaps(r[b]):
                raise SystemExit(f"fig4: boxes {a!r} and {b!r} overlap")
    _save(fig, "fig4_pipeline")


for f in (fig1_detectors, fig2_conditions, fig3_ablation, fig4_pipeline):
    f(); print(f"  {f.__name__}")
print(f"wrote {len(list(OUT.glob('fig?_*.pdf')))} figures to {OUT}")
