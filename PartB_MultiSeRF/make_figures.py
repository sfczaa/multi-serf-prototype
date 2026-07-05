"""
Figure generation for the Part B write-up.

Reads the existing results_partB_*.json files (does NOT rerun experiments) and
writes PNGs under figures/:

  fig_ratio_by_K.png    QPS ratio (Multi-SeRF / SeRF+ResidualB) vs B selectivity
                        for K=4, K=16, K=32 — the bucket-count trade-off.
  fig_main_K16.png      Headline K=16 run: per-selectivity QPS ratio bars around
                        the ratio=1 parity line.
  fig_recall_qps.png    The measurement method: recall vs QPS as over-fetch α
                        grows, s_B=1%, both arms, with the 0.9 recall floor.
  fig_mechanism.png     Schematic: residual-B filtering vs B-bucket routing.

Usage:  py -3 make_figures.py
Requires matplotlib (only for this script; the prototype itself needs numpy only).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"

# palette (light surface); series identity is also carried by direct labels
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2ND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"K=4": "#2a78d6", "K=16": "#1baf7a", "K=32": "#eda100"}
BLUE = "#2a78d6"   # ratio >= 1 (Multi-SeRF faster)
RED = "#e34948"    # ratio  < 1 (Multi-SeRF slower)

plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.labelcolor": INK_2ND,
})


def load_sweep(path: Path) -> tuple[list[float], list[float]]:
    """Return (s_B list, CS/SeRF QPS ratio list) from one results JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"]
    return ([r["s_B"] for r in rows],
            [r["cs_over_serf_qps_ratio"] for r in rows])


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def parity_line(ax, x_text: float, ha: str = "right"):
    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(x_text, 1.03, "ratio = 1 (parity with SeRF+ResidualB)",
            color=MUTED, fontsize=8.5, va="bottom", ha=ha)


RATIO_TICKS = [0.25, 0.5, 1, 2, 4, 8, 16, 32]


def fig_ratio_by_k():
    runs = {
        "K=4": HERE / "results_partB_K4.json",
        "K=16": HERE / "results_partB_main.json",
        "K=32": HERE / "results_partB_K32.json",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    for label, path in runs.items():
        s_b, ratio = load_sweep(path)
        x = [s * 100 for s in s_b]
        ax.plot(x, ratio, color=SERIES[label], linewidth=2,
                marker="o", markersize=5, label=label)
        ax.annotate(label, (x[-1], ratio[-1]), xytext=(8, 0),
                    textcoords="offset points", color=INK, fontsize=9,
                    va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    parity_line(ax, x_text=1.0, ha="left")
    style_axes(ax)
    ax.xaxis.set_major_locator(FixedLocator([1, 5, 10, 25, 50]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels(["1%", "5%", "10%", "25%", "50%"])
    ax.yaxis.set_major_locator(FixedLocator(RATIO_TICKS))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticklabels([f"{t:g}×" for t in RATIO_TICKS])
    ax.set_xlim(0.8, 75)
    ax.set_xlabel("B-range selectivity (fraction of points passing the B predicate)")
    ax.set_ylabel("QPS ratio: Multi-SeRF / SeRF+ResidualB")
    ax.set_title("Bucket count K trades narrow-B wins against wide-B overhead",
                 fontsize=11, color=INK, loc="left", pad=24)
    ax.text(0, 1.03, "QPS at recall ≥ 0.9 · n=5000, dim=32, nq=100 · synthetic data",
            transform=ax.transAxes, color=INK_2ND, fontsize=8.5, va="bottom")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    out = FIGDIR / "fig_ratio_by_K.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_main_k16():
    s_b, ratio = load_sweep(HERE / "results_partB_main.json")
    # min–max whiskers across the data-seed reruns, when they are present
    seed_ratios = [ratio]
    for f in ("results_partB_seed1.json", "results_partB_seed2.json"):
        if (HERE / f).exists():
            seed_ratios.append(load_sweep(HERE / f)[1])
    labels = [f"{s:.0%}" for s in s_b]
    xs = range(len(s_b))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    ax.set_yscale("log")
    for i, r in zip(xs, ratio):
        # bar drawn from the parity line (1) to the ratio, not from zero
        ax.bar(i, r - 1, bottom=1, width=0.55, color=BLUE if r >= 1 else RED)
        lo = min(sr[i] for sr in seed_ratios)
        hi = max(sr[i] for sr in seed_ratios)
        if len(seed_ratios) > 1 and hi > lo:
            ax.errorbar(i, (lo * hi) ** 0.5, yerr=[[(lo * hi) ** 0.5 - lo],
                                                   [hi - (lo * hi) ** 0.5]],
                        fmt="none", ecolor=INK_2ND, elinewidth=1.2, capsize=4)
        top, bot = max(hi, r), min(lo, r)
        if r >= 1:
            ax.text(i, top * 1.12, f"{r:.2f}×", ha="center", va="bottom",
                    color=INK, fontsize=9.5)
        else:
            ax.text(i, bot / 1.12, f"{r:.2f}×", ha="center", va="top",
                    color=INK, fontsize=9.5)
    parity_line(ax, x_text=len(s_b) - 0.55)
    style_axes(ax)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_locator(FixedLocator(RATIO_TICKS))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticklabels([f"{t:g}×" for t in RATIO_TICKS])
    ax.set_ylim(0.25, 45)
    ax.set_xlabel("B-range selectivity")
    ax.set_ylabel("QPS ratio: Multi-SeRF / SeRF+ResidualB")
    ax.set_title("Headline run (K=16): large wins on narrow B, real cost on wide B",
                 fontsize=11, color=INK, loc="left", pad=24)
    ax.text(0, 1.03, "QPS at recall ≥ 0.9 · n=5000, dim=32, nq=100 · bars: "
                     "recorded headline run · whiskers: min–max over 3 data seeds",
            transform=ax.transAxes, color=INK_2ND, fontsize=8.5, va="bottom")
    fig.tight_layout()
    out = FIGDIR / "fig_main_K16.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_recall_qps():
    data = json.loads((HERE / "results_partB_main.json").read_text(encoding="utf-8"))
    row = next(r for r in data["results"] if r["s_B"] == 0.01)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    series = [("Multi-SeRF", row["cs_curve"], BLUE),
              ("SeRF+ResidualB", row["serf_curve"], MUTED)]
    for name, curve, color in series:
        qps = [c["qps"] for c in curve]
        rec = [c["recall"] for c in curve]
        ax.plot(qps, rec, color=color, linewidth=2, marker="o", markersize=5,
                label=name)
        best = next(c for c in curve if c["recall"] >= 0.9)
        for c in curve:
            # skip α labels in the near-zero-recall cluster to avoid collisions
            if c["recall"] < 0.08 and c is not curve[0]:
                continue
            ax.annotate(f"α={c['alpha']:g}", (c["qps"], c["recall"]),
                        xytext=(0, -14), textcoords="offset points",
                        ha="center", color=MUTED, fontsize=7.5)
        # ring the reported point: the first α that clears the recall floor
        ax.plot(best["qps"], best["recall"], marker="o", markersize=11,
                mfc="none", mec=color, mew=1.5)
    ax.axhline(0.9, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(0.42, 0.905, "recall floor 0.9", color=MUTED, fontsize=8.5,
            va="bottom",
            transform=ax.get_yaxis_transform())  # x in axes frac, y in data
    ax.set_xscale("log")
    style_axes(ax)
    ax.set_xlabel("QPS (log scale)")
    ax.set_ylabel("mean recall@10")
    ax.set_title("Each method raises over-fetch α until it clears recall 0.9; "
                 "the ringed point is reported", fontsize=11, color=INK,
                 loc="left", pad=24)
    ax.text(0, 1.03, "s_B = 1%, K=16 headline run · SeRF+ResidualB needs "
                     "α=256 to reach the floor; Multi-SeRF needs α=8",
            transform=ax.transAxes, color=INK_2ND, fontsize=8.5, va="bottom")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    out = FIGDIR / "fig_recall_qps.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_mechanism():
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=200)
    K = 8
    b_lo, b_hi = 0.38, 0.50          # query B window (schematic)
    box = dict(x=0.10, y=0.12, w=0.84, h=0.74)

    def base(ax, title):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
        ax.add_patch(Rectangle((box["x"], box["y"]), box["w"], box["h"],
                               fill=False, edgecolor=BASELINE, linewidth=1.2))
        ax.annotate("", xytext=(box["x"], box["y"] - 0.045),
                    xy=(box["x"] + 0.2, box["y"] - 0.045),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
        ax.text(box["x"] + 0.22, box["y"] - 0.045, "attribute A", color=MUTED,
                fontsize=8.5, va="center")
        ax.annotate("", xytext=(box["x"] - 0.045, box["y"]),
                    xy=(box["x"] - 0.045, box["y"] + 0.2),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
        ax.text(box["x"] - 0.045, box["y"] + 0.24, "attribute B", color=MUTED,
                fontsize=8.5, ha="center", rotation=90, va="bottom")

    def query_band(ax):
        y0 = box["y"] + b_lo * box["h"]
        y1 = box["y"] + b_hi * box["h"]
        ax.add_patch(Rectangle((box["x"], y0), box["w"], y1 - y0, fill=False,
                               edgecolor=INK, linewidth=1.4,
                               linestyle=(0, (4, 3))))
        ax.text(box["x"] + box["w"] + 0.015, (y0 + y1) / 2,
                "query\nB window", color=INK, fontsize=8, va="center")

    # -- left: baseline ---------------------------------------------------
    ax = axes[0]
    base(ax, "SeRF+ResidualB (baseline, K=1)")
    ax.add_patch(Rectangle((box["x"], box["y"]), box["w"], box["h"],
                           facecolor=RED, alpha=0.14, edgecolor="none"))
    query_band(ax)
    ax.text(box["x"] + box["w"] / 2, box["y"] + box["h"] * 0.78,
            "one A-graph over all points:\nthe search visits candidates\n"
            "across the whole B range",
            ha="center", color=INK_2ND, fontsize=8.5)
    ax.text(box["x"] + box["w"] / 2, box["y"] + box["h"] * 0.16,
            "residual filter discards everything\noutside the window "
            "→ wasted work\nwhen the B predicate is narrow",
            ha="center", color=INK_2ND, fontsize=8.5)

    # -- right: multi-serf -------------------------------------------------
    ax = axes[1]
    base(ax, f"Multi-SeRF (Compound Segment, K={K})")
    gap = 0.006
    for j in range(K):
        y0 = box["y"] + (j / K) * box["h"] + gap
        hh = box["h"] / K - 2 * gap
        lo, hi = j / K, (j + 1) / K
        hit = not (hi < b_lo or lo > b_hi)
        ax.add_patch(Rectangle((box["x"] + gap, y0), box["w"] - 2 * gap, hh,
                               facecolor=BLUE if hit else "#f0efec",
                               alpha=0.30 if hit else 1.0, edgecolor="none"))
    query_band(ax)
    ax.text(box["x"] + box["w"] / 2, box["y"] + box["h"] * 0.78,
            "K equal-frequency B-buckets,\none SegmentGraph1D over A\n"
            "inside each bucket",
            ha="center", color=INK_2ND, fontsize=8.5)
    ax.text(box["x"] + box["w"] / 2, box["y"] + box["h"] * 0.16,
            "only buckets overlapping the window\nare searched (here 2 of 8) —\n"
            "the rest are skipped before any graph work",
            ha="center", color=INK_2ND, fontsize=8.5)

    fig.suptitle("Same SegmentGraph1D code in both arms — the measured "
                 "difference isolates B-bucket routing",
                 fontsize=10, color=INK_2ND, y=0.03, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = FIGDIR / "fig_mechanism.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    FIGDIR.mkdir(exist_ok=True)
    for fn in (fig_ratio_by_k, fig_main_k16, fig_recall_qps, fig_mechanism):
        # print the repo-relative path; the absolute path may contain non-ASCII
        # segments the Windows console codepage cannot encode
        print("wrote", fn().relative_to(HERE).as_posix())


if __name__ == "__main__":
    main()
