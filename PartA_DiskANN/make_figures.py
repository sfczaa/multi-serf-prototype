"""
Figure generation for the Part A write-up.

Reads the recorded held-out result JSONs (never reruns experiments) and writes
PNGs under figures/:

  fig_recall_vs_L.png   recall@10 vs beam width L (n=5k, n=10k held-out), with
                        the pynndescent reference and the proposal's 0.95 bar.
  fig_latency.png       eager vs mmap mean query latency by L (warm cache).
  fig_file_layout.png   the page-aligned file format + to-scale size breakdown.

Usage:  py -3 make_figures.py     (requires matplotlib)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2ND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"

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

RUNS = {"n=5k": "results_5k_heldout.json", "n=10k": "results_10k_heldout.json"}


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load_rows(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["name"]: r for r in data["results"]}


def fig_recall_vs_l():
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    Ls = [64, 128, 256]
    runs = dict(RUNS)
    if (HERE / "results_siftsmall.json").exists():
        runs["SIFT10K"] = "results_siftsmall.json"
    for (label, fname), color in zip(runs.items(), (BLUE, AQUA, YELLOW)):
        rows = load_rows(HERE / fname)
        rec = [rows[f"proto-eager-L{L}"]["recall@10"] for L in Ls]
        ax.plot(Ls, rec, color=color, linewidth=2, marker="o", markersize=5,
                label=f"prototype, {label}")
        ax.annotate(label, (Ls[-1], rec[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK,
                    fontsize=9)
        if label == "SIFT10K":       # keep the reference lines uncluttered:
            continue                 # gaussian pynndescent refs only
        pynn = rows["pynndescent"]["recall@10"]
        ax.axhline(pynn, color=color, linewidth=1.0, linestyle=(0, (1, 2)))
        ax.text(64, pynn + 0.004, f"pynndescent ({label}): {pynn:.3f}",
                color=color, fontsize=8)
    ax.axhline(0.95, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(330, 0.955, "proposal target 0.95", color=MUTED, fontsize=8.5,
            ha="right")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(Ls))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([str(L) for L in Ls])
    style_axes(ax)
    ax.set_xlim(58, 340)
    ax.set_ylim(0.55, 1.0)
    ax.set_xlabel("beam width L")
    ax.set_ylabel("recall@10 (held-out queries)")
    ax.set_title("Recall misses the 0.95 bar on held-out Gaussian — and "
                 "clears it on real SIFT10K", fontsize=11, color=INK,
                 loc="left", pad=24)
    ax.text(0, 1.03, "Gaussian: dim=128, nq=200 held-out · SIFT10K: corpus "
                     "queries, nq=100 · pynndescent refs are Gaussian runs",
            transform=ax.transAxes, color=INK_2ND, fontsize=8.5, va="bottom")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    out = FIGDIR / "fig_recall_vs_L.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_latency():
    Ls = [64, 128, 256]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), dpi=200, sharey=True)
    for ax, (label, fname) in zip(axes, RUNS.items()):
        rows = load_rows(HERE / fname)
        eager = [rows[f"proto-eager-L{L}"]["query_mean_ms"] for L in Ls]
        mm = [rows[f"proto-mmap-L{L}"]["query_mean_ms"] for L in Ls]
        xs = range(len(Ls))
        ax.bar([x - 0.19 for x in xs], eager, width=0.36, color=BLUE,
               label="eager (preloaded)")
        ax.bar([x + 0.19 for x in xs], mm, width=0.36, color=AQUA,
               label="mmap (warm cache)")
        for x, (e, m) in enumerate(zip(eager, mm)):
            ax.text(x + 0.19, m + 0.4, f"{m / e:.2f}x", ha="center",
                    color=INK, fontsize=8.5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"L={L}" for L in Ls])
        style_axes(ax)
        ax.set_title(label, fontsize=10, color=INK_2ND)
    axes[0].set_ylabel("mean query latency (ms)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("mmap surcharge over eager is 1.2-1.9x at warm cache "
                 "(not a cold-disk measurement)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.90, "same algorithm, same bytes, identical results — "
                         "the delta is the storage path · single run, nq=200",
             color=INK_2ND, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = FIGDIR / "fig_latency.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_file_layout():
    rows = load_rows(HERE / "results_10k_heldout.json")
    lay = next(r["layout"] for r in rows.values()
               if r.get("layout") and "total_bytes" in r["layout"])
    segs = [("header", 4096, MUTED),
            ("graph adjacency", lay["graph_bytes"], BLUE),
            ("PQ codes", lay["pq_bytes"], AQUA),
            ("full vectors", lay["full_bytes"], YELLOW),
            ("PQ codebook", lay["total_bytes"] - 4096 - lay["graph_bytes"]
             - lay["pq_bytes"] - lay["full_bytes"], VIOLET)]
    total = lay["total_bytes"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 4.8), dpi=200,
                                   height_ratios=[1.6, 1])
    # -- top: format diagram (not to scale) -----------------------------
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    widths = [0.10, 0.24, 0.16, 0.28, 0.16]
    x = 0.03
    for (name, nbytes, color), w in zip(segs, widths):
        ax1.add_patch(Rectangle((x, 0.42), w - 0.006, 0.30, facecolor=color,
                                alpha=0.30, edgecolor=color, linewidth=1.2))
        ax1.text(x + w / 2 - 0.003, 0.57, name, ha="center", va="center",
                 color=INK, fontsize=8.5)
        size = f"{nbytes / 1e6:.2f} MB" if nbytes >= 100_000 else f"{nbytes // 1024} KB"
        ax1.text(x + w / 2 - 0.003, 0.36, size, ha="center", va="top",
                 color=INK_2ND, fontsize=8)
        x += w
    ax1.text(0.03, 0.86, "single page-aligned file · every segment starts on a "
                         "4 KB page boundary · widths not to scale",
             color=INK_2ND, fontsize=8.5)
    ax1.text(0.03, 0.16, f"graph record: deg(u16) pad(u16) M×u32 neighbours, "
             f"64 B-aligned -> {lay['rec_size']} B/node, "
             f"{lay['nodes_per_page']} nodes/page",
             color=INK_2ND, fontsize=8.5)
    ax1.set_title("On-disk layout (n=10k, dim=128, M=32, pq_m=16)",
                  fontsize=11, color=INK, loc="left")

    # -- bottom: to-scale byte breakdown --------------------------------
    x = 0.0
    for name, nbytes, color in segs:
        frac = nbytes / total
        ax2.barh(0, frac, left=x, height=0.5, color=color, alpha=0.85)
        if frac > 0.08:
            ax2.text(x + frac / 2, 0, f"{100 * frac:.0f}%", ha="center",
                     va="center", color="white", fontsize=8.5)
        x += frac
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_yticks([])
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", "", f"to scale · total {total / 1e6:.2f} MB", "", "100%"])
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)
    ax2.set_title("full vectors dominate the footprint; the beam search's hot "
                  "working set (graph + PQ codes) is the small share",
                  fontsize=8.5, color=INK_2ND, loc="left")
    fig.tight_layout()
    out = FIGDIR / "fig_file_layout.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    FIGDIR.mkdir(exist_ok=True)
    for fn in (fig_recall_vs_l, fig_latency, fig_file_layout):
        print("wrote", fn().relative_to(HERE).as_posix())


if __name__ == "__main__":
    main()
