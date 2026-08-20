"""Data-efficiency curves: Spec-adherence and Robustness vs training-set size.

Numbers = the self-consistent 100-scenario eval_dev runs (FINAL_RESULTS.md §4).
Frontier reference = gpt-5.6-luna + structured prompt (best prompted combo).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

N = [0, 125, 250, 500, 1000]
CURVES = {
    "adherence": {
        "y": [0.0, 43.2, 93.6, 97.2, 96.2],
        "ref": 89.0,
        "title": "Spec-adherence vs training-set size",
        "ylabel": "Spec-adherence (%)",
        "fname": "data_efficiency_adherence.png",
    },
    "robustness": {
        "y": [0.0, 15.0, 84.0, 93.0, 92.0],
        "ref": 66.7,
        "title": "Robustness vs training-set size",
        "ylabel": "Robustness (%)",
        "fname": "data_efficiency_robustness.png",
    },
}

SERIES = "#2563eb"   # single-series hue
REF = "#6b7280"      # neutral gray reference
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"

for key, c in CURVES.items():
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # frontier reference line, direct-labeled
    ax.axhline(c["ref"], color=REF, lw=1.5, ls=(0, (5, 4)), zorder=2)
    ax.annotate(f"best prompted frontier ({c['ref']:.1f}%)\ngpt-5.6-luna, structured prompt",
                xy=(1000, c["ref"]), xytext=(990, c["ref"] - 4.5),
                ha="right", va="top", fontsize=8.5, color=MUTED, linespacing=1.3)

    # the curve
    ax.plot(N, c["y"], color=SERIES, lw=2, marker="o", ms=7,
            markerfacecolor=SERIES, markeredgecolor="white", markeredgewidth=1.5,
            zorder=3, clip_on=False)

    # direct value labels (sparse ablation points - the values are the evidence)
    for x, y in zip(N, c["y"]):
        dy = 3.5 if key == "adherence" and x != 0 else 3.5
        va = "bottom"
        if x == 0:
            ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(x + 12, y + 2.5),
                        fontsize=9, color=INK, ha="left", va="bottom")
        else:
            ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(x, y + dy),
                        fontsize=9, color=INK, ha="center", va=va)

    ax.set_title(c["title"] + "  —  Qwen3-1.7B QLoRA, no system prompt",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.set_xlabel("training conversations (N)", fontsize=10, color=MUTED)
    ax.set_ylabel(c["ylabel"], fontsize=10, color=MUTED)

    ax.set_xlim(-25, 1025)
    ax.set_ylim(0, 104)
    ax.xaxis.set_major_locator(FixedLocator(N))
    ax.set_xticklabels(["0\n(base)", "125", "250", "500", "1000"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.tight_layout()
    fig.savefig(OUT / c["fname"], facecolor="white", bbox_inches="tight")
    print("wrote", OUT / c["fname"])
