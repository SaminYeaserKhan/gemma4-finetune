"""Draw the headline results figure from reports/fydp3_summary.csv.

    python experiments/plot_pareto.py

This was a scatter of accuracy against cost, and it did not work. Nothing sits
between 0 and 296 off-device tokens, and four of the six configurations fall
within eleven tokens of each other, so ten of the twelve points piled into a
sliver of the axis and every label collided with its neighbour's. Breaking the
axis helped and still left a chart that needed a paragraph of explanation.

The form here follows the data's job rather than the two measures. Each
configuration has a *before* and an *after* -- the same run scored without and
with the majority vote taken first -- which is a dumbbell, one row per
configuration, one hue in two shades. Rows cannot collide, so the crowding
problem cannot come back.

Cost stays out of the geometry and is printed as a column. Two measures on two
scales in one plot is the classic misleading chart; as a column the number is
exact and the axis stays honest.

Palette checked with the dataviz validator: #0b5d9e / #46bbfe / #c2691f pass
the lightness, chroma, CVD-separation and normal-vision checks on a light
surface. The mid blue carries a contrast warning, which the axis labels and the
accompanying table relieve.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent

AFTER = "#0b5d9e"     # stacked on the vote
BEFORE = "#46bbfe"    # used instead of the vote
FREE = "#c2691f"      # the free control
MUTED = "#6b6b6b"
INK = "#1a1a1a"

# (un-stacked key, stacked key, row label) -- plain names, not run names.
ROWS = [
    ("BEST-qwen9b", "BEST-qwen9b + voting", "9B checker"),
    ("more-escalation", "more-escalation + voting", "30B, asked more often"),
    ("smart-gate", "smart-gate + voting", "30B, better picking"),
    ("hint-full", "hint-full + voting", "30B, full hint"),
    ("hint-short", "hint-short + voting", "30B, short hint"),
    ("hint-none", "hint-none + voting", "30B, no hint"),
]


def load(path: Path) -> dict[str, tuple[float, float, int]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["condition"]] = (
                float(row["cloud_tokens_per_q"]),
                float(row["accuracy"]),
                int(row["correct"]),
            )
    return rows


def draw(data: dict, out: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })

    vote = data["self-consistency@3 (majority)"][1]
    solo = data["local only (1 sample)"][1]
    ceiling = data["pass@3 CEILING"][1]

    ordered = sorted(ROWS, key=lambda r: data[r[1]][1])   # worst at the bottom
    fig, ax = plt.subplots(figsize=(9.0, 4.9))

    # --- reference lines -------------------------------------------------
    # Two reference lines, not three. The fine-tuned model's own 55.9% sits
    # three points from the free vote, so its caption could not be placed
    # without colliding; it belongs in the caption and in Table III instead.
    ax.axvspan(0.54, vote, color=FREE, alpha=0.05, zorder=0)
    top = len(ordered)
    for value, colour, text, ha, x, y in (
        (vote, FREE, f"free: ask 3 times,\nkeep the majority — {vote:.0%}",
         "left", vote + 0.004, top + 0.18),
        # Tucked below and to the left of its line, so it cannot be read as
        # part of the cost column's heading.
        (ceiling, MUTED, f"best possible — {ceiling:.0%}",
         "right", ceiling - 0.005, top - 0.52),
    ):
        ax.axvline(value, ls="--", lw=1.6, color=colour, alpha=0.9, zorder=1)
        ax.text(x, y, text, fontsize=11.5, color=colour,
                ha=ha, va="bottom", linespacing=1.35)

    # --- the dumbbells ---------------------------------------------------
    for y, (plain, stacked, label) in enumerate(ordered):
        x_before, x_after = data[plain][1], data[stacked][1]
        gained = data[stacked][2] - data[plain][2]
        assert gained == 42, f"{stacked}: expected +42 answers, got {gained}"

        ax.plot([x_before, x_after], [y, y], "-", lw=3.0, color=BEFORE,
                alpha=0.55, zorder=2, solid_capstyle="round")
        ax.plot(x_before, y, "o", ms=12, color=BEFORE, zorder=3,
                mec="white", mew=2.0)
        ax.plot(x_after, y, "o", ms=13, color=AFTER, zorder=4,
                mec="white", mew=2.0)

        # Cost, as a column rather than a second axis.
        ax.text(0.762, y, f"{data[stacked][0]:,.0f}", fontsize=12, color=MUTED,
                ha="right", va="center", fontfamily="DejaVu Sans")

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([r[2] for r in ordered], fontsize=12.5)

    # Direct-label only the row that matters.
    best_y = len(ordered) - 1
    ax.annotate(f"{data['BEST-qwen9b + voting'][1]:.0%}",
                xy=(data["BEST-qwen9b + voting"][1], best_y),
                xytext=(11, 0), textcoords="offset points",
                fontsize=13, fontweight="bold", color=AFTER, va="center")

    # --- axes -------------------------------------------------------------
    ax.set_xlim(0.54, 0.775)
    ax.set_ylim(-0.7, len(ordered) + 1.15)
    ax.set_xticks([0.55, 0.60, 0.65, 0.70, 0.75])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Share of the 1,319 maths questions answered correctly", fontsize=13)
    ax.text(0.762, len(ordered) + 0.18, "off-device\nwords",
            fontsize=11.5, color=MUTED, ha="right", va="bottom", linespacing=1.35)

    ax.grid(axis="x", alpha=0.18, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cfcfcf")
    ax.tick_params(axis="y", length=0)

    legend = [
        Line2D([], [], marker="o", ls="", ms=12, color=BEFORE, mec="white", mew=2.0,
               label="checker used instead of the majority vote"),
        Line2D([], [], marker="o", ls="", ms=13, color=AFTER, mec="white", mew=2.0,
               label="checker used as well as it  (+42 answers, same cost)"),
    ]
    # Below the axis: inside, the legend landed on the bottom row's dumbbell.
    fig.legend(handles=legend, loc="lower center", ncol=1, fontsize=11.5,
               frameon=False, handletextpad=0.6, labelspacing=0.45,
               bbox_to_anchor=(0.56, -0.008))

    fig.subplots_adjust(left=0.225, right=0.975, top=0.855, bottom=0.30)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


def main_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=REPO / "reports" / "fydp3_summary.csv")
    parser.add_argument("--output", type=Path,
                        default=REPO / "reports" / "figures" / "pareto.png")
    args = parser.parse_args()
    draw(load(args.summary), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
