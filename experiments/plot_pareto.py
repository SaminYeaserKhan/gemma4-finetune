"""The headline figure: accuracy against cloud tokens per question.

The thesis argues that a supervisor is worth calling only if it buys accuracy
that a free on-device method cannot. That claim lives or dies on one picture,
and the picture has to include the free controls or it is not an argument --
it is a chart of the cascade agreeing with itself.

Three things are drawn deliberately:

- **The free control line** at self-consistency@3. Every point below it is a
  configuration that paid cloud tokens for nothing. Before the stacked arm
  (dossier 5.7.3) every cascade point sat on it.
- **The pass@3 ceiling.** No verifier can find an answer the local model never
  produced, so this bounds the whole study.
- **The Pareto frontier**, which is the only part a deployer would read: the
  points where nothing else is at least as accurate for no more cost.

Reads reports/fydp3_summary.csv so the figure cannot drift from the table.

    python experiments/plot_pareto.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

Point = tuple[str, float, float]  # label, cloud tokens/q, accuracy


def pareto_frontier(points: list[Point]) -> list[Point]:
    """Points nothing else beats on both cost and accuracy, cheapest first.

    Ties on both axes collapse to one point: two identical operating points
    are the same choice, and drawing both puts a visible doubled marker on the
    frontier that readers take for two distinct configurations.
    """
    frontier: list[Point] = []
    for point in sorted(points, key=lambda p: (p[1], -p[2])):
        if frontier and point[2] <= frontier[-1][2]:
            continue
        frontier.append(point)
    return frontier


def load_points(csv_path: Path) -> list[Point]:
    points = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cost, accuracy = row.get("cloud_tokens_per_q"), row.get("accuracy")
            if not cost or not accuracy:
                continue
            points.append((row["condition"], float(cost), float(accuracy)))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="reports/fydp3_summary.csv")
    parser.add_argument("--output", default="reports/figures/pareto.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = load_points(Path(args.summary))
    by_label = {label: (cost, accuracy) for label, cost, accuracy in points}

    ceiling = by_label.get("pass@3 CEILING")
    control = by_label.get("self-consistency@3 (majority)")
    local = by_label.get("local only (1 sample)")
    # The ceiling and the blind-retry control are reference lines, not
    # operating points -- nobody can deploy "pass@3", it needs the gold answer.
    excluded = {"pass@3 CEILING", "blind retry, take last"}
    operating = [p for p in points if p[0] not in excluded]

    stacked = [p for p in operating if "voting" in p[0] and p[1] > 0]
    plain = [p for p in operating if "voting" not in p[0] and p[1] > 0]
    free = [p for p in operating if p[1] == 0]

    fig, ax = plt.subplots(figsize=(9.5, 6))

    # Zoom to the band the results actually occupy. Anchoring at 0 would spend
    # four fifths of the panel on empty space and squeeze a 4.9-point effect
    # into something the eye reads as no difference.
    lo = min(p[2] for p in operating) - 0.025
    hi = (ceiling[1] if ceiling else max(p[2] for p in operating)) + 0.022
    ax.set_ylim(lo, hi)
    xmax = max(p[1] for p in operating) * 1.30
    ax.set_xlim(-22, xmax)

    if control:
        # Everything under this line paid cloud tokens for nothing.
        ax.axhspan(lo, control[1], color="#c1440e", alpha=0.055, zorder=0)

    def rule(value, style, colour, text, weight="normal"):
        # Parked in the empty mid-band. The right edge holds the arm cluster
        # and the left edge the free markers, so a label at either end lands
        # on top of a data point.
        ax.axhline(value, ls=style, lw=1.4, color=colour, zorder=1)
        ax.text(xmax * 0.28, value + (hi - lo) * 0.010, text, fontsize=9,
                color=colour, ha="center", fontweight=weight,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.6, alpha=0.85))

    if ceiling:
        rule(ceiling[1], ":", "#555555", f"pass@3 ceiling  {ceiling[1]:.1%}")
    if control:
        rule(control[1], "--", "#c1440e",
             f"free control: self-consistency@3  {control[1]:.1%}", "bold")
    if local:
        rule(local[1], "-", "#9a9a9a", f"fine-tuned model alone  {local[1]:.1%}")

    frontier = pareto_frontier(operating)
    ax.plot([p[1] for p in frontier], [p[2] for p in frontier],
            "-", lw=1.6, color="#1f4e79", alpha=0.5, zorder=2,
            label="Pareto frontier")

    ax.scatter([p[1] for p in free], [p[2] for p in free], s=120, marker="s",
               facecolors="#c1440e", edgecolors="#7a2b09", zorder=4,
               label="free, on-device only")
    ax.scatter([p[1] for p in plain], [p[2] for p in plain], s=115, marker="o",
               facecolors="white", edgecolors="#1f4e79", linewidths=1.8, zorder=4,
               label="cascade run instead of voting")
    ax.scatter([p[1] for p in stacked], [p[2] for p in stacked], s=125, marker="o",
               facecolors="#1f4e79", edgecolors="#12324d", zorder=5,
               label="cascade stacked on voting")

    # L0/L1/L2 sit within 5 tokens of each other on x, so labels cannot go to
    # the side without overprinting the neighbouring arm. Splitting them by
    # family instead -- stacked above, plain below -- keeps the two clusters
    # apart, which is also the comparison the reader is here to make.
    for label, cost, accuracy in operating:
        stacked_arm = "voting" in label
        if cost == 0:
            # Named by its own reference line, which also carries the figure.
            # Annotating the marker too just prints the phrase twice.
            continue
        elif cost < 340:
            # The six arms share x to within 5 tokens but are well separated on
            # y, so a single label column in the empty mid-band reads cleanly
            # where per-point offsets collide.
            ax.annotate(label, (cost, accuracy), textcoords="offset points",
                        xytext=(-14, -3.5), fontsize=8.5, ha="right",
                        color="#12324d",
                        fontweight="bold" if stacked_arm else "normal")
        else:
            ax.annotate(label, (cost, accuracy), textcoords="offset points",
                        xytext=(14, -3.5), fontsize=8.5, ha="left",
                        color="#12324d",
                        fontweight="bold" if stacked_arm else "normal")

    ax.set_xlabel("Cloud tokens per question   (0 = nothing leaves the device)")
    ax.set_ylabel("GSM8K exact-answer accuracy")
    ax.set_title(
        "Accuracy vs. cloud cost: gemma-4-E2B-it with a GLM-4.7-Flash supervisor\n"
        "1,319 GSM8K test questions. Hollow points are off the frontier - "
        "no deployer would choose them.",
        fontsize=10.5, linespacing=1.4)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.96)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")
    print("\nPareto frontier (what a deployer would actually choose):")
    for label, cost, accuracy in frontier:
        print(f"  {cost:7.1f} tokens/q   {accuracy:.4f}   {label}")


if __name__ == "__main__":
    main()
