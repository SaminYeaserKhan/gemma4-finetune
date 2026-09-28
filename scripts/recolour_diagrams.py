"""Repaint every diagram source in the hand-drawn Canva palette.

The diagrams were originally drawn in a muted green/orange/blue scheme. The
hand-drawn set in docs/diagram_images uses a brighter, higher-contrast palette
that reads better in print, so the generated diagrams are remapped onto it and
the two sets stop looking like they came from different projects.

Colours are sampled from the hand-drawn PNGs, not invented:

    #99daff  light blue   start / terminal
    #86e5a1  green        result
    #f9b972  orange       decision, and the off-device marker
    #46bbfe  mid blue     arrows and node outlines
    #1185e1  strong blue  emphasis

Run once; it edits the sources in place and is a no-op the second time.

    python scripts/recolour_diagrams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# old fill/stroke -> new fill/stroke
PALETTE = {
    # "on the device" region: pale blue wash instead of pale green
    "#e8f4ea": "#eaf7ff",
    "#2d6a4f": "#46bbfe",
    # "off the device" region and the marker: keep orange, brighten it
    "#fdf0e6": "#fff2e3",
    "#b5651d": "#f0a04b",
    # results: the hand-drawn set ends every flow on a green pill
    "#e6eefc": "#86e5a1",
    "#2b4c8c": "#009d25",
    # a marker's mistakes: soft red
    "#fdece9": "#ffd9d4",
    "#b03a2e": "#e06a5a",
    # the answer key, which is code rather than a model: stays neutral grey
    "#eeeeee": "#f2f2f2",
    "#888": "#9a9a9a",
}

TARGETS = [
    REPO / "docs" / "DIAGRAMS.md",
    REPO / "docs" / "diagrams-technical" / "DIAGRAMS_TECHNICAL.md",
    REPO / "scripts" / "build_experiment_catalogue.py",
]


def repaint(text: str) -> tuple[str, int]:
    changed = 0
    # Longest keys first so "#888" cannot clip a longer literal.
    for old in sorted(PALETTE, key=len, reverse=True):
        if old in text:
            changed += text.count(old)
            text = text.replace(old, PALETTE[old])
    return text, changed


def main() -> int:
    total = 0
    for path in TARGETS:
        if not path.exists():
            print(f"  [skip] missing {path}", file=sys.stderr)
            continue
        before = path.read_text(encoding="utf-8")
        after, changed = repaint(before)
        if changed:
            path.write_text(after, encoding="utf-8")
        total += changed
        print(f"  {path.relative_to(REPO)}: {changed} colour(s) remapped")
    if not total:
        print("\nNothing to change -- already in the new palette.")
    else:
        print(f"\n{total} colour references remapped. Re-render to see it:")
        print("  python scripts/render_diagrams.py")
        print("  python scripts/build_experiment_catalogue.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
