"""Trim the blank canvas around exported diagrams.

The hand-drawn diagrams are exported as 1920x1080 slides, so the drawing itself
occupies 38-63% of the file. Placed in a document at full text width, the
content therefore renders far smaller than the space allows. Cropping to the
ink, with a small uniform margin, makes each diagram roughly 1.4-2x larger for
free.

Originals are never modified; cropped copies are written to a sibling folder.

    python scripts/crop_diagrams.py docs/diagram_images docs/diagram_images/cropped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageChops

PAD = 12  # px of breathing room kept around the ink


def content_box(image: Image.Image):
    """Bounding box of everything that is not the background colour.

    Uses the top-left pixel as the background reference rather than assuming
    white, so a diagram exported on a tinted canvas still crops correctly.
    """
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
    return ImageChops.difference(rgb, background).getbbox()


def crop(source: Path, destination: Path) -> tuple[int, int] | None:
    image = Image.open(source)
    box = content_box(image)
    if box is None:
        return None
    left, top, right, bottom = box
    width, height = image.size
    box = (
        max(0, left - PAD),
        max(0, top - PAD),
        min(width, right + PAD),
        min(height, bottom + PAD),
    )
    cropped = image.crop(box)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(destination)
    return cropped.size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    files = sorted(p for p in args.source.iterdir() if p.suffix.lower() == ".png")
    if not files:
        print(f"no .png files in {args.source}", file=sys.stderr)
        return 1

    for path in files:
        before = Image.open(path).size
        after = crop(path, args.destination / path.name)
        if after is None:
            print(f"  {path.name}: blank, skipped")
            continue
        gain = (before[0] * before[1]) / (after[0] * after[1])
        print(f"  {path.name:22} {before[0]}x{before[1]} -> {after[0]}x{after[1]}"
              f"   {gain:.2f}x larger at the same width")
    print(f"\nWrote {len(files)} cropped copies to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
