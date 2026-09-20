"""Draw the thesis diagrams as SVG and PNG, in both wordings.

Two Markdown files describe the same six diagrams:

    docs/DIAGRAMS.md                              plain language, for teammates
    docs/diagrams-technical/DIAGRAMS_TECHNICAL.md standard terminology, for the paper

Same system, same measured numbers, same output file names -- only the wording in the
boxes differs. Keeping the file names identical across the two output folders is what
lets one set be swapped for the other without editing the document that cites them.

The Markdown is the master copy. Generating the pictures from it, rather than drawing
them by hand in design software, is what stops them drifting out of date: the counts
inside Figure 2 are measured results, and a hand-drawn duplicate would keep showing
the old ones after a re-run.

    python scripts/render_diagrams.py                     both wordings
    python scripts/render_diagrams.py --variant plain     just one
    python scripts/render_diagrams.py --only fig2         just one figure, while iterating

SVG is the format to put in the report: it stays sharp at any size, and Word and
LaTeX both take it. The PNG exists for tools that refuse SVG.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Two wordings of the same diagrams, kept deliberately. The plain-language set is
# for teammates and for anyone being shown the system; the technical set is for a
# paper whose supervisor prefers standard terminology in figures. Same system, same
# measured numbers, same file names -- so one set can be swapped for the other
# without touching a document's figure references.
VARIANTS = {
    "plain": (
        REPO / "docs" / "DIAGRAMS.md",
        REPO / "reports" / "figures",
    ),
    "technical": (
        REPO / "docs" / "diagrams-technical" / "DIAGRAMS_TECHNICAL.md",
        REPO / "reports" / "figures" / "technical",
    ),
}

# A heading, then anything, then the first mermaid block under it.
FIGURE = re.compile(
    r"^##\s+Figure\s+(\d+)\s*[-—–]+\s*(.+?)\s*$(.*?)^```mermaid$\n(.*?)^```$",
    re.MULTILINE | re.DOTALL,
)

# Optional `<!-- figure: some_stable_name -->` under a heading, pinning the filename.
NAME_OVERRIDE = re.compile(r"<!--\s*figure:\s*([A-Za-z0-9_\-]+)\s*-->")


def slugify(text: str) -> str:
    """'What the system is made of' -> 'what_the_system_is_made_of'."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return cleaned.strip("_")


def find_figures(markdown: str) -> list[tuple[str, str]]:
    """(stem, mermaid source) for every `## Figure N` section, in document order.

    The output name normally comes from the heading, but a figure may pin its own
    with `<!-- figure: fig2_per_question_data_flow -->` under the heading. The
    headings are written in plain English for non-technical readers, and plain
    English makes long names that change whenever the wording is improved -- while
    the paper cites those filenames. Pinning keeps the two independent.
    """
    figures = []
    for match in FIGURE.finditer(markdown):
        number, title, between, body = match.groups()
        pinned = NAME_OVERRIDE.search(between)
        stem = pinned.group(1) if pinned else f"fig{number}_{slugify(title)}"
        figures.append((stem, body.rstrip()))
    return figures


def mermaid_command() -> list[str]:
    """How to invoke mmdc: installed if present, otherwise through npx."""
    local = shutil.which("mmdc")
    if local:
        return [local]
    npx = shutil.which("npx")
    if not npx:
        raise RuntimeError(
            "Neither mmdc nor npx is on PATH. Install Node.js, then either "
            "`npm i -g @mermaid-js/mermaid-cli` or make sure npx works."
        )
    return [npx, "-y", "@mermaid-js/mermaid-cli"]


def render(base: list[str], source: Path, target: Path, scale: int | None) -> None:
    # A transparent background turns into black boxes when pasted into Word, so the
    # background is forced white rather than left to the default.
    cmd = [*base, "-i", str(source), "-o", str(target), "-b", "white"]
    if scale:
        cmd += ["-s", str(scale)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(
            f"mermaid failed for {target.name}:\n"
            f"{result.stdout.strip()}\n{result.stderr.strip()}"
        )


def render_variant(name: str, only: str | None, base: list[str]) -> int:
    """Draw one wording of the diagrams. Returns how many figures it produced."""
    source, out_dir = VARIANTS[name]
    if not source.exists():
        print(f"  [skip] {name}: no {source}", file=sys.stderr)
        return 0

    figures = find_figures(source.read_text(encoding="utf-8"))
    if not figures:
        print(f"  [skip] {name}: no figure sections in {source}", file=sys.stderr)
        return 0

    if only:
        figures = [f for f in figures if only in f[0]]
        if not figures:
            print(f"  [skip] {name}: nothing matched --only {only!r}")
            return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_mermaid_src"
    scratch.mkdir(exist_ok=True)

    print(f"\n{name} wording -> {out_dir}")
    for stem, body in figures:
        src = scratch / f"{stem}.mmd"
        src.write_text(body + "\n", encoding="utf-8")
        render(base, src, out_dir / f"{stem}.svg", scale=None)
        render(base, src, out_dir / f"{stem}.png", scale=3)
        svg_kb = (out_dir / f"{stem}.svg").stat().st_size / 1024
        png_kb = (out_dir / f"{stem}.png").stat().st_size / 1024
        print(f"  {stem:<34} svg {svg_kb:6.1f} KB   png {png_kb:7.1f} KB")
    return len(figures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default=None,
        help="Render only figures whose name contains this, e.g. 'fig2'.",
    )
    parser.add_argument(
        "--variant",
        choices=[*VARIANTS, "both"],
        default="both",
        help="Which wording to draw. Default: both, so the two sets never drift apart.",
    )
    args = parser.parse_args()

    names = list(VARIANTS) if args.variant == "both" else [args.variant]
    base = mermaid_command()
    total = sum(render_variant(name, args.only, base) for name in names)

    if not total:
        print("\nNothing was rendered.", file=sys.stderr)
        return 1

    print(f"\nDone — {total} figure(s). Put the .svg files in the report where possible.")
    if args.variant == "both":
        print("The two wordings share file names, so swapping one set for the")
        print("other needs no change to your document's figure references.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
