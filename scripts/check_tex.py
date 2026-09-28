"""Static checks for a LaTeX source we cannot compile locally.

Covers the failures that actually break an Overleaf build: undefined citations,
missing graphics, unbalanced environments and braces, and unescaped specials.
"""
import re
import sys
from pathlib import Path

root = Path(r"C:\thesis\gemma4-finetune\paper")
tex = (root / "main.tex").read_text(encoding="utf-8")
bib = (root / "references.bib").read_text(encoding="utf-8")

problems = []

# 1. citations
keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
cited = set()
for group in re.findall(r"\\cite\{([^}]+)\}", tex):
    cited.update(k.strip() for k in group.split(","))
undefined = sorted(cited - keys)
unused = sorted(keys - cited)
print(f"bib entries: {len(keys)}   cited: {len(cited)}")
if undefined:
    problems.append(f"UNDEFINED CITATIONS: {undefined}")
print(f"  undefined: {undefined or 'none'}")
print(f"  in bib but not cited: {unused or 'none'}")

# 2. graphics
graphicspath = root / "figures"
missing = []
for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
    if not (graphicspath / g).exists():
        stem = Path(g).stem
        if not any((graphicspath / f"{stem}{e}").exists() for e in (".pdf", ".png", ".jpg", ".eps")):
            missing.append(g)
n_graphics = len(re.findall(r"\\includegraphics", tex))
print(f"graphics referenced: {n_graphics}, missing: {missing or 'none'}")
if missing:
    problems.append(f"MISSING GRAPHICS: {missing}")

# 3. environments
begins = re.findall(r"\\begin\{([^}]+)\}", tex)
ends = re.findall(r"\\end\{([^}]+)\}", tex)
from collections import Counter
b, e = Counter(begins), Counter(ends)
for env in set(b) | set(e):
    if b[env] != e[env]:
        problems.append(f"UNBALANCED ENV {env}: {b[env]} begin vs {e[env]} end")
print(f"environments: {len(begins)} begin / {len(ends)} end -- "
      f"{'balanced' if b == e else 'UNBALANCED'}")

# 4. braces. Escaped specials must go FIRST: a literal \% would otherwise look
#    like the start of a comment and swallow the rest of the line, including
#    its closing braces.
stripped = re.sub(r"\\[{}%&#_$]", "", tex)
stripped = re.sub(r"(?m)%.*$", "", stripped)     # real comments
if stripped.count("{") != stripped.count("}"):
    problems.append(f"BRACES: {stripped.count('{')} open vs {stripped.count('}')} close")
print(f"braces: {stripped.count('{')} open / {stripped.count('}')} close")

# 5. unescaped '#'. Legal in two places: \verb, and #1 argument placeholders
#    inside \newcommand. Anything else would break the build.
# \verb must go first: the document legitimately contains \verb|%|, and
# stripping comments before it would eat the closing delimiter and desync
# every later \verb on the line.
body = re.sub(r"\\verb\|[^|]*\|", "", tex)
body = re.sub(r"\\[{}%&#_$]", "", body)
body = re.sub(r"(?m)%.*$", "", body)
body = re.sub(r"\\newcommand.*", "", body)
hits = [m.start() for m in re.finditer(r"#", body)]
if hits:
    problems.append(f"UNESCAPED #: {len(hits)} occurrence(s) outside verb/newcommand")
print(f"unescaped '#' outside verb/newcommand: {len(hits)}")

# 6. table column counts
for m in re.finditer(r"\\begin\{tabular\}\{@?\{?\}?([^}]*)\}", tex):
    spec = re.sub(r"@\{[^}]*\}", "", m.group(1))
    cols = len(re.findall(r"[lcrp]", spec))
    block = tex[m.end():tex.find(r"\end{tabular}", m.end())]
    for row in block.split(r"\\"):
        row = re.sub(r"\\(toprule|midrule|bottomrule)", "", row).strip()
        if not row or row.startswith("%"):
            continue
        n = len(re.split(r"(?<!\\)&", row))
        if n != cols:
            problems.append(f"TABLE ROW has {n} cells, spec says {cols}: {row[:60]}")

print()
if problems:
    print(f"{len(problems)} PROBLEM(S):")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("All static checks passed.")
