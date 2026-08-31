from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    # utf-8-sig, not utf-8: these files get inspected and truncated with
    # PowerShell during long runs, and Set-Content -Encoding utf8 prepends a
    # BOM that would otherwise break the first line on resume.
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def repair_jsonl(path: str | Path) -> int:
    """Drop unreadable lines from an append-only file. Returns how many.

    Long runs append one row per question over many hours. Losing power
    partway through an append leaves a half-written final line -- or, because
    NTFS can extend a file's length before its data reaches disk, a run of NUL
    bytes. Both make `read_jsonl` raise, which turns a recoverable
    interruption into a crash on the resume path that exists to recover from
    it.

    The damaged row is discarded rather than reconstructed: that question was
    never finished, so the next run simply answers it again.

    The file is rewritten rather than the bad line merely being skipped on
    read, because the run goes on appending to it and `analyze_supervision.py`
    reads the result strictly. Tolerating the line would leave it buried in
    the middle of the file to fail hours later. A healthy file is not
    rewritten at all -- the rewrite is itself a window in which power can be
    lost.
    """
    path = Path(path)
    if not path.exists():
        return 0

    good: list[str] = []
    dropped = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            # A NUL byte is never legitimate content here; it is the signature
            # of a length-extended but unflushed write. Count it as damage so
            # the file is cleaned and the loss is reported, unlike a blank
            # line, which is harmless and silently ignored.
            if "\x00" in line:
                dropped += 1
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                dropped += 1
                continue
            good.append(stripped)

    if dropped:
        # Write beside the original and swap, so an interruption during the
        # repair cannot leave the file shorter than what it is replacing.
        temporary = path.with_suffix(path.suffix + ".repair")
        with temporary.open("w", encoding="utf-8") as handle:
            for line in good:
                handle.write(line + "\n")
        temporary.replace(path)
    return dropped


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

