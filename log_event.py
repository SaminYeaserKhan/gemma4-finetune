from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_pipeline.experiment_log import (
    append_experiment_event,
    environment_snapshot,
    git_snapshot,
)


def parse_key_value(items: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got: {item}")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a thesis experiment log entry.")
    parser.add_argument("--type", default="note", dest="event_type")
    parser.add_argument("--title", required=True)
    parser.add_argument("--detail", action="append", help="key=value detail. Can repeat.")
    parser.add_argument("--details-json", default=None)
    parser.add_argument("--include-env", action="store_true")
    parser.add_argument("--include-git", action="store_true")
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details = parse_key_value(args.detail)
    if args.details_json:
        details.update(json.loads(Path(args.details_json).read_text(encoding="utf-8")))
    if args.include_env:
        details["environment"] = environment_snapshot()
    if args.include_git:
        details["git"] = git_snapshot()
    event = append_experiment_event(
        event_type=args.event_type,
        title=args.title,
        details=details,
        reports_dir=args.reports_dir,
    )
    print(f"Logged event at {event['timestamp_utc']} -> {args.reports_dir}")


if __name__ == "__main__":
    main()

