from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def append_experiment_event(
    event_type: str,
    title: str,
    details: dict[str, Any] | None = None,
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp_utc": utc_now(),
        "event_type": event_type,
        "title": title,
        "details": details or {},
    }

    jsonl_path = reports_path / "experiment_log.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    markdown_path = reports_path / "experiment_log.md"
    with markdown_path.open("a", encoding="utf-8") as handle:
        handle.write(f"## {event['timestamp_utc']} - {title}\n\n")
        handle.write(f"- Type: `{event_type}`\n")
        for key, value in event["details"].items():
            handle.write(f"- {key}: `{format_detail(value)}`\n")
        handle.write("\n")

    return event


def format_detail(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def command_string(args: list[str] | tuple[str, ...]) -> str:
    return " ".join(args)


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = torch.cuda.is_available()
        snapshot["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            snapshot["gpu"] = torch.cuda.get_device_name(0)
            snapshot["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        snapshot["torch_error"] = repr(exc)
    return snapshot


def git_snapshot() -> dict[str, str]:
    root = Path.cwd()
    if not (root / ".git").exists():
        return {"git": "not_a_git_repository"}
    result: dict[str, str] = {}
    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "status": ["git", "status", "--short"],
    }
    for key, cmd in commands.items():
        try:
            completed = subprocess.run(
                cmd,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            result[key] = completed.stdout.strip()
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            result[f"{key}_error"] = repr(exc)
    return result

