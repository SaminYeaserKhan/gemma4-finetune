"""Content-addressed cache of supervisor verdicts.

Two things make this load-bearing rather than a mere optimisation.

**The arms must reject the same set.** L0, L1 and L2 differ only in how much of
the supervisor's reply is sent back to the model. Attempt 1 is byte-identical
across all three (it is read from the greedy FYDP 2 prediction file), so the
attempt-1 verdict must be identical too -- otherwise a judge that is not
perfectly deterministic would reject slightly different questions in each arm
and the comparison would be confounded by *which* questions were retried rather
than by hint content. Judging once and replaying makes that structural.

**Long runs must be free to resume.** A cascade arm is hours of GPU plus
hundreds of API calls. A power cut, an OOM, or a rate limit part-way through
should cost the time already spent and nothing more.

The key is the full judging context -- provider, model, system prompt, question
and the exact candidate text -- so changing the prompt or the verifier
invalidates the cache automatically instead of silently replaying stale
verdicts. That last point matters: a cache keyed only on the question would
happily hand attempt 2 the verdict for attempt 1.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .io_utils import append_jsonl
from .supervisor_client import SupervisorDecision

_FIELDS = ("accepted", "pointer", "correction", "raw_response", "provider",
           "input_tokens", "output_tokens", "error")


def verdict_key(
    provider: str,
    model: str,
    system_prompt: str,
    question: str,
    candidate_answer: str,
) -> str:
    digest = hashlib.sha256()
    # NUL separator: it cannot occur in any of the parts, so no combination of
    # values can collide by shifting the boundary between two fields.
    for part in (provider, model, system_prompt, question, candidate_answer):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class VerdictCache:
    """Append-only JSONL store, loaded into memory once at startup.

    Append-only rather than rewrite-on-save so that a process killed mid-write
    loses at most the last line; `read_jsonl` skips blanks and the loader
    tolerates a truncated final record.
    """

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.entries: dict[str, SupervisorDecision] = {}
        self.hits = 0
        self.misses = 0
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        for row in self._read_tolerantly():
            key = row.get("key")
            if not key:
                continue
            self.entries[key] = SupervisorDecision(
                accepted=bool(row.get("accepted")),
                pointer=row.get("pointer", ""),
                correction=row.get("correction", ""),
                raw_response=row.get("raw_response", ""),
                provider=row.get("provider", ""),
                input_tokens=int(row.get("input_tokens") or 0),
                output_tokens=int(row.get("output_tokens") or 0),
                error=row.get("error", ""),
            )

    def _read_tolerantly(self):
        """Yield records, skipping any the writer did not finish.

        Not `read_jsonl`, which raises: a prediction file with a broken line is
        a problem worth stopping for, whereas this file is append-only and its
        last line is expected to be torn whenever a run is killed. Losing one
        replayed verdict costs one extra call; refusing to load costs the whole
        cache.
        """
        assert self.path is not None
        import json

        with self.path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row

    def get(self, key: str) -> SupervisorDecision | None:
        decision = self.entries.get(key)
        # A cached error is not a cached verdict. Errors become accepts inside
        # `judge`, so replaying one would make a transient rate limit permanent
        # for that question -- exactly the silent-approval failure this design
        # is trying to eliminate.
        if decision is None or decision.error:
            self.misses += 1
            return None
        self.hits += 1
        return decision

    def put(self, key: str, decision: SupervisorDecision) -> None:
        # `--verdict-cache none` means re-judge everything, so it must not
        # replay from memory either -- that is the switch you reach for when
        # you suspect the cached verdicts themselves.
        if self.path is None or decision.error:
            return
        self.entries[key] = decision
        append_jsonl(self.path, {"key": key, **{f: getattr(decision, f) for f in _FIELDS}})

    def stats(self) -> dict[str, int]:
        return {"cached": len(self.entries), "hits": self.hits, "misses": self.misses}
