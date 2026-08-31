"""Wall-clock accounting for the cascade.

The thesis argues about *time to answer* on hardware without a serious GPU, but
every run until now recorded only tokens and used them as a proxy. Tokens are a
good proxy for cloud cost and a poor one for latency: a local token and a remote
token cost the device completely different amounts of waiting, and the whole
point of the gate is to trade one for the other.

A question can generate locally more than once (attempt, then retry) and can
call the supervisor more than once, so a single start/stop pair cannot describe
where its time went. `Stopwatch` accumulates across spans instead, and counts
them, so a run can report both totals and per-call averages.

The clock is injectable so the tests can drive it by hand rather than sleeping.
`time.monotonic` is the default deliberately: the wall clock can step backwards
over an NTP correction, and a negative duration in the results would be
indistinguishable from a bug.
"""

from __future__ import annotations

import time
from types import TracebackType


class Stopwatch:
    """Total wall-clock spent inside its `with` blocks, and how many there were."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._started: float | None = None
        self.seconds: float = 0.0
        self.count: int = 0

    def __enter__(self) -> "Stopwatch":
        self._started = self._clock()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Recorded even when the block raised: a supervisor call that failed
        # still made the device wait, and a run that under-reports its own cost
        # is worse than one that fails loudly.
        if self._started is not None:
            self.seconds += self._clock() - self._started
            self.count += 1
            self._started = None

    @property
    def mean(self) -> float:
        """Seconds per span. Zero rather than an error when nothing ran."""
        return self.seconds / self.count if self.count else 0.0
