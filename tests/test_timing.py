import unittest

from thesis_pipeline.timing import Stopwatch


class FakeClock:
    """A clock the test drives by hand, so timings are exact, not flaky."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class StopwatchTests(unittest.TestCase):
    """Accumulated wall-clock across several spans.

    The cascade spends its time in three places -- generating locally, waiting
    on the supervisor, and everything else -- and a question can pass through
    the first two more than once. A plain start/stop pair cannot express that,
    which is why this accumulates.
    """

    def setUp(self):
        self.clock = FakeClock()

    def test_a_fresh_stopwatch_has_recorded_nothing(self):
        self.assertEqual(Stopwatch(self.clock).seconds, 0.0)

    def test_one_span_records_its_duration(self):
        watch = Stopwatch(self.clock)
        with watch:
            self.clock.advance(2.5)
        self.assertAlmostEqual(watch.seconds, 2.5)

    def test_spans_accumulate(self):
        watch = Stopwatch(self.clock)
        for span in (1.0, 2.0, 0.5):
            with watch:
                self.clock.advance(span)
        self.assertAlmostEqual(watch.seconds, 3.5)

    def test_time_outside_a_span_is_not_counted(self):
        watch = Stopwatch(self.clock)
        with watch:
            self.clock.advance(1.0)
        self.clock.advance(100.0)  # idle, or spent somewhere else
        self.assertAlmostEqual(watch.seconds, 1.0)

    def test_it_counts_the_spans(self):
        watch = Stopwatch(self.clock)
        for _ in range(3):
            with watch:
                self.clock.advance(1.0)
        self.assertEqual(watch.count, 3)

    def test_an_exception_still_records_the_span(self):
        # A supervisor call that raises still consumed real time, and a run
        # that under-reports its own cost is worse than one that fails loudly.
        watch = Stopwatch(self.clock)
        with self.assertRaises(RuntimeError):
            with watch:
                self.clock.advance(4.0)
                raise RuntimeError("boom")
        self.assertAlmostEqual(watch.seconds, 4.0)
        self.assertEqual(watch.count, 1)

    def test_mean_is_seconds_over_spans(self):
        watch = Stopwatch(self.clock)
        for span in (1.0, 3.0):
            with watch:
                self.clock.advance(span)
        self.assertAlmostEqual(watch.mean, 2.0)

    def test_mean_of_nothing_is_zero_not_a_crash(self):
        self.assertEqual(Stopwatch(self.clock).mean, 0.0)


if __name__ == "__main__":
    unittest.main()
