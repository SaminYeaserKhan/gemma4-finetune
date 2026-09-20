import unittest

from thesis_pipeline.benchmark import (
    estimated_seconds,
    flips,
    judge_quality,
)


class EstimatedSecondsTests(unittest.TestCase):
    """Wall-clock per run, composed from measured component speeds.

    No full run recorded its own duration -- the timing fields reached the code
    after the last full arm finished -- so the benchmark composes time from speeds
    that *were* measured. These tests pin that arithmetic, because it is the one
    column in the table that is computed rather than counted.
    """

    def test_generation_time_is_tokens_over_speed(self):
        self.assertAlmostEqual(
            estimated_seconds(generated_tokens=880, judge_calls=0,
                              solver_tokens_per_second=8.8, judge_seconds=2.0),
            100.0,
        )

    def test_each_judgement_adds_its_measured_duration(self):
        self.assertAlmostEqual(
            estimated_seconds(generated_tokens=0, judge_calls=5,
                              solver_tokens_per_second=8.8, judge_seconds=2.0),
            10.0,
        )

    def test_the_two_parts_add(self):
        self.assertAlmostEqual(
            estimated_seconds(generated_tokens=88, judge_calls=3,
                              solver_tokens_per_second=8.8, judge_seconds=1.0),
            13.0,
        )

    def test_a_non_positive_speed_is_refused(self):
        # A zero here would divide by zero; a negative one would report negative
        # time. Both mean the speed was never measured, which must not look like
        # a number.
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                estimated_seconds(generated_tokens=10, judge_calls=0,
                                  solver_tokens_per_second=bad, judge_seconds=1.0)


class FlipsTests(unittest.TestCase):
    """Answers turned right and answers turned wrong, against a reference run.

    A net accuracy figure hides both. Retrying can break correct answers, and the
    thesis framing requires the two directions to be reported separately.
    """

    def test_counts_fixed_and_broken_separately(self):
        before = [False, True, False, True]
        after = [True, False, False, True]
        self.assertEqual(flips(before, after), (1, 1))

    def test_identical_runs_flip_nothing(self):
        self.assertEqual(flips([True, False], [True, False]), (0, 0))

    def test_mismatched_lengths_are_refused(self):
        # Silently zipping would drop questions and undercount both directions.
        with self.assertRaises(ValueError):
            flips([True], [True, False])


class JudgeQualityTests(unittest.TestCase):
    """How well a checker marks, from its verdicts against the real answers.

    A rejection is the checker saying "wrong". Recall is the share of truly wrong
    answers it rejected; the false-reject rate is the share of truly right answers
    it rejected. A verdict of None means the checker was never asked, and must not
    count as either.
    """

    def test_recall_and_false_reject_rate(self):
        # 4 wrong answers, 3 rejected; 4 right answers, 1 rejected.
        accepted = [False, False, False, True, True, True, True, False]
        correct = [False, False, False, False, True, True, True, True]
        q = judge_quality(accepted, correct)
        self.assertAlmostEqual(q["recall"], 0.75)
        self.assertAlmostEqual(q["false_reject_rate"], 0.25)
        self.assertAlmostEqual(q["precision"], 0.75)

    def test_unasked_questions_are_ignored(self):
        q = judge_quality([None, False, True], [False, False, True])
        self.assertEqual(q["judged"], 2)
        self.assertAlmostEqual(q["recall"], 1.0)

    def test_no_rejections_gives_zero_recall_not_a_crash(self):
        q = judge_quality([True, True], [False, True])
        self.assertEqual(q["recall"], 0.0)
        self.assertIsNone(q["precision"])


if __name__ == "__main__":
    unittest.main()
