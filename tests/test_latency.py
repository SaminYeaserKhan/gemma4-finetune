import unittest

from experiments.benchmark_latency import latency_profile


class LatencyProfileTests(unittest.TestCase):
    """Turning measured component times into what a user actually waits.

    The components are measured once each; the interesting number is how they
    compose under a given escalation rate, because that is the trade the whole
    thesis is about. Getting this arithmetic wrong would misreport the headline
    latency claim, so it is pinned here rather than written inline.
    """

    def profile(self, **kw):
        base = dict(
            seconds_per_generation=1.0,
            seconds_per_judgement=2.0,
            samples=3,
            escalation_rate=0.5,
            rejection_rate=1.0,
        )
        return latency_profile(**{**base, **kw})

    def test_the_local_path_is_just_the_samples(self):
        # Three generations, nothing else: no checker, no retry.
        self.assertAlmostEqual(self.profile()["seconds_local_path"], 3.0)

    def test_a_single_sample_needs_no_vote(self):
        self.assertAlmostEqual(
            self.profile(samples=1)["seconds_local_path"], 1.0
        )

    def test_the_escalated_path_adds_a_judgement_and_a_retry(self):
        # 3 generations + 1 judgement + 1 retry generation = 3 + 2 + 1
        self.assertAlmostEqual(self.profile()["seconds_escalated_path"], 6.0)

    def test_an_accepted_answer_skips_the_retry(self):
        # Nothing is rejected, so the checker is paid for but no retry happens.
        self.assertAlmostEqual(
            self.profile(rejection_rate=0.0)["seconds_escalated_path"], 5.0
        )

    def test_average_is_the_two_paths_weighted_by_escalation_rate(self):
        # half at 3.0, half at 6.0
        self.assertAlmostEqual(self.profile()["seconds_average"], 4.5)

    def test_escalating_nothing_costs_the_local_path(self):
        p = self.profile(escalation_rate=0.0)
        self.assertAlmostEqual(p["seconds_average"], p["seconds_local_path"])

    def test_escalating_everything_costs_the_escalated_path(self):
        p = self.profile(escalation_rate=1.0)
        self.assertAlmostEqual(p["seconds_average"], p["seconds_escalated_path"])

    def test_partial_rejection_scales_only_the_retry(self):
        # 3 + 2 + 0.25 * 1
        self.assertAlmostEqual(
            self.profile(rejection_rate=0.25)["seconds_escalated_path"], 5.25
        )

    def test_an_impossible_escalation_rate_is_refused(self):
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                self.profile(escalation_rate=bad)

    def test_an_impossible_rejection_rate_is_refused(self):
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                self.profile(rejection_rate=bad)


if __name__ == "__main__":
    unittest.main()
