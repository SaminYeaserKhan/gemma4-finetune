import math
import unittest

from thesis_pipeline.gate import (
    auc,
    confidence_score,
    disagreement_score,
    escalation_order,
    gate_curve,
    majority_answer,
    select_for_escalation,
    tie_broken_score,
)


class ConfidenceGateTests(unittest.TestCase):
    def test_confident_answer_scores_lower_than_hesitant_one(self):
        # -0.05 per token is a near-certain answer; -1.2 is the model guessing.
        self.assertLess(confidence_score(-0.05), confidence_score(-1.2))

    def test_score_is_the_negated_log_probability(self):
        # The gate contract is "higher = less trustworthy" and log-probability
        # runs the other way, so the only transformation is a sign flip. No
        # rescaling: AUC and rank-based escalation are invariant to it, and a
        # squashing function would only obscure the raw measurement.
        self.assertAlmostEqual(confidence_score(-0.42), 0.42)

    def test_missing_log_probability_is_maximally_untrusted(self):
        # A row without the field was never scored. Escalating it is the safe
        # failure; treating it as confident would silently hide it from the
        # gate, which is how length_score's `or 0` default misbehaves.
        self.assertEqual(confidence_score(None), float("inf"))


class DisagreementGateTests(unittest.TestCase):
    def test_unanimous_samples_score_zero(self):
        self.assertEqual(disagreement_score(["72", "72", "72"]), 0.0)

    def test_agreement_is_numeric_not_textual(self):
        # $72, 72.0 and 72 are the same answer under the thesis metric, so the
        # gate must not treat them as disagreement.
        self.assertEqual(disagreement_score(["$72", "72.0", "72"]), 0.0)

    def test_all_different_scores_high(self):
        self.assertAlmostEqual(disagreement_score(["1", "2", "3"]), 2 / 3)

    def test_minority_dissent_scores_low(self):
        self.assertAlmostEqual(disagreement_score(["72", "72", "18"]), 1 / 3)

    def test_empty_is_maximally_untrusted(self):
        self.assertEqual(disagreement_score([]), 1.0)


class MajorityVoteTests(unittest.TestCase):
    def test_picks_modal_answer(self):
        self.assertEqual(majority_answer(["18", "72", "72"]), "72")

    def test_tie_breaks_to_first_sample(self):
        # Callers pass attempt 1 (greedy) first, so ties keep the deterministic
        # answer rather than a sampled one.
        self.assertEqual(majority_answer(["18", "72"]), "18")


class AucTests(unittest.TestCase):
    def test_perfect_separation(self):
        self.assertEqual(auc([1.0, 2.0, 3.0, 4.0], [False, False, True, True]), 1.0)

    def test_inverted_separation(self):
        self.assertEqual(auc([4.0, 3.0, 2.0, 1.0], [False, False, True, True]), 0.0)

    def test_all_ties_is_a_coin_flip(self):
        self.assertEqual(auc([1.0, 1.0, 1.0, 1.0], [True, False, True, False]), 0.5)

    def test_undefined_without_both_classes(self):
        self.assertTrue(math.isnan(auc([1.0, 2.0], [False, False])))


class EscalationTests(unittest.TestCase):
    def test_selects_worst_scoring_fraction(self):
        self.assertEqual(select_for_escalation([0.1, 0.9, 0.5, 0.7], 0.5), {1, 3})

    def test_zero_rate_escalates_nothing(self):
        self.assertEqual(select_for_escalation([0.1, 0.9], 0.0), set())

    def test_rate_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            select_for_escalation([0.1], 1.5)


class GateCurveTests(unittest.TestCase):
    def test_full_escalation_catches_every_error(self):
        rows = gate_curve([0.1, 0.9, 0.5], [False, True, True], rates=(1.0,))
        self.assertEqual(rows[0]["recall"], 1.0)
        self.assertEqual(rows[0]["ceiling_accuracy"], 1.0)

    def test_no_escalation_leaves_baseline_accuracy(self):
        rows = gate_curve([0.1, 0.9, 0.5], [False, True, True], rates=(0.0,))
        self.assertEqual(rows[0]["errors_caught"], 0)
        self.assertAlmostEqual(rows[0]["ceiling_accuracy"], 1 / 3)

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            gate_curve([0.1, 0.2], [True])


if __name__ == "__main__":
    unittest.main()


class TieBrokenScoreTests(unittest.TestCase):
    """Rank by one signal, break ties with a second.

    Measured on the FYDP 3 test split: agreement between samples is the
    stronger signal (AUC 0.840) and confidence the weaker (0.715), but
    agreement takes only three values with k=3, so it cannot rank inside its
    own groups. Letting confidence break those ties reaches 0.869. Letting
    confidence *override* agreement scores worse than either, which is why
    this is strictly lexicographic and has no weight to tune.
    """

    def test_ties_in_the_primary_are_broken_by_the_secondary(self):
        combined = tie_broken_score([1.0, 1.0], [0.0, 5.0])
        self.assertLess(combined[0], combined[1])

    def test_the_primary_ordering_is_never_overridden(self):
        # Secondary disagrees as hard as it can; primary must still win.
        combined = tie_broken_score([0.0, 1.0], [999.0, -999.0])
        self.assertLess(combined[0], combined[1])

    def test_a_constant_secondary_leaves_the_ranking_alone(self):
        primary = [0.0, 1.0, 1.0, 2.0]
        combined = tie_broken_score(primary, [7.0] * 4)
        self.assertEqual(escalation_order(combined), escalation_order(primary))

    def test_an_all_equal_primary_hands_the_decision_to_the_secondary(self):
        combined = tie_broken_score([1.0] * 3, [5.0, 1.0, 3.0])
        self.assertEqual(escalation_order(combined), [0, 2, 1])

    def test_unscored_rows_sort_first_within_their_group(self):
        # confidence_score returns inf for a row that was never scored, and a
        # half-finished scoring pass must not bury those questions.
        combined = tie_broken_score([1.0, 1.0], [2.0, float("inf")])
        self.assertLess(combined[0], combined[1])

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            tie_broken_score([1.0, 2.0], [1.0])

    def test_empty_input(self):
        self.assertEqual(tie_broken_score([], []), [])
