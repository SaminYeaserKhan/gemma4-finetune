import unittest

from analyze_supervision import (
    approx_tokens,
    stack_voting,
    flip_report,
    gate_report,
    mcnemar_exact,
    significance_report,
    verifier_confusion,
)


def cascade_row(example_id, gold, first_correct, accepted, final):
    """A minimal cascade output row."""
    return {
        "id": example_id,
        "gold_final_answer": gold,
        "accepted_final_answer": final,
        "escalated": True,
        "attempts": [{"attempt": 1, "exact_correct": first_correct, "supervisor_accepted": accepted}],
    }


class McNemarTests(unittest.TestCase):
    def test_no_discordant_pairs_is_not_significant(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_all_changes_in_one_direction_is_significant(self):
        # 10 fixed, 0 broken: p = 2 * (1/2^10)
        self.assertAlmostEqual(mcnemar_exact(0, 10), 2 / 1024)

    def test_balanced_changes_are_not_significant(self):
        self.assertEqual(mcnemar_exact(5, 5), 1.0)

    def test_symmetric_in_its_arguments(self):
        self.assertEqual(mcnemar_exact(3, 11), mcnemar_exact(11, 3))

    def test_p_value_never_exceeds_one(self):
        for b in range(6):
            for c in range(6):
                self.assertLessEqual(mcnemar_exact(b, c), 1.0)


class VerifierConfusionTests(unittest.TestCase):
    def test_counts_the_four_cells(self):
        rows = [
            cascade_row(0, "72", first_correct=False, accepted=False, final="72"),  # true reject
            cascade_row(1, "72", first_correct=True, accepted=False, final="18"),   # false reject
            cascade_row(2, "72", first_correct=True, accepted=True, final="72"),    # true accept
            cascade_row(3, "72", first_correct=False, accepted=True, final="18"),   # false accept
        ]
        result = verifier_confusion("test", rows)
        self.assertEqual(result["true_reject"], 1)
        self.assertEqual(result["false_reject"], 1)
        self.assertEqual(result["true_accept"], 1)
        self.assertEqual(result["false_accept"], 1)
        self.assertEqual(result["judged"], 4)

    def test_unescalated_rows_are_not_judged(self):
        row = cascade_row(0, "72", first_correct=True, accepted=None, final="72")
        self.assertEqual(verifier_confusion("test", [row])["judged"], 0)


class FlipTests(unittest.TestCase):
    def test_separates_repairs_from_damage(self):
        rows = [
            cascade_row(0, "72", first_correct=False, accepted=False, final="72"),  # fixed
            cascade_row(1, "72", first_correct=True, accepted=False, final="18"),   # broken
            cascade_row(2, "72", first_correct=True, accepted=True, final="72"),    # held
            cascade_row(3, "72", first_correct=False, accepted=False, final="18"),  # still wrong
        ]
        result = flip_report("test", rows)
        self.assertEqual(result["wrong_to_right"], 1)
        self.assertEqual(result["right_to_wrong"], 1)
        self.assertEqual(result["net"], 0)
        self.assertEqual(result["stayed_right"], 1)
        self.assertEqual(result["stayed_wrong"], 1)


class SignificanceTests(unittest.TestCase):
    def test_pairs_against_the_baseline_by_id(self):
        baseline = [
            {"id": 0, "gold_final_answer": "72", "pred_final_answer": "18"},  # wrong
            {"id": 1, "gold_final_answer": "72", "pred_final_answer": "72"},  # right
        ]
        cascade = [
            cascade_row(0, "72", first_correct=False, accepted=False, final="72"),  # now right
            cascade_row(1, "72", first_correct=True, accepted=False, final="18"),   # now wrong
        ]
        result = significance_report("test", baseline, cascade)
        self.assertEqual(result["cascade_only_right"], 1)
        self.assertEqual(result["baseline_only_right"], 1)


def baseline_row(example_id, gold, predicted, tokens=50):
    return {
        "id": example_id,
        "gold_final_answer": gold,
        "pred_final_answer": predicted,
        "tokens_generated": tokens,
    }


class GateReportTests(unittest.TestCase):
    """The confidence gate has to survive being partly or wholly absent.

    Scoring is a separate offline pass, so the analysis routinely runs before
    it exists, or against a run that was interrupted halfway through.
    """

    def setUp(self):
        self.baseline = [
            baseline_row(0, "72", "72"),
            baseline_row(1, "72", "18"),
        ]

    def gates(self, rows):
        return {row["gate"] for row in rows}

    def test_confidence_gates_appear_when_scores_are_supplied(self):
        confidence = {
            0: {"mean_logprob": -0.1, "min_logprob": -1.0, "final_logprob": -0.01},
            1: {"mean_logprob": -0.9, "min_logprob": -4.0, "final_logprob": -0.50},
        }
        gates = self.gates(gate_report(self.baseline, {}, confidence))
        self.assertEqual(
            sum(1 for gate in gates if gate.startswith("confidence")), 3
        )

    def test_no_confidence_gates_without_a_score_file(self):
        gates = self.gates(gate_report(self.baseline, {}, {}))
        self.assertFalse(any(gate.startswith("confidence") for gate in gates))

    def test_a_measure_that_is_never_populated_is_dropped(self):
        # final_logprob is None whenever the final answer could not be located
        # in the text. If that happens for every row the gate is undefined, and
        # reporting an AUC over a column of identical infinities would look
        # like a real measurement of 0.5.
        confidence = {
            0: {"mean_logprob": -0.1, "min_logprob": -1.0, "final_logprob": None},
            1: {"mean_logprob": -0.9, "min_logprob": -4.0, "final_logprob": None},
        }
        gates = self.gates(gate_report(self.baseline, {}, confidence))
        self.assertFalse(any("final-answer" in gate for gate in gates))
        self.assertTrue(any("mean logprob" in gate for gate in gates))

    def test_the_combined_gate_appears_only_with_both_signals(self):
        confidence = {
            0: {"mean_logprob": -0.1, "min_logprob": -1.0, "final_logprob": -0.01},
            1: {"mean_logprob": -0.9, "min_logprob": -4.0, "final_logprob": -0.50},
        }
        bank = {0: ["72", "72"], 1: ["18", "5"]}
        with_both = self.gates(gate_report(self.baseline, bank, confidence))
        self.assertTrue(any("combined" in gate for gate in with_both))
        # Either signal alone is not the combined gate and must not be labelled
        # as one -- the whole claim is that it beats both of its parts.
        self.assertFalse(
            any("combined" in gate for gate in self.gates(gate_report(self.baseline, bank, {})))
        )
        self.assertFalse(
            any("combined" in gate for gate in self.gates(gate_report(self.baseline, {}, confidence)))
        )

    def test_rows_missing_from_the_score_file_are_escalated_first(self):
        # A half-finished scoring pass must not make unscored questions look
        # like the model's most confident ones.
        confidence = {0: {"mean_logprob": -0.1, "min_logprob": -1.0, "final_logprob": -0.01}}
        rows = gate_report(self.baseline, {}, confidence)
        mean_rows = [r for r in rows if "mean logprob" in r["gate"]]
        self.assertTrue(mean_rows)
        # id 1 is unscored, so at a 50% escalation rate it is the one sent up;
        # it is also the wrong answer, so the gate catches exactly one error.
        half = next(r for r in mean_rows if float(r["escalation_rate"]) == 0.5)
        self.assertEqual(half["errors_caught"], 1)


class ApproxTokenTests(unittest.TestCase):
    def test_empty_costs_nothing(self):
        self.assertEqual(approx_tokens(""), 0)
        self.assertEqual(approx_tokens(None), 0)

    def test_roughly_four_characters_per_token(self):
        self.assertEqual(approx_tokens("12345678"), 2)


class StackVotingTests(unittest.TestCase):
    """Vote first, escalate only what the vote could not settle.

    The disagreement gate already pays for k samples, so taking their majority
    on the questions it decides *not* to escalate costs the device nothing it
    has not already spent. These tests pin which answer wins where.
    """

    def row(self, example_id, gold, first, final, escalated):
        return {
            "id": example_id,
            "gold_final_answer": gold,
            "accepted_final_answer": final,
            "correct": final == gold,
            "escalated": escalated,
            "attempts": [
                {
                    "attempt": 1,
                    "prediction": f"#### {first}",
                    "pred_final_answer": first,
                    "exact_correct": first == gold,
                    "supervisor_accepted": None,
                }
            ],
        }

    def test_unescalated_row_takes_the_majority_vote(self):
        # Attempt 1 said 5 and lost the vote 2-1 to 72, which is right.
        rows = [self.row(0, "72", first="5", final="5", escalated=False)]
        stacked = stack_voting(rows, {0: ["72", "72"]})
        self.assertEqual(stacked[0]["accepted_final_answer"], "72")
        self.assertTrue(stacked[0]["correct"])

    def test_unescalated_row_keeps_attempt_one_when_it_wins_the_vote(self):
        rows = [self.row(0, "72", first="72", final="72", escalated=False)]
        stacked = stack_voting(rows, {0: ["72", "5"]})
        self.assertEqual(stacked[0]["accepted_final_answer"], "72")
        self.assertTrue(stacked[0]["correct"])

    def test_escalated_row_keeps_the_cascade_answer_over_the_vote(self):
        # The whole point of escalating is that the supervisor outranks the
        # vote on these. Here the vote would have said 5 and been wrong.
        rows = [self.row(0, "72", first="5", final="72", escalated=True)]
        stacked = stack_voting(rows, {0: ["5", "5"]})
        self.assertEqual(stacked[0]["accepted_final_answer"], "72")
        self.assertTrue(stacked[0]["correct"])

    def test_an_empty_bank_changes_nothing(self):
        rows = [self.row(0, "72", first="5", final="5", escalated=False)]
        self.assertEqual(stack_voting(rows, {}), rows)

    def test_the_original_rows_are_not_mutated(self):
        # The caller reports the un-stacked arm from the same list.
        rows = [self.row(0, "72", first="5", final="5", escalated=False)]
        stack_voting(rows, {0: ["72", "72"]})
        self.assertEqual(rows[0]["accepted_final_answer"], "5")

    def test_vote_is_numeric_not_string_equality(self):
        # 72.0 and $72 are the same answer; cluster_answers already knows that
        # and the stack must not regress to string matching.
        rows = [self.row(0, "72", first="5", final="5", escalated=False)]
        stacked = stack_voting(rows, {0: ["72.0", "$72"]})
        self.assertTrue(stacked[0]["correct"])


if __name__ == "__main__":
    unittest.main()
