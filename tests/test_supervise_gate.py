import unittest

from supervise import gate_scores


def attempt1(example_id, answer, tokens):
    return {
        "id": example_id,
        "pred_final_answer": answer,
        "prediction": f"#### {answer}",
        "tokens_generated": tokens,
    }


class GateScoresTests(unittest.TestCase):
    """Which on-device signal picks the questions that leave the device."""

    def setUp(self):
        self.ids = [0, 1]
        self.attempt1 = {0: attempt1(0, "72", 100), 1: attempt1(1, "18", 50)}
        # id 0: all three samples differ.  id 1: all three agree.
        self.bank = {0: ["5", "9"], 1: ["18", "18"]}
        self.confidence = {
            0: {"final_logprob": -0.01},   # confident
            1: {"final_logprob": -3.00},   # not confident
        }

    def test_none_escalates_everything_equally(self):
        self.assertEqual(gate_scores(self.ids, self.attempt1, "none", {}, {}), [1.0, 1.0])

    def test_length_uses_the_recorded_token_count(self):
        self.assertEqual(
            gate_scores(self.ids, self.attempt1, "length", {}, {}), [100.0, 50.0]
        )

    def test_disagreement_ranks_the_split_question_first(self):
        scores = gate_scores(self.ids, self.attempt1, "disagreement", self.bank, {})
        self.assertGreater(scores[0], scores[1])

    def test_disagreement_without_a_sample_bank_is_refused(self):
        with self.assertRaises(ValueError):
            gate_scores(self.ids, self.attempt1, "disagreement", {}, {})

    def test_combined_keeps_the_disagreement_ordering(self):
        # id 1 is the less confident of the two, but it is also the one whose
        # three samples agree. Agreement must still win: confidence only
        # breaks ties inside a disagreement group.
        scores = gate_scores(self.ids, self.attempt1, "combined", self.bank, self.confidence)
        self.assertGreater(scores[0], scores[1])

    def test_combined_breaks_ties_that_disagreement_cannot(self):
        # Both questions have all-different samples, so disagreement scores
        # them identically and cannot choose. Confidence must decide, and the
        # less confident answer is the one to send up.
        ids = [0, 1]
        rows = {0: attempt1(0, "72", 100), 1: attempt1(1, "18", 100)}
        bank = {0: ["5", "9"], 1: ["3", "7"]}
        flat = gate_scores(ids, rows, "disagreement", bank, {})
        self.assertEqual(flat[0], flat[1])
        scores = gate_scores(ids, rows, "combined", bank, self.confidence)
        self.assertGreater(scores[1], scores[0])

    def test_combined_without_confidence_is_refused(self):
        with self.assertRaises(ValueError):
            gate_scores(self.ids, self.attempt1, "combined", self.bank, {})

    def test_an_unknown_gate_is_refused(self):
        with self.assertRaises(ValueError):
            gate_scores(self.ids, self.attempt1, "nonsense", {}, {})
