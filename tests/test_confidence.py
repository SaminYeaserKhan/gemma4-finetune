"""Tests for answer log-probability extraction.

The whole confidence gate rests on one easily-botched index shift. A causal
language model's logits at position i predict the token at position i+1, so
scoring token i means reading logits[i-1]. Get it wrong and nothing crashes:
you get a full set of plausible negative numbers that describe the wrong
tokens, and an AUC that means nothing. Hence a test that fails loudly on the
unshifted version.
"""

import unittest

import torch

from thesis_pipeline.model_utils import token_logprobs


class TokenLogprobTests(unittest.TestCase):
    def setUp(self):
        # Vocabulary of 3. Position i puts all its mass on token i, so the
        # correctly-shifted lookup always scores a token the model considered
        # very unlikely (~-10) and the off-by-one version always scores one it
        # considered near-certain (~0). The two are impossible to confuse.
        self.logits = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]
        )
        self.input_ids = torch.tensor([[0, 1, 2]])

    def test_scores_each_token_from_the_preceding_position(self):
        scores = token_logprobs(self.logits, self.input_ids, start=1)
        self.assertEqual(len(scores), 2)
        for score in scores:
            self.assertAlmostEqual(score, -10.0, places=3)

    def test_start_index_selects_which_tokens_are_scored(self):
        # Only the answer should be scored; the prompt is given, not predicted.
        self.assertEqual(len(token_logprobs(self.logits, self.input_ids, start=2)), 1)

    def test_scoring_from_zero_is_rejected(self):
        # There is no logit preceding the first token, so its probability is
        # undefined. Returning a silent 0.0 would make an empty answer look
        # maximally confident.
        with self.assertRaises(ValueError):
            token_logprobs(self.logits, self.input_ids, start=0)


if __name__ == "__main__":
    unittest.main()
