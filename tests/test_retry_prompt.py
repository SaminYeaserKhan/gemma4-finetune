import unittest

from thesis_pipeline.gsm8k import MODEL_TAG, build_retry_prompt


QUESTION = "A robe takes 2 bolts of blue fiber and half that much white fiber."
PREVIOUS = "White is 2*2=4 bolts. Total 2+4=6.\n#### 6"
HINT = "Step 1 misreads 'half that much'."


class RetryPromptTests(unittest.TestCase):
    def test_level_zero_sends_no_hint(self):
        prompt = build_retry_prompt(QUESTION, PREVIOUS, level=0, hint=HINT)
        self.assertNotIn(HINT, prompt)
        self.assertIn("A verifier rejected that attempt.", prompt)

    def test_level_one_sends_the_hint(self):
        prompt = build_retry_prompt(QUESTION, PREVIOUS, level=1, hint=HINT)
        self.assertIn(HINT, prompt)

    def test_hint_is_optional_at_hinted_levels(self):
        # The supervisor sometimes returns an empty pointer even on a NO.
        prompt = build_retry_prompt(QUESTION, PREVIOUS, level=2, hint="")
        self.assertIn("Solve the problem again", prompt)

    def test_carries_question_and_previous_attempt(self):
        prompt = build_retry_prompt(QUESTION, PREVIOUS, level=1, hint=HINT)
        self.assertIn(QUESTION, prompt)
        self.assertIn("#### 6", prompt)

    def test_ends_ready_for_the_model_turn(self):
        prompt = build_retry_prompt(QUESTION, PREVIOUS, level=0)
        self.assertTrue(prompt.endswith(MODEL_TAG))

    def test_invalid_level_rejected(self):
        with self.assertRaises(ValueError):
            build_retry_prompt(QUESTION, PREVIOUS, level=3)


if __name__ == "__main__":
    unittest.main()
