import unittest

from thesis_pipeline.gsm8k import answers_match, extract_final_answer


class Gsm8kParsingTests(unittest.TestCase):
    def test_extracts_marker_answer(self):
        text = "Natalia sold 48+24 = <<48+24=72>>72 clips.\n#### 72"
        self.assertEqual(extract_final_answer(text), "72")

    def test_falls_back_to_last_number(self):
        text = "First 12, then 24, so the answer is 36."
        self.assertEqual(extract_final_answer(text), "36")

    def test_matches_equivalent_numbers(self):
        self.assertTrue(answers_match("72.0", "72"))
        self.assertTrue(answers_match("1/2", "0.5"))
        self.assertTrue(answers_match("$1,200", "1200"))

    def test_rejects_different_numbers(self):
        self.assertFalse(answers_match("73", "72"))


if __name__ == "__main__":
    unittest.main()

