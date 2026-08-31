import tempfile
import unittest
from pathlib import Path

from thesis_pipeline.supervisor_client import SupervisorDecision
from thesis_pipeline.verdict_cache import VerdictCache, verdict_key


ARGS = ("gemini", "gemini-2.5-flash", "SYSTEM", "What is 2+2?", "2+2=4\n#### 4")


class VerdictKeyTests(unittest.TestCase):
    def test_same_inputs_same_key(self):
        self.assertEqual(verdict_key(*ARGS), verdict_key(*ARGS))

    def test_every_field_changes_the_key(self):
        base = verdict_key(*ARGS)
        for index in range(len(ARGS)):
            altered = list(ARGS)
            altered[index] += "x"
            self.assertNotEqual(base, verdict_key(*altered), f"field {index} ignored")

    def test_candidate_answer_is_part_of_the_key(self):
        # The reason this matters: attempt 2 asks about a different candidate
        # for the same question. Keying on the question alone would replay
        # attempt 1's verdict and the retry would never be judged.
        first = verdict_key("p", "m", "s", "q", "answer one")
        second = verdict_key("p", "m", "s", "q", "answer two")
        self.assertNotEqual(first, second)

    def test_field_boundaries_cannot_be_shifted(self):
        # "ab"+"c" must not hash the same as "a"+"bc".
        self.assertNotEqual(
            verdict_key("ab", "c", "s", "q", "a"),
            verdict_key("a", "bc", "s", "q", "a"),
        )


class VerdictCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "verdicts.jsonl"
        self.addCleanup(self.tmp.cleanup)

    def test_roundtrip_through_disk(self):
        cache = VerdictCache(self.path)
        decision = SupervisorDecision(
            accepted=False, pointer="Step 1.", correction="Halve it.",
            raw_response="{}", provider="gemini", input_tokens=300, output_tokens=40,
        )
        cache.put("k", decision)

        reloaded = VerdictCache(self.path).get("k")
        self.assertIsNotNone(reloaded)
        self.assertFalse(reloaded.accepted)
        self.assertEqual(reloaded.pointer, "Step 1.")
        self.assertEqual(reloaded.correction, "Halve it.")
        self.assertEqual(reloaded.input_tokens, 300)
        self.assertEqual(reloaded.output_tokens, 40)

    def test_miss_returns_none_and_counts(self):
        cache = VerdictCache(self.path)
        self.assertIsNone(cache.get("absent"))
        self.assertEqual(cache.stats()["misses"], 1)
        self.assertEqual(cache.stats()["hits"], 0)

    def test_errors_are_not_cached(self):
        # An error becomes an accept inside `judge`. Persisting it would make a
        # transient rate limit into a permanent approval for that question.
        cache = VerdictCache(self.path)
        cache.put("k", SupervisorDecision(True, error="HTTP 429", provider="gemini"))
        self.assertIsNone(cache.get("k"))
        self.assertIsNone(VerdictCache(self.path).get("k"))

    def test_disabled_cache_never_hits_and_writes_nothing(self):
        cache = VerdictCache(None)
        cache.put("k", SupervisorDecision(True))
        self.assertIsNone(cache.get("k"))
        self.assertFalse(self.path.exists())

    def test_survives_a_truncated_final_line(self):
        # Long runs get killed by power cuts; the file is append-only so a
        # partial last record must not poison everything written before it.
        cache = VerdictCache(self.path)
        cache.put("a", SupervisorDecision(False, pointer="one"))
        cache.put("b", SupervisorDecision(False, pointer="two"))
        text = self.path.read_text(encoding="utf-8")
        self.path.write_text(text[: -len(text.splitlines()[-1]) // 2], encoding="utf-8")

        reloaded = VerdictCache(self.path)
        self.assertIsNotNone(reloaded.get("a"))


if __name__ == "__main__":
    unittest.main()
