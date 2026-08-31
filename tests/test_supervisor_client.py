import json
import unittest
from unittest import mock

from thesis_pipeline.config import ThesisConfig
from thesis_pipeline.supervisor_client import (
    SupervisorAborted,
    SupervisorClient,
    SupervisorDecision,
    SupervisorError,
    _reject_truncated,
    parse_verdict,
)


class ParseVerdictTests(unittest.TestCase):
    def test_clean_json(self):
        accepted, pointer, correction = parse_verdict(
            '{"verdict": "NO", "pointer": "Step 1 is wrong.", "correction": "Halve it."}'
        )
        self.assertFalse(accepted)
        self.assertEqual(pointer, "Step 1 is wrong.")
        self.assertEqual(correction, "Halve it.")

    def test_markdown_fenced_json(self):
        accepted, pointer, _ = parse_verdict(
            '```json\n{"verdict": "NO", "pointer": "Bad step."}\n```'
        )
        self.assertFalse(accepted)
        self.assertEqual(pointer, "Bad step.")

    def test_json_embedded_in_prose(self):
        accepted, _, _ = parse_verdict(
            'Here is my verdict:\n{"verdict": "YES"}\nHope that helps!'
        )
        self.assertTrue(accepted)

    def test_bare_yes_no_fallback(self):
        self.assertTrue(parse_verdict("YES")[0])
        self.assertFalse(parse_verdict("NO, the second step is wrong")[0])

    def test_unparseable_reply_accepts(self):
        # A supervisor bug must not manufacture rejections; those would send
        # correct answers into a stochastic retry and corrupt the results.
        self.assertTrue(parse_verdict("")[0])
        self.assertTrue(parse_verdict("   ")[0])

    def test_strips_leaked_final_answer_from_hint(self):
        # A hint containing the answer would make the supervisor the solver.
        _, pointer, _ = parse_verdict(
            '{"verdict": "NO", "pointer": "Wrong. The answer is #### 3"}'
        )
        self.assertNotIn("####", pointer)
        self.assertNotIn("3", pointer.split("answer is")[-1])


class FeedbackLevelTests(unittest.TestCase):
    def setUp(self):
        self.decision = SupervisorDecision(
            accepted=False, pointer="Step 1 is wrong.", correction="Halve it."
        )

    def test_level_zero_sends_nothing(self):
        self.assertEqual(self.decision.hint(0), "")

    def test_level_one_sends_pointer_only(self):
        self.assertEqual(self.decision.hint(1), "Step 1 is wrong.")

    def test_level_two_sends_pointer_and_correction(self):
        self.assertEqual(self.decision.hint(2), "Step 1 is wrong. Halve it.")


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ThesisConfig()

    def test_exact_provider_uses_gold_answer(self):
        client = SupervisorClient(self.cfg, provider="exact")
        self.assertTrue(client.judge("q", "so the total is\n#### 72", "72").accepted)
        self.assertFalse(client.judge("q", "so the total is\n#### 18", "72").accepted)

    def test_none_provider_accepts_everything(self):
        client = SupervisorClient(self.cfg, provider="none")
        self.assertTrue(client.judge("q", "#### 18", "72").accepted)

    def test_missing_api_key_fails_at_construction(self):
        # Not at call time: `judge` turns API errors into accepts, so a lazily
        # detected missing key would yield a run that approves all 1,319
        # answers with no visible error.
        with mock.patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            with self.assertRaises(RuntimeError):
                SupervisorClient(self.cfg, provider="gemini")

    def test_local_provider_requires_a_runner(self):
        with self.assertRaises(ValueError):
            SupervisorClient(self.cfg, provider="local")

    def test_local_runner_is_used_and_tokens_recorded(self):
        def runner(system, user):
            return '{"verdict": "NO", "pointer": "Step 2."}', 300, 40

        client = SupervisorClient(self.cfg, provider="local", local_runner=runner)
        decision = client.judge("q", "#### 18", "72")
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.input_tokens, 300)
        self.assertEqual(decision.output_tokens, 40)

    def test_gold_answer_never_reaches_the_prompt(self):
        captured = {}

        def runner(system, user):
            captured["user"] = user
            return '{"verdict": "YES"}', 1, 1

        client = SupervisorClient(self.cfg, provider="local", local_runner=runner)
        client.judge("What is 6 times 12?", "6*12 = 72\n#### 72", gold_final_answer="72")
        self.assertNotIn("gold", captured["user"].lower())
        self.assertEqual(captured["user"].count("72"), 2)  # both from the candidate


class TruncationTests(unittest.TestCase):
    """A cut-off reply must never read as approval.

    `parse_verdict` accepts anything it cannot understand, on purpose -- the
    cascade must not manufacture rejections out of its own bugs. That makes
    truncation dangerous: Gemini 2.5 Flash spends its output budget on thinking
    tokens and returns empty text with finishReason MAX_TOKENS, which would
    otherwise be indistinguishable from a YES.
    """

    def test_empty_text_raises(self):
        with self.assertRaises(SupervisorError):
            _reject_truncated("", "STOP")

    def test_whitespace_only_raises(self):
        with self.assertRaises(SupervisorError):
            _reject_truncated("   \n ", "STOP")

    def test_max_tokens_finish_reason_raises_even_with_text(self):
        # Partial JSON parses as an accept via the bare-YES/NO fallback, so the
        # presence of text is not evidence the verdict survived.
        with self.assertRaises(SupervisorError):
            _reject_truncated('{"verdict": "N', "MAX_TOKENS")

    def test_openai_style_length_reason_raises(self):
        with self.assertRaises(SupervisorError):
            _reject_truncated("partial", "length")

    def test_complete_reply_passes(self):
        _reject_truncated('{"verdict": "YES"}', "STOP")
        _reject_truncated('{"verdict": "YES"}', "")


class ErrorAbortTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ThesisConfig(**{**ThesisConfig().__dict__, "supervisor_error_abort": 3})

    def _client(self, runner):
        return SupervisorClient(self.cfg, provider="local", local_runner=runner)

    def test_errors_below_the_limit_become_accepts(self):
        client = self._client(lambda s, u: (_ for _ in ()).throw(SupervisorError("boom")))
        decision = client.judge("q", "a", "1")
        self.assertTrue(decision.accepted)
        self.assertIn("boom", decision.error)

    def test_run_aborts_after_consecutive_failures(self):
        # Without this, a dead endpoint approves every remaining question and
        # the run exits reporting success on answers nothing ever judged.
        client = self._client(lambda s, u: (_ for _ in ()).throw(SupervisorError("boom")))
        client.judge("q", "a", "1")
        client.judge("q", "a", "1")
        with self.assertRaises(SupervisorAborted):
            client.judge("q", "a", "1")

    def test_a_success_resets_the_streak(self):
        calls = {"n": 0}

        def runner(system, user):
            calls["n"] += 1
            if calls["n"] == 3:
                return '{"verdict": "YES"}', 1, 1
            raise SupervisorError("boom")

        client = self._client(runner)
        for _ in range(5):
            client.judge("q", "a", "1")  # 2 fail, 1 succeeds, 2 fail -> no abort
        self.assertEqual(calls["n"], 5)

    def test_truncated_local_reply_counts_as_an_error(self):
        client = self._client(lambda s, u: ("", 10, 0))
        decision = client.judge("q", "a", "1")
        self.assertIn("empty response", decision.error)


class LlamaCppTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ThesisConfig()

    def _capture(self, response: dict):
        captured = {}

        def fake_post(self, url, headers, payload):
            captured["url"] = url
            captured["payload"] = payload
            return response

        return captured, fake_post

    def test_payload_disables_thinking_and_constrains_json(self):
        response = {
            "choices": [{"message": {"content": '{"verdict": "NO", "pointer": "Step 1."}'},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 25},
        }
        captured, fake_post = self._capture(response)
        with mock.patch.object(SupervisorClient, "_post", fake_post):
            client = SupervisorClient(self.cfg, provider="llamacpp")
            decision = client.judge("q", "candidate", "72")

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.input_tokens, 300)
        self.assertEqual(decision.output_tokens, 25)
        self.assertTrue(captured["url"].endswith("/v1/chat/completions"))
        # Thinking tokens are billed as supervisor output, which is the exact
        # quantity the thesis reports as cost.
        self.assertFalse(captured["payload"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(captured["payload"]["response_format"]["type"], "json_schema")

    def test_gold_answer_is_not_in_the_payload(self):
        response = {
            "choices": [{"message": {"content": '{"verdict": "YES"}'}, "finish_reason": "stop"}],
            "usage": {},
        }
        captured, fake_post = self._capture(response)
        with mock.patch.object(SupervisorClient, "_post", fake_post):
            client = SupervisorClient(self.cfg, provider="llamacpp")
            client.judge("What is 6 times 12?", "6*12 = 72\n#### 72", gold_final_answer="72")
        sent = json.dumps(captured["payload"])
        self.assertEqual(sent.count("72"), 2)  # both occurrences come from the candidate

    def test_truncated_completion_becomes_an_error_not_an_accept(self):
        response = {
            "choices": [{"message": {"content": '{"verdict": "N'}, "finish_reason": "length"}],
            "usage": {},
        }
        _, fake_post = self._capture(response)
        with mock.patch.object(SupervisorClient, "_post", fake_post):
            client = SupervisorClient(self.cfg, provider="llamacpp")
            decision = client.judge("q", "candidate", "72")
        self.assertNotEqual(decision.error, "")
        self.assertIn("truncated", decision.error)


if __name__ == "__main__":
    unittest.main()
