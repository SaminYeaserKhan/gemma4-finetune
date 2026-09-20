import unittest
from pathlib import Path

from thesis_pipeline.gsm8k import (
    GEMMA,
    QWEN,
    build_fewshot_prompt,
    build_prompt,
    build_retry_prompt,
    build_training_text,
    family_for,
)
from thesis_pipeline.model_utils import trim_at_end_tag
from supervise import resolve_adapter_dir


class FamilyLookupTests(unittest.TestCase):
    """Which chat template and which loader class a base model needs.

    The solver was Gemma-only until we re-ran the cascade on Qwen to show the
    gate transfers. Getting this wrong is the worst kind of bug for a thesis:
    the wrong chat tags still generate fluent text, still parse, and still
    produce a plausible-looking accuracy -- just a meaningless one. So an
    unrecognised model is refused rather than defaulted.
    """

    def test_gemma_uses_start_of_turn_tags(self):
        self.assertEqual(family_for("google/gemma-4-E2B-it"), GEMMA)

    def test_qwen_uses_chatml(self):
        self.assertEqual(family_for("Qwen/Qwen2.5-1.5B-Instruct"), QWEN)

    def test_lookup_ignores_case_and_org_prefix(self):
        self.assertEqual(family_for("qwen/QWEN2.5-3B-INSTRUCT"), QWEN)

    def test_an_unknown_model_is_refused_not_guessed(self):
        # Silently defaulting would emit the wrong chat tags and yield a
        # fluent, parseable, entirely meaningless run.
        with self.assertRaises(ValueError):
            family_for("mistralai/Mistral-7B-Instruct-v0.3")

    def test_gemma_is_multimodal_and_qwen_is_not(self):
        # Decides AutoModelForImageTextToText vs AutoModelForCausalLM.
        self.assertTrue(GEMMA.multimodal)
        self.assertFalse(QWEN.multimodal)

    def test_families_have_distinct_end_tags(self):
        self.assertNotEqual(GEMMA.end_tag, QWEN.end_tag)


class PromptBuildingTests(unittest.TestCase):
    """The prompt text each family expects, and that the default is unchanged."""

    def test_default_prompt_is_still_gemma(self):
        # Regression guard: every existing result was generated through this
        # path, so the default must not move.
        self.assertEqual(
            build_prompt("2+2?"),
            "<start_of_turn>user\n2+2?\n<end_of_turn>\n<start_of_turn>model\n",
        )

    def test_qwen_prompt_uses_chatml_tags(self):
        self.assertEqual(
            build_prompt("2+2?", QWEN),
            "<|im_start|>user\n2+2?\n<|im_end|>\n<|im_start|>assistant\n",
        )

    def test_training_text_closes_with_the_family_end_tag(self):
        self.assertTrue(build_training_text("q", "a", QWEN).endswith("<|im_end|>"))

    def test_fewshot_prompt_repeats_the_family_tags(self):
        prompt = build_fewshot_prompt("q3", [("q1", "a1"), ("q2", "a2")], QWEN)
        self.assertEqual(prompt.count("<|im_start|>user"), 3)
        self.assertNotIn("<start_of_turn>", prompt)

    def test_retry_prompt_honours_the_family(self):
        # The retry is the cascade's whole output on escalated questions. If it
        # alone kept Gemma tags, the escalated subset would silently degrade
        # while the un-escalated majority looked fine.
        prompt = build_retry_prompt("q", "wrong", 1, "hint", QWEN)
        self.assertTrue(prompt.startswith("<|im_start|>user"))
        self.assertNotIn("<start_of_turn>", prompt)

    def test_fewshot_prompt_ends_ready_for_the_model_to_speak(self):
        prompt = build_fewshot_prompt("q2", [("q1", "a1")], QWEN)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))


class TrimAtEndTagTests(unittest.TestCase):
    """Cutting the model's turn marker off a decoded answer.

    Generation decodes with `skip_special_tokens=False`, because the stop tag
    has to be visible to be cut. Leaving it in would put `<|im_end|>` inside
    the stored prediction, where the last-number fallback in
    `extract_final_answer` would happily read digits out of whatever followed.
    """

    def test_it_cuts_at_the_gemma_tag(self):
        self.assertEqual(
            trim_at_end_tag("The answer is 7. #### 7<end_of_turn>", "<end_of_turn>"),
            "The answer is 7. #### 7",
        )

    def test_it_cuts_at_the_qwen_tag(self):
        self.assertEqual(
            trim_at_end_tag("#### 7<|im_end|>", "<|im_end|>"), "#### 7"
        )

    def test_text_without_the_tag_is_returned_stripped(self):
        self.assertEqual(trim_at_end_tag("  #### 7  ", "<|im_end|>"), "#### 7")

    def test_it_cuts_at_the_first_tag_not_the_last(self):
        # A model that keeps talking past its turn must not have that text
        # folded into the answer.
        self.assertEqual(
            trim_at_end_tag("#### 7<|im_end|>junk<|im_end|>", "<|im_end|>"), "#### 7"
        )

    def test_the_wrong_family_tag_does_not_cut(self):
        # Guards the failure this whole abstraction exists to prevent.
        self.assertEqual(
            trim_at_end_tag("#### 7<|im_end|>", "<end_of_turn>"), "#### 7<|im_end|>"
        )


class ResolveAdapterDirTests(unittest.TestCase):
    """Choosing the LoRA adapter, including choosing none at all.

    `--adapter-dir ""` falls through to the configured default, which is the
    Gemma adapter. Running a different base model needs a way to say "no
    adapter" that cannot be confused with "not specified" -- otherwise a Qwen
    run silently tries to load Gemma weights.
    """

    def test_an_explicit_path_wins(self):
        self.assertEqual(
            resolve_adapter_dir("my-adapter", False, Path("default")),
            Path("my-adapter"),
        )

    def test_unspecified_falls_back_to_the_default(self):
        self.assertEqual(
            resolve_adapter_dir(None, False, Path("default")), Path("default")
        )

    def test_no_adapter_means_none_not_the_default(self):
        self.assertIsNone(resolve_adapter_dir(None, True, Path("default")))

    def test_no_adapter_overrides_an_explicit_path(self):
        # Asking for both is a contradiction; refusing to load anything is the
        # safe reading, since the alternative loads weights the user disowned.
        self.assertIsNone(resolve_adapter_dir("my-adapter", True, Path("default")))


if __name__ == "__main__":
    unittest.main()
