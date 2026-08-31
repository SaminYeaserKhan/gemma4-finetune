"""The supervisor: an external judge that reviews the local model's reasoning.

One call returns everything the three feedback arms need:

    {"verdict": "NO",
     "pointer": "Step 1 misreads 'half that much'.",
     "correction": "It means half of the 2 bolts of blue fiber, not double it."}

L0 uses the verdict alone, L1 adds the pointer, L2 adds the correction. Asking
once and stripping fields per arm means all three arms judge the same outputs
and reject the same set, so hint content is the only variable between them --
and it costs a third of the API calls. The per-level token costs reported in
the paper are measured by tokenizing each field separately (see
`analyze_supervision.py`), which `--validate-levels` cross-checks against
genuinely separate per-level calls.

Providers, cheapest first:
- `exact`     deterministic oracle using the gold answer. Free, and the ceiling.
- `self`      the fine-tuned model critiques itself. Free, and the floor.
- `local`     a larger model on the same GPU via transformers. Free, offline.
- `llamacpp`  a GGUF model behind `llama-server`. Free, offline, and the
              primary supervisor: quantised MoE weights that transformers +
              bitsandbytes cannot host, swapped by restarting the server rather
              than by changing code, with JSON-schema-constrained decoding.
- `gemini`    free API tier, kept as a small cloud reference arm.
- `openai` / `anthropic`  paid.

Every network path treats an empty or truncated reply as an **error**, never as
an acceptance. `judge` converts errors into accepts on purpose -- a dead API
must not look like a supervisor that hates the model -- which means anything
ambiguous that reaches `parse_verdict` silently becomes approval. Truncation is
the likely failure here, because thinking tokens count against the output
budget, so it is caught at the provider boundary where the finish reason is
still visible.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

import requests

from .config import ThesisConfig
from .gsm8k import answers_match, extract_final_answer


# The rejection standard is spelled out as a closed list, and the things that
# must NOT trigger a rejection are named explicitly, because leaving the bar to
# the judge's discretion does not work. An earlier version asked it to reject
# any step that was "wrong, unjustified, or misreads the question"; GLM-4.7-Flash
# read that hyper-literally and confabulated objections with no content --
# "the four apples cost $1.50 x 4 = $6.00, not $6", or a pointer that quoted the
# model's own correct line back as the error. Measured over 150 test questions,
# that wording rejected 45 of 82 correct answers (false-reject rate 0.549); this
# wording rejects 23 (0.280), while still catching 59 of 68 wrong ones. Since
# every rejection sends a correct answer into a stochastic retry that may break
# it, false rejections are the expensive kind of mistake here.
SYSTEM_PROMPT = (
    "You are a strict verifier for grade-school math solutions.\n\n"
    "You are given a question and a candidate solution containing step-by-step "
    "reasoning and a final answer.\n\n"
    "Work the problem yourself first. Then compare, step by step.\n\n"
    "Answer NO only if the solution has a REAL defect:\n"
    "- it uses a number that does not come from the question, or the wrong number;\n"
    "- it applies the wrong operation;\n"
    "- its arithmetic produces the wrong value;\n"
    "- it misreads what the question asks for.\n\n"
    "Answer YES if every step produces the right value and follows from the "
    "question. In particular do NOT answer NO because of:\n"
    "- formatting, notation, units, or trailing zeros ($6 and $6.00 are the same);\n"
    "- a step being brief, restated, or explained in a way you would not have chosen;\n"
    "- an objection that would not change any number in the solution.\n\n"
    "If you cannot name a specific step whose VALUE is wrong, answer YES.\n\n"
    "Reply with JSON and nothing else:\n"
    '{"verdict": "YES or NO", '
    '"pointer": "one short sentence naming the first step that is wrong, '
    'empty string if YES", '
    '"correction": "one short sentence stating the correct interpretation, '
    'empty string if YES"}\n\n'
    "Never state the final numeric answer in pointer or correction. Point at "
    "the mistake; let the student redo the arithmetic."
)

# A hint that leaks the answer would make the supervisor the solver and the
# local model a transcriber, which is exactly the criticism this design has to
# survive. Strip any answer marker the supervisor emits anyway.
_MARKER_RE = re.compile(r"####\s*[^\n\r]*")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# Finish reasons meaning "the model ran out of output budget". Whatever text
# came back is a prefix, so the JSON is unclosed and would parse as an accept.
_TRUNCATED = {"MAX_TOKENS", "length", "max_tokens", "model_length"}

# Phrases Google uses when the *daily* quota is gone rather than the per-minute
# one. Backing off cannot fix that, so it is worth naming in the error.
_DAILY_QUOTA_MARKERS = ("PerDay", "per day", "RESOURCE_EXHAUSTED")

# Constrained decoding: llama.cpp compiles this into a grammar, so the reply is
# valid JSON with these keys by construction and the tolerant fallbacks in
# `parse_verdict` become unreachable on the local path.
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["YES", "NO"]},
        "pointer": {"type": "string"},
        "correction": {"type": "string"},
    },
    "required": ["verdict", "pointer", "correction"],
    "additionalProperties": False,
}

_KEY_BY_PROVIDER = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@dataclass
class SupervisorDecision:
    accepted: bool
    pointer: str = ""
    correction: str = ""
    raw_response: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    def hint(self, level: int) -> str:
        """The feedback text this arm is allowed to send back to the model."""
        if level <= 0:
            return ""
        if level == 1:
            return self.pointer
        return " ".join(part for part in (self.pointer, self.correction) if part)


class SupervisorError(RuntimeError):
    """One call failed. `judge` absorbs this into an accept."""


class SupervisorAborted(RuntimeError):
    """Too many calls failed in a row -- stop the run.

    Deliberately *not* a `SupervisorError`, so it propagates out of `judge`
    instead of being absorbed. A dead quota or a stopped llama-server would
    otherwise mark every remaining question "approved" and the run would exit
    reporting success on results that were never judged.
    """


class SupervisorClient:
    def __init__(
        self,
        cfg: ThesisConfig,
        provider: str | None = None,
        local_runner: Callable[[str, str], tuple[str, int, int]] | None = None,
        model: str | None = None,
        base_url: str | None = None,
        rpm: float | None = None,
    ):
        """`local_runner(system, user) -> (text, input_tokens, output_tokens)`

        Injected rather than imported so this module stays testable without a
        GPU, and so `local` and `self` can share one code path with different
        models behind it.
        """
        self.cfg = cfg
        self.provider = (provider or cfg.supervisor_provider).lower()
        self.local_runner = local_runner
        self.model = model or _default_model(cfg, self.provider)
        self.base_url = (base_url or cfg.llamacpp_base_url).rstrip("/")
        # Free tiers rate-limit per minute. Pacing under the limit is cheaper
        # than provoking 429s and backing off through them.
        self._min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._last_call = 0.0
        self._consecutive_errors = 0
        if self.provider in {"local", "self"} and local_runner is None:
            raise ValueError(
                f"provider '{self.provider}' requires a local_runner "
                "(see supervise.py:build_local_runner)"
            )
        # Fail now, not per-call. `judge` deliberately swallows API errors into
        # an accept, so a missing key checked lazily would produce a run where
        # the supervisor approves all 1,319 answers and nothing looks wrong.
        required_key = _KEY_BY_PROVIDER.get(self.provider)
        if required_key and not os.getenv(required_key):
            raise RuntimeError(
                f"{required_key} is required when the supervisor provider is "
                f"'{self.provider}'. Set it before running."
            )

    def judge(
        self,
        question: str,
        candidate_answer: str,
        gold_final_answer: str | None = None,
    ) -> SupervisorDecision:
        """Judge one candidate solution.

        `gold_final_answer` is accepted only by the `exact` oracle. Every other
        provider is built a prompt that cannot see it -- `_prompt` takes no
        gold argument at all, so there is no path by which a real supervisor
        could be marking its own homework.
        """
        if self.provider == "none":
            return SupervisorDecision(True, raw_response="YES", provider=self.provider)
        if self.provider == "exact":
            predicted = extract_final_answer(candidate_answer)
            accepted = answers_match(predicted, gold_final_answer)
            return SupervisorDecision(
                accepted,
                pointer="" if accepted else "The final answer is incorrect.",
                raw_response="YES" if accepted else "NO",
                provider=self.provider,
            )

        user_prompt = self._prompt(question, candidate_answer)
        try:
            if self.provider in {"local", "self"}:
                text, in_tok, out_tok = self.local_runner(SYSTEM_PROMPT, user_prompt)
                _reject_truncated(text, "")
            elif self.provider == "llamacpp":
                text, in_tok, out_tok = self._call_llamacpp(user_prompt)
            elif self.provider == "gemini":
                text, in_tok, out_tok = self._call_gemini(user_prompt)
            elif self.provider == "openai":
                text, in_tok, out_tok = self._call_openai(user_prompt)
            elif self.provider == "anthropic":
                text, in_tok, out_tok = self._call_anthropic(user_prompt)
            else:
                raise ValueError(f"Unsupported supervisor provider: {self.provider}")
        except SupervisorError as exc:
            # A dead API must not silently reject every answer, which would
            # look like a supervisor that hates the model. Accept and record --
            # but only up to a point, because a *permanently* dead API would
            # then approve every remaining question without complaint.
            self._consecutive_errors += 1
            limit = self.cfg.supervisor_error_abort
            if limit and self._consecutive_errors >= limit:
                raise SupervisorAborted(
                    f"{self._consecutive_errors} consecutive supervisor failures "
                    f"({self.provider}/{self.model}); last error: {exc}. Rows already "
                    "written are intact -- fix the cause and rerun with --resume."
                ) from exc
            return SupervisorDecision(
                True, provider=self.provider, error=str(exc), raw_response=""
            )

        self._consecutive_errors = 0
        accepted, pointer, correction = parse_verdict(text)
        return SupervisorDecision(
            accepted=accepted,
            pointer=pointer,
            correction=correction,
            raw_response=text,
            provider=self.provider,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    def _prompt(self, question: str, candidate_answer: str) -> str:
        return (
            f"Question:\n{question.strip()}\n\n"
            f"Candidate solution:\n{candidate_answer.strip()}\n\n"
            "Verify the reasoning step by step, then reply with the JSON object."
        )

    # -- providers ---------------------------------------------------------

    def _call_llamacpp(self, user_prompt: str) -> tuple[str, int, int]:
        """A GGUF model behind `llama-server` (OpenAI-compatible endpoint).

        `enable_thinking: false` matters for cost, not just speed: the thesis
        metric is supervisor output tokens, and a verifier that reasons for
        several hundred tokens per judgment breaks the premise that feedback is
        a ~12-60 token hint. Run the thinking-on ablation separately.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": self.cfg.supervisor_max_output_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": _VERDICT_SCHEMA, "strict": True},
            },
        }
        data = self._post(f"{self.base_url}/v1/chat/completions", {}, payload)
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        _reject_truncated(text, choice.get("finish_reason", ""))
        usage = data.get("usage", {})
        return (
            text.strip(),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    def _call_gemini(self, user_prompt: str) -> tuple[str, int, int]:
        api_key = _require_key("GEMINI_API_KEY", "gemini")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": self.cfg.supervisor_max_output_tokens,
                "responseMimeType": "application/json",
                # Gemini 2.5 Flash thinks by default and thinking tokens are
                # charged against maxOutputTokens, so without this the budget is
                # spent before any JSON is emitted and the reply comes back
                # empty with finishReason MAX_TOKENS. Only 2.5 accepts a budget
                # of 0; the 3.x line cannot disable thinking, which is why the
                # config pins 2.5 Flash.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        data = self._post(url, {"x-goog-api-key": api_key}, payload)
        candidate = (data.get("candidates") or [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        text = " ".join(part.get("text", "") for part in parts).strip()
        _reject_truncated(text, candidate.get("finishReason", ""))
        usage = data.get("usageMetadata", {})
        return text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)

    def _call_openai(self, user_prompt: str) -> tuple[str, int, int]:
        api_key = _require_key("OPENAI_API_KEY", "openai")
        payload = {
            "model": self.cfg.openai_model,
            "instructions": SYSTEM_PROMPT,
            "input": user_prompt,
            "max_output_tokens": self.cfg.supervisor_max_output_tokens,
            "reasoning": {"effort": "minimal"},
        }
        data = self._post(
            "https://api.openai.com/v1/responses",
            {"Authorization": f"Bearer {api_key}"},
            payload,
        )
        usage = data.get("usage", {})
        text = _extract_openai_text(data)
        reason = (data.get("incomplete_details") or {}).get("reason", "")
        _reject_truncated(text, "length" if reason == "max_output_tokens" else "")
        return (
            text,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

    def _call_anthropic(self, user_prompt: str) -> tuple[str, int, int]:
        api_key = _require_key("ANTHROPIC_API_KEY", "anthropic")
        payload = {
            "model": self.cfg.anthropic_model,
            "max_tokens": self.cfg.supervisor_max_output_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            payload,
        )
        text = " ".join(
            part.get("text", "")
            for part in data.get("content", [])
            if part.get("type") == "text"
        ).strip()
        _reject_truncated(text, data.get("stop_reason", ""))
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        """POST with exponential backoff.

        Free tiers rate-limit and overnight runs are long; a single 429 must
        not end a five-hour job.
        """
        headers = {"Content-Type": "application/json", **headers}
        last_error = ""
        for attempt in range(self.cfg.supervisor_max_retries):
            self._throttle()
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.cfg.supervisor_timeout,
                )
                if response.status_code in _RETRYABLE_STATUS:
                    last_error = f"HTTP {response.status_code}: {response.text[:400]}"
                    # A per-minute limit clears by waiting. A per-day one does
                    # not, and burning the retry budget against it just delays
                    # the abort by a few minutes per question.
                    if response.status_code == 429 and _is_daily_quota(response.text):
                        raise SupervisorError(
                            "daily quota exhausted -- resume after the quota "
                            f"resets (midnight Pacific): {response.text[:200]}"
                        )
                elif not response.ok:
                    raise SupervisorError(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                else:
                    return response.json()
            except requests.RequestException as exc:
                last_error = repr(exc)
            sleep_for = self.cfg.supervisor_backoff_seconds * (2**attempt)
            time.sleep(sleep_for + random.uniform(0, 0.5))  # jitter
        raise SupervisorError(
            f"giving up after {self.cfg.supervisor_max_retries} attempts: {last_error}"
        )

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()


def parse_verdict(text: str) -> tuple[bool, str, str]:
    """Pull (accepted, pointer, correction) out of a supervisor reply.

    Models wrap JSON in prose or markdown fences even when told not to, so this
    degrades gracefully: strict JSON, then the first brace-delimited block,
    then a bare YES/NO. An unparseable reply is treated as acceptance, matching
    the API-failure path -- the cascade should never reject on its own bugs.
    """
    if not text or not text.strip():
        return True, "", ""

    payload = None
    stripped = _FENCE_RE.sub("", text.strip())
    for candidate in (stripped, _first_json_block(stripped)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break

    if payload is None:
        return _looks_like_yes(stripped), "", ""

    verdict = str(payload.get("verdict", "")).strip()
    accepted = _looks_like_yes(verdict) if verdict else _looks_like_yes(stripped)
    return (
        accepted,
        _clean_hint(payload.get("pointer")),
        _clean_hint(payload.get("correction")),
    )


def _reject_truncated(text: str, finish_reason: str) -> None:
    """Raise unless the reply is a complete one.

    The whole point: `parse_verdict` treats anything it cannot read as an
    accept, so a reply cut off mid-JSON is indistinguishable from approval.
    Caught here, where the finish reason is still in scope, it becomes an error
    that the run can count and abort on.
    """
    if finish_reason in _TRUNCATED:
        raise SupervisorError(
            f"response truncated (finish_reason={finish_reason}); raise "
            "SUPERVISOR_MAX_OUTPUT_TOKENS or disable the model's thinking mode"
        )
    if not text or not text.strip():
        raise SupervisorError("empty response from supervisor")


def _is_daily_quota(body: str) -> bool:
    return any(marker in body for marker in _DAILY_QUOTA_MARKERS)


def _default_model(cfg: ThesisConfig, provider: str) -> str:
    """The exact model string, recorded per run.

    Free-tier hosted models get swapped without notice, and a GGUF file name
    pins a quantisation as well as a checkpoint, so this string is what makes a
    verifier result reproducible.
    """
    return {
        "gemini": cfg.gemini_model,
        "openai": cfg.openai_model,
        "anthropic": cfg.anthropic_model,
        "local": cfg.local_supervisor_model,
        "llamacpp": cfg.llamacpp_model,
        "self": f"{cfg.model_name}+adapter",
    }.get(provider, provider)


def _first_json_block(text: str) -> str | None:
    match = _JSON_RE.search(text)
    return match.group(0) if match else None


def _clean_hint(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return _MARKER_RE.sub("", value).strip()


def _looks_like_yes(text: str) -> bool:
    return text.strip().upper().startswith("YES")


def _require_key(env_var: str, provider: str) -> str:
    key = os.getenv(env_var)
    if not key:
        # RuntimeError, not SupervisorError: `judge` swallows the latter into
        # an accept, and a config mistake must never masquerade as approval.
        raise RuntimeError(
            f"{env_var} is required when SUPERVISOR_PROVIDER={provider}."
        )
    return key


def _extract_openai_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()
    parts: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text") or content.get("output_text")
            if text:
                parts.append(text)
    return " ".join(parts).strip()
