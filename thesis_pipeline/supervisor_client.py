from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from .config import ThesisConfig
from .gsm8k import answers_match, extract_final_answer


SYSTEM_PROMPT = (
    "You are a strict math-answer verifier. Decide whether the candidate "
    "answer correctly solves the question. Return exactly YES or NO. Do not "
    "include explanation."
)


@dataclass
class SupervisorDecision:
    accepted: bool
    raw_response: str
    provider: str


class SupervisorClient:
    def __init__(self, cfg: ThesisConfig, provider: str | None = None):
        self.cfg = cfg
        self.provider = (provider or cfg.supervisor_provider).lower()

    def judge(
        self,
        question: str,
        candidate_answer: str,
        gold_final_answer: str | None = None,
    ) -> SupervisorDecision:
        if self.provider == "none":
            return SupervisorDecision(True, "YES", self.provider)
        if self.provider == "exact":
            pred = extract_final_answer(candidate_answer)
            accepted = answers_match(pred, gold_final_answer)
            return SupervisorDecision(accepted, "YES" if accepted else "NO", self.provider)
        if self.provider == "openai":
            return self._judge_openai(question, candidate_answer)
        if self.provider == "anthropic":
            return self._judge_anthropic(question, candidate_answer)
        raise ValueError(f"Unsupported supervisor provider: {self.provider}")

    def _prompt(self, question: str, candidate_answer: str) -> str:
        return (
            f"Question:\n{question.strip()}\n\n"
            f"Candidate answer:\n{candidate_answer.strip()}\n\n"
            "Is the candidate answer correct? Return exactly YES or NO."
        )

    def _judge_openai(self, question: str, candidate_answer: str) -> SupervisorDecision:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when SUPERVISOR_PROVIDER=openai.")
        payload = {
            "model": self.cfg.openai_model,
            "instructions": SYSTEM_PROMPT,
            "input": self._prompt(question, candidate_answer),
            "max_output_tokens": 12,
            "reasoning": {"effort": "minimal"},
        }
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        text = _extract_openai_text(response.json())
        return SupervisorDecision(_is_yes(text), text, self.provider)

    def _judge_anthropic(self, question: str, candidate_answer: str) -> SupervisorDecision:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required when SUPERVISOR_PROVIDER=anthropic."
            )
        payload = {
            "model": self.cfg.anthropic_model,
            "max_tokens": 12,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": self._prompt(question, candidate_answer)}],
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        text_parts = [
            part.get("text", "")
            for part in data.get("content", [])
            if part.get("type") == "text"
        ]
        text = " ".join(text_parts).strip()
        return SupervisorDecision(_is_yes(text), text, self.provider)


def _is_yes(text: str) -> bool:
    return text.strip().upper().startswith("YES")


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

