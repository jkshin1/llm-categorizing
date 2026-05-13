from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_categorizing.config import LLMSettings
from llm_categorizing.models import FinalClassificationResult, Stage1Result
from llm_categorizing.prompts import (
    SYSTEM_PROMPT,
    correction_prompt,
    employee_context,
    stage1_user_prompt,
    stage2_user_prompt,
)
from llm_categorizing.taxonomy import TAXONOMY_COLUMNS, Taxonomy, normalize_cell


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response does not contain a JSON object")

    payload = stripped[start : end + 1]
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object")
    return parsed


@dataclass
class ClassificationConfig:
    include_team_in_prompt: bool = True
    max_review_chars: int = 12000
    max_candidates_per_prompt: int = 300
    validation_attempts: int = 2
    api_retry_attempts: int = 3
    confidence_review_threshold: float = 0.6


class JsonlCache:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._items: dict[str, dict[str, Any]] = {}
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    key = item.get("cache_key")
                    value = item.get("value")
                    if isinstance(key, str) and isinstance(value, dict):
                        self._items[key] = value

    def get(self, key: str) -> dict[str, Any] | None:
        value = self._items.get(key)
        return dict(value) if value else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._items[key] = dict(value)
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"cache_key": key, "value": value}, ensure_ascii=False))
            handle.write("\n")


class OpenAICompatibleJobClassifier:
    def __init__(
        self,
        *,
        settings: LLMSettings,
        taxonomy: Taxonomy,
        config: ClassificationConfig,
        cache: JsonlCache | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed. Run: pip install -r requirements.txt") from exc

        self.settings = settings
        self.taxonomy = taxonomy
        self.config = config
        self.cache = cache
        self.client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
        )

        pair_count = len(self.taxonomy.pairs())
        if pair_count > config.max_candidates_per_prompt:
            raise ValueError(
                f"Too many 중직무/소직무 candidates for one prompt: {pair_count}. "
                f"Increase --max-candidates-per-prompt or reduce taxonomy scope."
            )

    def classify_row(self, row: dict[str, Any]) -> dict[str, Any]:
        review = normalize_cell(row.get("self_review", ""))
        if not review:
            return self._review_required_result("self_review is blank")

        truncated = len(review) > self.config.max_review_chars
        review_for_prompt = review[: self.config.max_review_chars] if truncated else review
        context_json = employee_context(
            year=normalize_cell(row.get("year", "")),
            team=normalize_cell(row.get("team", "")),
            self_review=review_for_prompt,
            include_team=self.config.include_team_in_prompt,
            input_truncated=truncated,
        )

        cache_key = self._cache_key(context_json)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return dict(cached)

        try:
            result = self._classify_uncached(context_json)
        except Exception as exc:
            result = self._review_required_result(f"classification_error: {exc}")

        result["input_truncated"] = truncated
        result["taxonomy_version"] = self.taxonomy.version_hash()
        result["model_name"] = self.settings.model
        result["classified_at"] = datetime.now(timezone.utc).isoformat()

        if self.cache and not result.get("error"):
            self.cache.set(cache_key, result)
        return result

    def _classify_uncached(self, context_json: str) -> dict[str, Any]:
        stage1 = self._run_stage1(context_json)
        pair = self.taxonomy.canonical_pair(stage1.major_job, stage1.sub_job)
        if pair is None:
            return self._review_required_result("stage1 result is not in taxonomy")

        candidates = self.taxonomy.children_for_pair(pair["중직무"], pair["소직무"])
        if len(candidates) > self.config.max_candidates_per_prompt:
            return self._review_required_result(
                f"too many final candidates under selected pair: {len(candidates)}"
            )

        final = self._run_stage2(context_json, candidates)
        candidate = {
            "중직무": final.major_job,
            "소직무": final.sub_job,
            "Device": final.device,
            "단위 직무": final.unit_job,
            "세부 직무1": final.detail_job_1,
            "세부 직무2": final.detail_job_2,
        }
        canonical = self._canonical_from_candidates(candidate, candidates)
        if canonical is None:
            return self._review_required_result("final result is not in taxonomy")

        needs_review = final.needs_review or final.confidence < self.config.confidence_review_threshold
        return {
            **canonical,
            "confidence": final.confidence,
            "reason": final.reason,
            "needs_review": needs_review,
            "error": "",
        }

    def _run_stage1(self, context_json: str) -> Stage1Result:
        candidates_json = self.taxonomy.format_candidates_json(self.taxonomy.pairs())
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": stage1_user_prompt(context_json, candidates_json)},
        ]

        last_output = ""
        last_error = ""
        for attempt_index in range(self.config.validation_attempts):
            if attempt_index:
                messages.append(
                    {"role": "user", "content": correction_prompt(last_error, last_output)}
                )
            last_output = self._chat_json_text(messages)
            try:
                parsed = extract_json_object(last_output)
                result = Stage1Result.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                continue

            if self.taxonomy.canonical_pair(result.major_job, result.sub_job):
                return result
            last_error = "중직무/소직무 pair is not in candidate list"

        raise ValueError(f"stage1 validation failed: {last_error}")

    def _run_stage2(self, context_json: str, candidates: list[dict[str, str]]) -> FinalClassificationResult:
        candidates_json = self.taxonomy.format_candidates_json(candidates)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": stage2_user_prompt(context_json, candidates_json)},
        ]

        last_output = ""
        last_error = ""
        for attempt_index in range(self.config.validation_attempts):
            if attempt_index:
                messages.append(
                    {"role": "user", "content": correction_prompt(last_error, last_output)}
                )
            last_output = self._chat_json_text(messages)
            try:
                parsed = extract_json_object(last_output)
                result = FinalClassificationResult.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                continue

            candidate = {
                "중직무": result.major_job,
                "소직무": result.sub_job,
                "Device": result.device,
                "단위 직무": result.unit_job,
                "세부 직무1": result.detail_job_1,
                "세부 직무2": result.detail_job_2,
            }
            if self._canonical_from_candidates(candidate, candidates):
                return result
            last_error = "final hierarchy path is not in candidate list"

        raise ValueError(f"stage2 validation failed: {last_error}")

    def _canonical_from_candidates(
        self,
        candidate: dict[str, object],
        candidates: list[dict[str, str]],
    ) -> dict[str, str] | None:
        wanted = {
            column: normalize_cell(candidate.get(column, "")).casefold()
            for column in TAXONOMY_COLUMNS
        }
        for row in candidates:
            row_key = {column: normalize_cell(row.get(column, "")).casefold() for column in TAXONOMY_COLUMNS}
            if row_key == wanted:
                return dict(row)
        return None

    def _chat_json_text(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if self.settings.use_json_response_format:
            payload["response_format"] = {"type": "json_object"}

        for attempt in Retrying(
            stop=stop_after_attempt(self.config.api_retry_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=12),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                completion = self.client.chat.completions.create(**payload)
                content = completion.choices[0].message.content
                if not content:
                    raise ValueError("empty LLM response")
                return content
        raise RuntimeError("unreachable retry state")

    def _cache_key(self, context_json: str) -> str:
        payload = {
            "context": context_json,
            "taxonomy_version": self.taxonomy.version_hash(),
            "model": self.settings.model,
            "include_team": self.config.include_team_in_prompt,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _review_required_result(self, reason: str) -> dict[str, Any]:
        return {
            **{column: "" for column in TAXONOMY_COLUMNS},
            "confidence": 0.0,
            "reason": "",
            "needs_review": True,
            "error": reason,
        }
