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
from llm_categorizing.diagnosis import DiagnosisContext, empty_diagnosis_output_payload
from llm_categorizing.models import FinalClassificationResult, Stage1Result
from llm_categorizing.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    correction_prompt,
    employee_context,
    stage1_user_prompt,
    stage2_user_prompt,
)
from llm_categorizing.taxonomy import TAXONOMY_COLUMNS, Taxonomy, normalize_cell, normalize_key


PROCESS_EVIDENCE_TERMS = [
    "Process Qual",
    "Process Flow",
    "Base Line",
    "Baseline",
    "Scheme",
    "Via First",
    "Via Last",
    "Low-k",
    "IMD",
    "PLR",
    "Reticle",
    "MTS",
    "단위공정",
    "공정 조건",
    "공정 평가",
    "공정 Tuning",
    "MLM Module",
]

DEVICE_CONTEXT_TERMS = ["DRAM", "NAND", "Logic", "Cell", "Lucy Base Line"]
DIAGNOSIS_PROCESS_TERMS = ["공정", "Process", "Etch", "CLN", "CVD", "PVD", "Photo", "CMP", "Diff", "IMP"]


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
    include_team_in_prompt: bool = False
    max_review_chars: int = 12000
    max_candidates_per_prompt: int = 300
    validation_attempts: int = 2
    api_retry_attempts: int = 5
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

    def classify_row(
        self,
        row: dict[str, Any],
        diagnosis_context: DiagnosisContext | None = None,
    ) -> dict[str, Any]:
        review = normalize_cell(row.get("self_review", ""))
        if not review and not diagnosis_context:
            return self._review_required_result("self_review is blank")

        truncated = len(review) > self.config.max_review_chars
        review_for_prompt = review[: self.config.max_review_chars] if truncated else review
        classification_hints = self._build_classification_hints(review_for_prompt, diagnosis_context)
        context_json = employee_context(
            year=normalize_cell(row.get("year", "")),
            team=normalize_cell(row.get("team", "")),
            self_review=review_for_prompt,
            include_team=self.config.include_team_in_prompt,
            input_truncated=truncated,
            classification_hints=classification_hints,
            diagnosis_context=diagnosis_context.to_prompt_payload() if diagnosis_context else None,
        )

        cache_key = self._cache_key(context_json)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return dict(cached)

        try:
            result = self._classify_uncached(context_json, review_for_prompt, diagnosis_context)
        except Exception as exc:
            result = self._review_required_result(f"classification_error: {exc}")

        result.update(
            diagnosis_context.to_output_payload()
            if diagnosis_context
            else empty_diagnosis_output_payload()
        )
        result["input_truncated"] = truncated
        result["taxonomy_version"] = self.taxonomy.version_hash()
        result["model_name"] = self.settings.model
        result["classified_at"] = datetime.now(timezone.utc).isoformat()

        if self.cache and not result.get("error"):
            self.cache.set(cache_key, result)
        return result

    def _classify_uncached(
        self,
        context_json: str,
        review: str,
        diagnosis_context: DiagnosisContext | None,
    ) -> dict[str, Any]:
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

        canonical, guardrail_reason = self._apply_diagnosis_guardrail(canonical, diagnosis_context)
        if not guardrail_reason:
            canonical, guardrail_reason = self._apply_process_guardrail(canonical, review)
        ambiguity_reason = self.taxonomy.ambiguity_reason_for_row(canonical)
        needs_review = (
            final.needs_review
            or final.confidence < self.config.confidence_review_threshold
            or bool(ambiguity_reason)
        )
        reason = final.reason
        if guardrail_reason:
            reason = f"{reason} | {guardrail_reason}" if reason else guardrail_reason
        return {
            **canonical,
            "confidence": final.confidence,
            "reason": reason,
            "needs_review": needs_review,
            "ambiguity_reason": ambiguity_reason,
            "guardrail_reason": guardrail_reason,
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
        candidates_json = self.taxonomy.format_candidates_json(
            self.taxonomy.annotate_candidates(candidates)
        )
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

    def _build_classification_hints(
        self,
        review: str,
        diagnosis_context: DiagnosisContext | None = None,
    ) -> list[str]:
        hints: list[str] = []
        review_key = review.casefold()

        process_terms = self._matched_terms(review, PROCESS_EVIDENCE_TERMS)
        if process_terms:
            hints.append(
                "공정 수행 근거 키워드 감지: "
                + ", ".join(process_terms[:12])
                + ". 중직무 판단 시 제품명보다 이 업무 단서를 우선 검토."
            )

        device_terms = self._matched_terms(review, DEVICE_CONTEXT_TERMS)
        if device_terms:
            hints.append(
                "제품/Device/라인 용어 감지: "
                + ", ".join(device_terms[:8])
                + ". 이 용어만으로 중직무를 소자 또는 공정으로 확정하지 말 것."
            )

        ambiguous_unit_jobs = self.taxonomy.ambiguous_unit_jobs_in_text(review)
        for item in ambiguous_unit_jobs[:5]:
            major_jobs = ", ".join(item["중복 중직무"])
            hints.append(
                f"단위 직무 '{item['단위 직무']}'은 taxonomy에서 여러 중직무({major_jobs})에 중복 존재. "
                "self_review의 실제 업무 동사와 산출물로 구분."
            )

        if "mlm" in review_key and process_terms:
            hints.append(
                "MLM이 Process Qual/Base Line/Scheme/단위공정/Process Flow 같은 표현과 함께 나오면 "
                "DRAM 등 제품명이 있어도 공정 후보를 우선 검토."
            )

        if diagnosis_context:
            hints.extend(self._build_diagnosis_hints(diagnosis_context))

        return hints

    def _build_diagnosis_hints(self, diagnosis_context: DiagnosisContext) -> list[str]:
        hints: list[str] = []
        diagnosis_text = self._diagnosis_text(diagnosis_context)

        process_terms = self._matched_terms(diagnosis_text, DIAGNOSIS_PROCESS_TERMS)
        if process_terms:
            hints.append(
                "진단 데이터에서 공정 계열 단서 감지: "
                + ", ".join(process_terms[:12])
                + ". self_review와 충돌하지 않으면 중직무 공정 근거로 강하게 사용."
            )

        matched_pairs = self._diagnosis_pair_matches(diagnosis_context)
        for pair in matched_pairs[:5]:
            hints.append(
                f"진단 시 직무명/팀이 taxonomy 후보 '{pair['중직무']} > {pair['소직무']}'와 매칭됨."
            )

        matched_units = self._diagnosis_unit_matches(diagnosis_context)
        if matched_units:
            hints.append(
                "진단 Category/항목이 taxonomy 단위 직무와 매칭됨: "
                + ", ".join(matched_units[:12])
                + ". 단위 직무 판단 근거로 사용."
            )

        if diagnosis_context.row_count > 1:
            hints.append(
                f"진단 데이터가 {diagnosis_context.row_count}개 행으로 존재함. "
                "같은 구성원/연도 내 여러 스킬 항목이므로 반복되는 단서를 종합."
            )

        return hints

    def _apply_process_guardrail(
        self,
        canonical: dict[str, str],
        review: str,
    ) -> tuple[dict[str, str], str]:
        if normalize_cell(canonical.get("중직무", "")).casefold() == "공정".casefold():
            return canonical, ""

        unit_job = canonical.get("단위 직무", "")
        major_jobs = self.taxonomy.major_jobs_for_unit_job(unit_job)
        if "공정" not in major_jobs:
            return canonical, ""

        process_terms = self._matched_terms(review, PROCESS_EVIDENCE_TERMS)
        if len(process_terms) < 2:
            return canonical, ""

        same_device_candidates = self.taxonomy.rows_for_unit_job(
            unit_job,
            major_job="공정",
            device=canonical.get("Device", ""),
        )
        if len(same_device_candidates) == 1:
            terms = ", ".join(process_terms[:6])
            return (
                same_device_candidates[0],
                f"process_guardrail: '{unit_job}'이 중복 단위직무이고 공정 수행 단서({terms})가 강해 동일 Device의 공정 후보로 보정",
            )

        process_candidates = self.taxonomy.rows_for_unit_job(unit_job, major_job="공정")
        if len(process_candidates) == 1:
            terms = ", ".join(process_terms[:6])
            return (
                process_candidates[0],
                f"process_guardrail: '{unit_job}'이 중복 단위직무이고 공정 수행 단서({terms})가 강해 유일한 공정 후보로 보정",
            )

        return (
            canonical,
            f"process_guardrail_review: '{unit_job}'이 중복 단위직무이고 공정 수행 단서가 강하지만 공정 후보가 여러 개라 자동 보정하지 않음",
        )

    def _apply_diagnosis_guardrail(
        self,
        canonical: dict[str, str],
        diagnosis_context: DiagnosisContext | None,
    ) -> tuple[dict[str, str], str]:
        if not diagnosis_context:
            return canonical, ""

        best = self._best_diagnosis_taxonomy_match(diagnosis_context)
        if not best:
            return canonical, ""

        best_row, score, second_score, evidence = best
        if score < 8 or score - second_score < 2:
            return canonical, ""

        current_key = tuple(normalize_cell(canonical.get(column, "")) for column in TAXONOMY_COLUMNS)
        best_key = tuple(normalize_cell(best_row.get(column, "")) for column in TAXONOMY_COLUMNS)
        if current_key == best_key:
            return canonical, ""

        return (
            best_row,
            f"diagnosis_guardrail: 진단 데이터 근거({evidence})가 taxonomy 후보와 강하게 매칭되어 보정",
        )

    def _best_diagnosis_taxonomy_match(
        self,
        diagnosis_context: DiagnosisContext,
    ) -> tuple[dict[str, str], int, int, str] | None:
        scored: list[tuple[int, dict[str, str], list[str]]] = []
        diagnosis_fields = self._diagnosis_search_fields(diagnosis_context)

        for row in self.taxonomy.rows:
            score = 0
            evidence: list[str] = []

            major_job = row["중직무"]
            if self._value_in_fields(major_job, diagnosis_fields["team_job_item"]):
                score += 5
                evidence.append(f"중직무={major_job}")

            sub_job = row["소직무"]
            if self._value_in_fields(sub_job, diagnosis_fields["team_job_item"]):
                score += 5
                evidence.append(f"소직무={sub_job}")

            unit_job = row["단위 직무"]
            if self._value_in_fields(unit_job, diagnosis_fields["category_item_job"]):
                score += 4
                evidence.append(f"단위 직무={unit_job}")

            device = row["Device"]
            if self._value_in_fields(device, diagnosis_fields["all"]):
                score += 1
                evidence.append(f"Device={device}")

            for detail_column in ["세부 직무1", "세부 직무2"]:
                detail_job = row[detail_column]
                if self._value_in_fields(detail_job, diagnosis_fields["category_item_job"]):
                    score += 2
                    evidence.append(f"{detail_column}={detail_job}")

            if score:
                scored.append((score, row, evidence))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_row, best_evidence = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0
        return best_row, best_score, second_score, ", ".join(best_evidence[:6])

    def _diagnosis_pair_matches(self, diagnosis_context: DiagnosisContext) -> list[dict[str, str]]:
        fields = self._diagnosis_search_fields(diagnosis_context)["team_job_item"]
        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for pair in self.taxonomy.pairs():
            if not self._value_in_fields(pair["중직무"], fields):
                continue
            if not self._value_in_fields(pair["소직무"], fields):
                continue
            key = (pair["중직무"], pair["소직무"])
            if key in seen:
                continue
            seen.add(key)
            result.append(pair)
        return result

    def _diagnosis_unit_matches(self, diagnosis_context: DiagnosisContext) -> list[str]:
        fields = self._diagnosis_search_fields(diagnosis_context)["category_item_job"]
        result: list[str] = []
        seen: set[str] = set()
        for row in self.taxonomy.rows:
            unit_job = row["단위 직무"]
            key = normalize_cell(unit_job)
            if not key or normalize_key(key) in seen:
                continue
            if self._value_in_fields(unit_job, fields):
                seen.add(normalize_key(key))
                result.append(key)
        return result

    def _diagnosis_search_fields(self, diagnosis_context: DiagnosisContext) -> dict[str, list[str]]:
        teams = diagnosis_context.teams
        job_names = diagnosis_context.job_names
        categories = diagnosis_context.categories
        items = diagnosis_context.items
        return {
            "team_job_item": teams + job_names + items,
            "category_item_job": categories + items + job_names,
            "all": teams + job_names + categories + items,
        }

    def _diagnosis_text(self, diagnosis_context: DiagnosisContext) -> str:
        fields = self._diagnosis_search_fields(diagnosis_context)["all"]
        return " ".join(fields)

    def _value_in_fields(self, value: object, fields: list[str]) -> bool:
        value_key = normalize_cell(value)
        if len(value_key) < 2:
            return False
        target = normalize_key(value_key)
        for field in fields:
            field_key = normalize_key(field)
            if not field_key:
                continue
            if target == field_key or target in field_key or field_key in target:
                return True
        return False

    def _matched_terms(self, review: str, terms: list[str]) -> list[str]:
        review_key = review.casefold()
        return [term for term in terms if term.casefold() in review_key]

    def _chat_json_text(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if self.settings.use_json_response_format:
            payload["response_format"] = {"type": "json_object"}
        if self.settings.extra_body:
            payload["extra_body"] = self.settings.extra_body

        for attempt in Retrying(
            stop=stop_after_attempt(self.config.api_retry_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=12),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                completion = self.client.chat.completions.create(**payload)
                choice = completion.choices[0]
                content = choice.message.content
                if not content:
                    raise ValueError(self._empty_response_error(choice))
                return content
        raise RuntimeError("unreachable retry state")

    def _empty_response_error(self, choice: Any) -> str:
        message = choice.message
        finish_reason = getattr(choice, "finish_reason", None)
        message_dump = message.model_dump() if hasattr(message, "model_dump") else {}
        extra_keys = sorted(
            key
            for key, value in message_dump.items()
            if value not in (None, "", [], {})
        )
        return (
            "empty LLM response"
            f" (finish_reason={finish_reason}, message_non_empty_keys={extra_keys})"
        )

    def _cache_key(self, context_json: str) -> str:
        payload = {
            "context": context_json,
            "taxonomy_version": self.taxonomy.version_hash(),
            "model": self.settings.model,
            "include_team": self.config.include_team_in_prompt,
            "prompt_version": PROMPT_VERSION,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _review_required_result(self, reason: str) -> dict[str, Any]:
        return {
            **{column: "" for column in TAXONOMY_COLUMNS},
            "confidence": 0.0,
            "reason": "",
            "needs_review": True,
            "ambiguity_reason": "",
            "guardrail_reason": "",
            "error": reason,
        }
