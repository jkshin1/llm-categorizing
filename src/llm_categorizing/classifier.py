from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_categorizing.config import LLMSettings
from llm_categorizing.diagnosis import DiagnosisContext, empty_diagnosis_output_payload
from llm_categorizing.knowledge import (
    JobKnowledge,
    JobKnowledgeStore,
    KnowledgeSearchContext,
    taxonomy_target_from_knowledge,
)
from llm_categorizing.models import FinalClassificationResult, Stage1Result
from llm_categorizing.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    correction_prompt,
    employee_context,
    stage1_user_prompt,
    stage2_user_prompt,
)
from llm_categorizing.taxonomy import TAXONOMY_COLUMNS, Taxonomy, normalize_cell


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
_HARD_DIAGNOSIS_MATCH_MIN_SCORE = 1000


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = _strip_thinking_blocks(text).strip()
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


def _strip_thinking_blocks(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text)


@dataclass
class ClassificationConfig:
    include_team_in_prompt: bool = False
    max_review_chars: int = 12000
    max_candidates_per_prompt: int = 300
    max_knowledge_hints: int = 8
    knowledge_review_scope: str = "usable"
    validation_attempts: int = 2
    api_retry_attempts: int = 5
    confidence_review_threshold: float = 0.6
    previous_year_min_current_review_chars: int = 120


@dataclass(frozen=True)
class DiagnosisPriority:
    major_job: str = ""
    sub_job: str = ""
    reason: str = ""


@dataclass(frozen=True)
class KnowledgeHardPriority:
    rows: tuple[dict[str, str], ...] = ()
    reasons: tuple[str, ...] = ()
    stage1_override_rows: tuple[dict[str, str], ...] = ()
    stage1_override_reasons: tuple[str, ...] = ()


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
                    if (
                        isinstance(key, str)
                        and isinstance(value, dict)
                        and self._can_reuse(value)
                    ):
                        self._items[key] = value

    def get(self, key: str) -> dict[str, Any] | None:
        value = self._items.get(key)
        return dict(value) if value else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        if self._can_reuse(value):
            self._items[key] = dict(value)
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._cache_record(key, value), ensure_ascii=False))
            handle.write("\n")

    @staticmethod
    def _can_reuse(value: dict[str, Any]) -> bool:
        return not value.get("error")

    @staticmethod
    def _cache_record(key: str, value: dict[str, Any]) -> dict[str, Any]:
        return {
            "cache_key": key,
            "year": value.get("year", ""),
            "emp_num": value.get("emp_num", ""),
            "name": value.get("name", ""),
            "value": value,
        }


class OpenAICompatibleJobClassifier:
    def __init__(
        self,
        *,
        settings: LLMSettings,
        taxonomy: Taxonomy,
        config: ClassificationConfig,
        cache: JsonlCache | None = None,
        knowledge_store: JobKnowledgeStore | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed. Run: pip install -r requirements.txt") from exc

        self.settings = settings
        self.taxonomy = taxonomy
        self.config = config
        self.cache = cache
        self.knowledge_store = knowledge_store
        self._openai_client_class = OpenAI
        self._api_keys = settings.normalized_api_keys()
        if not self._api_keys:
            raise ValueError("LLM api_key must not be empty")
        self._api_key_lock = Lock()
        self._next_api_key_index = 0
        self._active_api_key: ContextVar[str | None] = ContextVar(
            "active_llm_api_key",
            default=None,
        )
        self._clients_by_api_key: dict[str, Any] = {}
        self.client = self._client_for_api_key(self._api_keys[0])

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
        previous_year_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        review = normalize_cell(row.get("self_review", ""))
        if not review and not diagnosis_context:
            previous_year_payload = self._previous_year_prompt_payload(previous_year_context)
            result = self._review_required_result("self_review is blank")
            result.update(self._previous_year_output_payload(previous_year_payload))
            return result

        truncated = len(review) > self.config.max_review_chars
        review_for_prompt = review[: self.config.max_review_chars] if truncated else review
        knowledge_items = self._retrieve_knowledge(review_for_prompt, diagnosis_context)
        classification_hints = self._build_classification_hints(
            review_for_prompt,
            diagnosis_context,
            knowledge_items,
        )
        diagnosis_priority = self._diagnosis_priority(diagnosis_context)
        previous_year_payload = self._previous_year_prompt_payload(
            previous_year_context,
            current_review=review_for_prompt,
        )
        context_json = employee_context(
            year=normalize_cell(row.get("year", "")),
            team=normalize_cell(row.get("team", "")),
            self_review=review_for_prompt,
            include_team=self.config.include_team_in_prompt,
            input_truncated=truncated,
            classification_hints=classification_hints,
            diagnosis_context=diagnosis_context.to_prompt_payload() if diagnosis_context else None,
            previous_year_classification=previous_year_payload,
        )

        cache_identity = self._cache_identity_payload(row)
        cache_key = self._cache_key(context_json, cache_identity=cache_identity)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                result = dict(cached)
                result.update(
                    diagnosis_context.to_output_payload()
                    if diagnosis_context
                    else empty_diagnosis_output_payload()
                )
                result.update(self._previous_year_output_payload(previous_year_payload))
                return result

        try:
            with self._api_key_scope_for_classification():
                result = self._classify_uncached(context_json, diagnosis_priority, knowledge_items)
        except Exception as exc:
            result = self._review_required_result(f"classification_error: {exc}")

        result.update(
            diagnosis_context.to_output_payload()
            if diagnosis_context
            else empty_diagnosis_output_payload()
        )
        result["used_knowledge_ids"] = [item.id for item in knowledge_items]
        result["used_knowledge_types"] = [item.knowledge_type for item in knowledge_items]
        result["used_knowledge_scores"] = [round(item.match_score, 3) for item in knowledge_items]
        result["used_knowledge_review_statuses"] = [item.review_status for item in knowledge_items]
        result["used_knowledge_enforcement_levels"] = [
            item.enforcement_level for item in knowledge_items
        ]
        result["used_knowledge_match_fields"] = [
            ",".join(item.match_fields) for item in knowledge_items
        ]
        result["knowledge_version"] = self._knowledge_version()
        result["knowledge_review_scope"] = self.config.knowledge_review_scope
        result.setdefault("diagnosis_priority_reason", "")
        result.setdefault("knowledge_priority_reason", "")
        result.update(self._previous_year_output_payload(previous_year_payload))
        result["input_truncated"] = truncated
        result["taxonomy_version"] = self.taxonomy.version_hash()
        result["model_name"] = self.settings.model
        result["classified_at"] = datetime.now(timezone.utc).isoformat()
        result.update(cache_identity)
        if self.knowledge_store:
            self.knowledge_store.record_usage(
                classification_id=cache_key,
                knowledge_items=knowledge_items,
                result=result,
            )

        if self.cache:
            self.cache.set(cache_key, result)
        return result

    def _classify_uncached(
        self,
        context_json: str,
        diagnosis_priority: DiagnosisPriority,
        knowledge_items: list[JobKnowledge],
    ) -> dict[str, Any]:
        applied_priority_reasons: list[str] = []
        knowledge_priority = self._near_hard_knowledge_priority(
            knowledge_items,
            context_json,
            diagnosis_priority=diagnosis_priority,
        )
        knowledge_priority_reasons = list(knowledge_priority.reasons)
        stage1_override_candidates, stage1_override_reason = self._knowledge_stage1_override_candidates(
            knowledge_priority
        )

        review_pair_candidates, review_pair_reason = self._review_pair_candidates(context_json)
        pair_candidates, pair_priority_reason = self._diagnosis_pair_candidates(diagnosis_priority)
        if pair_priority_reason:
            review_override_candidates, review_override_reason = self._self_review_pair_override_candidates(
                pair_candidates,
                review_pair_candidates,
                review_pair_reason,
            )
            if review_override_reason:
                pair_candidates = review_override_candidates
                applied_priority_reasons.append(f"{pair_priority_reason}; {review_override_reason}")
                if knowledge_priority.rows:
                    knowledge_priority_reasons.append(
                        "self_review 직접 직무 단서 우선 적용: 준하드룰 지식은 선택된 중직무/소직무 내부의 최종 후보 제한에만 사용"
                    )
            else:
                applied_stage1_override = False
                if self._should_apply_stage1_knowledge_override(
                    pair_candidates,
                    stage1_override_candidates,
                ):
                    pair_candidates = stage1_override_candidates
                    applied_stage1_override = True
                    applied_priority_reasons.append(
                        f"{pair_priority_reason}; 준하드룰 지식이 과거 diagnosis 직무명/team 보정으로 stage1 후보를 대체"
                    )
                    knowledge_priority_reasons.append(stage1_override_reason)
                else:
                    applied_priority_reasons.append(pair_priority_reason)
                if knowledge_priority.rows and not applied_stage1_override:
                    knowledge_priority_reasons.append(
                        "diagnosis 직무명 우선 적용: 준하드룰 지식은 선택된 중직무/소직무 내부의 최종 후보 제한에만 사용"
                    )
        else:
            if review_pair_reason:
                pair_candidates = review_pair_candidates
                applied_priority_reasons.append(review_pair_reason)
                if knowledge_priority.rows:
                    knowledge_priority_reasons.append(
                        "self_review 직접 직무 단서 우선 적용: 준하드룰 지식은 선택된 중직무/소직무 내부의 최종 후보 제한에만 사용"
                    )
            elif knowledge_priority.rows:
                pair_candidates, pair_priority_reason = self._knowledge_pair_candidates(knowledge_priority)
                if pair_priority_reason:
                    knowledge_priority_reasons.append(pair_priority_reason)

        if len(pair_candidates) == 1:
            pair = pair_candidates[0]
        else:
            stage1 = self._run_stage1(context_json, pair_candidates)
            pair = self.taxonomy.canonical_pair(stage1.major_job, stage1.sub_job)
            if pair is None:
                return self._review_required_result_with_priorities(
                    "stage1 result is not in taxonomy",
                    applied_priority_reasons,
                    knowledge_priority_reasons,
                )

        candidates = self._final_candidates_for_pair(pair, knowledge_priority)
        if len(candidates) > self.config.max_candidates_per_prompt:
            return self._review_required_result_with_priorities(
                f"too many final candidates under selected pair: {len(candidates)}",
                applied_priority_reasons,
                knowledge_priority_reasons,
            )

        if knowledge_priority.rows and len(candidates) == 1:
            canonical = dict(candidates[0])
            return {
                **canonical,
                "confidence": 0.95,
                "reason": "준하드룰 지식이 현재 입력과 매칭되어 단일 taxonomy 후보로 제한됨.",
                "needs_review": False,
                "ambiguity_reason": "",
                "guardrail_reason": "",
                "diagnosis_priority_reason": "; ".join(applied_priority_reasons),
                "knowledge_priority_reason": "; ".join(knowledge_priority_reasons),
                "error": "",
            }

        final = self._run_stage2(context_json, candidates)
        stage2_recovery_reason = ""
        recovery_pair, recovery_reason = self._stage2_reason_recovery_pair(
            final,
            selected_pair=pair,
            allowed_pairs=pair_candidates,
        )
        if recovery_pair:
            recovered_candidates = self._final_candidates_for_pair(recovery_pair, knowledge_priority)
            if len(recovered_candidates) <= self.config.max_candidates_per_prompt:
                try:
                    recovered_final = self._run_stage2(context_json, recovered_candidates)
                    recovered_canonical = self._canonical_from_candidates(
                        self._candidate_from_final_result(recovered_final),
                        recovered_candidates,
                    )
                except Exception:
                    recovered_canonical = None
                if recovered_canonical:
                    pair = recovery_pair
                    candidates = recovered_candidates
                    final = recovered_final
                    stage2_recovery_reason = recovery_reason

        candidate = self._candidate_from_final_result(final)
        canonical = self._canonical_from_candidates(candidate, candidates)
        if canonical is None:
            return self._review_required_result_with_priorities(
                "final result is not in taxonomy",
                applied_priority_reasons,
                knowledge_priority_reasons,
            )

        needs_review = (
            final.needs_review
            or final.confidence < self.config.confidence_review_threshold
        )
        return {
            **canonical,
            "confidence": final.confidence,
            "reason": final.reason,
            "needs_review": needs_review,
            "ambiguity_reason": "",
            "guardrail_reason": stage2_recovery_reason,
            "diagnosis_priority_reason": "; ".join(applied_priority_reasons),
            "knowledge_priority_reason": "; ".join(knowledge_priority_reasons),
            "error": "",
        }

    def _run_stage1(
        self,
        context_json: str,
        candidate_pairs: list[dict[str, str]] | None = None,
    ) -> Stage1Result:
        allowed_pairs = candidate_pairs or self.taxonomy.pairs()
        candidates_json = self.taxonomy.format_candidates_json(allowed_pairs)
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

            if self._canonical_pair_from_candidates(result.major_job, result.sub_job, allowed_pairs):
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

    def _stage2_reason_recovery_pair(
        self,
        final: FinalClassificationResult,
        *,
        selected_pair: dict[str, str],
        allowed_pairs: list[dict[str, str]],
    ) -> tuple[dict[str, str] | None, str]:
        if len(allowed_pairs) <= 1:
            return None, ""

        reason = normalize_cell(final.reason)
        if not reason:
            return None, ""

        selected_key = self._pair_key(selected_pair)
        matches: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for pair in allowed_pairs:
            key = self._pair_key(pair)
            if key == selected_key or key in seen:
                continue
            if _reason_mentions_pair_path(reason, pair):
                seen.add(key)
                matches.append(dict(pair))

        if len(matches) != 1:
            return None, ""

        recovered_pair = matches[0]
        return (
            recovered_pair,
            "stage2 reason이 선택된 후보 밖의 사용 가능 pair "
            f"'{recovered_pair['중직무']} > {recovered_pair['소직무']}'를 명시해 stage2 재시도",
        )

    def _candidate_from_final_result(
        self,
        final: FinalClassificationResult,
    ) -> dict[str, str]:
        return {
            "중직무": final.major_job,
            "소직무": final.sub_job,
            "Device": final.device,
            "단위 직무": final.unit_job,
            "세부 직무1": final.detail_job_1,
            "세부 직무2": final.detail_job_2,
        }

    def _pair_key(self, pair: dict[str, str]) -> tuple[str, str]:
        return (
            normalize_cell(pair.get("중직무", "")).casefold(),
            normalize_cell(pair.get("소직무", "")).casefold(),
        )

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

    def _canonical_pair_from_candidates(
        self,
        major_job: object,
        sub_job: object,
        candidate_pairs: list[dict[str, str]],
    ) -> dict[str, str] | None:
        wanted_major = normalize_cell(major_job).casefold()
        wanted_sub = normalize_cell(sub_job).casefold()
        for pair in candidate_pairs:
            if (
                normalize_cell(pair.get("중직무", "")).casefold() == wanted_major
                and normalize_cell(pair.get("소직무", "")).casefold() == wanted_sub
            ):
                return dict(pair)
        return None

    def _near_hard_knowledge_priority(
        self,
        knowledge_items: list[JobKnowledge],
        context_json: str = "",
        diagnosis_priority: DiagnosisPriority | None = None,
    ) -> KnowledgeHardPriority:
        context_year, diagnosis_job_names, diagnosis_teams = self._diagnosis_override_context(context_json)
        rows: list[dict[str, str]] = []
        reasons: list[str] = []
        stage1_override_rows: list[dict[str, str]] = []
        stage1_override_reasons: list[str] = []
        seen_rows: set[tuple[str, ...]] = set()
        seen_stage1_override_rows: set[tuple[str, ...]] = set()
        for item in knowledge_items:
            if item.enforcement_level != "near_hard":
                continue
            if item.review_status != "approved" or item.conflicts:
                continue
            target = {
                column: normalize_cell(value)
                for column, value in taxonomy_target_from_knowledge(item).items()
                if normalize_cell(value)
            }
            if not target:
                continue
            matched_rows = self._taxonomy_rows_matching(target)
            if not matched_rows:
                continue
            is_stage1_override = self._is_diagnosis_job_name_stage1_override(
                item,
                diagnosis_job_names=diagnosis_job_names,
                diagnosis_teams=diagnosis_teams,
                context_year=context_year,
                diagnosis_priority=diagnosis_priority,
            )
            for row in matched_rows:
                key = tuple(normalize_cell(row.get(column, "")).casefold() for column in TAXONOMY_COLUMNS)
                if key in seen_rows:
                    continue
                seen_rows.add(key)
                rows.append(row)
                if is_stage1_override:
                    if key in seen_stage1_override_rows:
                        continue
                    seen_stage1_override_rows.add(key)
                    stage1_override_rows.append(row)
            reasons.append(
                f"준하드룰 지식[{item.id}] '{item.title}' target 기준 후보 제한"
            )
            if is_stage1_override:
                stage1_override_reasons.append(
                    f"준하드룰 지식[{item.id}] '{item.title}' diagnosis 직무명/team 보정으로 stage1 후보 제한"
                )
        return KnowledgeHardPriority(
            rows=tuple(rows),
            reasons=tuple(reasons),
            stage1_override_rows=tuple(stage1_override_rows),
            stage1_override_reasons=tuple(stage1_override_reasons),
        )

    def _is_diagnosis_job_name_stage1_override(
        self,
        item: JobKnowledge,
        *,
        diagnosis_job_names: list[str],
        diagnosis_teams: list[str],
        context_year: str,
        diagnosis_priority: DiagnosisPriority | None = None,
    ) -> bool:
        job_name_match = (
            "diagnosis_job_name" in item.match_fields
            and self._knowledge_alias_matches_values(item, diagnosis_job_names)
        )
        team_match = (
            "diagnosis_team" in item.match_fields
            and self._knowledge_alias_matches_team(item, diagnosis_teams)
        )
        if (
            team_match
            and not job_name_match
            and self._team_override_conflicts_with_current_diagnosis(item, diagnosis_priority)
        ):
            return False
        return (
            bool(normalize_cell(item.target_major_job))
            and bool(normalize_cell(item.target_sub_job))
            and (job_name_match or team_match)
            and self._knowledge_year_allows(item, context_year)
        )

    def _team_override_conflicts_with_current_diagnosis(
        self,
        item: JobKnowledge,
        diagnosis_priority: DiagnosisPriority | None,
    ) -> bool:
        if diagnosis_priority is None:
            return False
        target_major = normalize_cell(item.target_major_job).casefold()
        target_sub = normalize_cell(item.target_sub_job).casefold()
        current_major = normalize_cell(diagnosis_priority.major_job).casefold()
        current_sub = normalize_cell(diagnosis_priority.sub_job).casefold()
        if not target_major or not target_sub or not current_major or not current_sub:
            return False
        return current_major == target_major and current_sub != target_sub

    def _diagnosis_override_context(self, context_json: str) -> tuple[str, list[str], list[str]]:
        try:
            context = json.loads(context_json) if context_json else {}
        except json.JSONDecodeError:
            return "", [], []
        if not isinstance(context, dict):
            return "", [], []
        diagnosis_context = context.get("diagnosis_context")
        if not isinstance(diagnosis_context, dict):
            diagnosis_context = {}
        raw_job_names = diagnosis_context.get("diagnosis_job_names", [])
        if not isinstance(raw_job_names, list):
            raw_job_names = []
        raw_teams = diagnosis_context.get("diagnosis_teams", [])
        if not isinstance(raw_teams, list):
            raw_teams = []
        job_names = [normalize_cell(value) for value in raw_job_names if normalize_cell(value)]
        teams = [normalize_cell(value) for value in raw_teams if normalize_cell(value)]
        return normalize_cell(context.get("year", "")), job_names, teams

    def _knowledge_alias_matches_values(self, item: JobKnowledge, values: list[str]) -> bool:
        if not values:
            return False
        for alias in item.aliases:
            for value in values:
                if _diagnosis_job_name_alias_matches(alias, value):
                    return True
        return False

    def _knowledge_alias_matches_team(self, item: JobKnowledge, teams: list[str]) -> bool:
        if not teams:
            return False
        for alias in item.aliases:
            for team in teams:
                if _diagnosis_team_alias_matches(alias, team):
                    return True
        return False

    def _knowledge_year_allows(self, item: JobKnowledge, context_year: str) -> bool:
        year = _parse_year(context_year)
        if year is None:
            return True

        text = " ".join(
            [
                item.raw_text,
                item.title,
                item.applies_when,
                item.hint,
            ]
        )
        ranges = _year_ranges(text)
        years = _explicit_years(text)
        if not ranges and not years:
            return True
        return any(start <= year <= end for start, end in ranges) or year in years

    def _taxonomy_rows_matching(self, specified: dict[str, str]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for row in self.taxonomy.rows:
            if all(
                normalize_cell(row.get(column, "")).casefold() == normalize_cell(value).casefold()
                for column, value in specified.items()
            ):
                rows.append(dict(row))
        return rows

    def _knowledge_pair_candidates(
        self,
        knowledge_priority: KnowledgeHardPriority,
    ) -> tuple[list[dict[str, str]], str]:
        pairs: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in knowledge_priority.rows:
            key = (
                normalize_cell(row.get("중직무", "")).casefold(),
                normalize_cell(row.get("소직무", "")).casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"중직무": row["중직무"], "소직무": row["소직무"]})
        reason = "준하드룰 지식 우선 적용: stage1 후보를 매칭 taxonomy row의 중직무/소직무로 제한"
        return pairs or self.taxonomy.pairs(), reason if pairs else ""

    def _knowledge_stage1_override_candidates(
        self,
        knowledge_priority: KnowledgeHardPriority,
    ) -> tuple[list[dict[str, str]], str]:
        pairs: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in knowledge_priority.stage1_override_rows:
            key = (
                normalize_cell(row.get("중직무", "")).casefold(),
                normalize_cell(row.get("소직무", "")).casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"중직무": row["중직무"], "소직무": row["소직무"]})
        reason = "; ".join(knowledge_priority.stage1_override_reasons)
        return pairs, reason

    def _should_apply_stage1_knowledge_override(
        self,
        current_pair_candidates: list[dict[str, str]],
        override_pair_candidates: list[dict[str, str]],
    ) -> bool:
        if len(override_pair_candidates) != 1:
            return False
        if len(current_pair_candidates) != 1:
            return True
        return self._pair_key(current_pair_candidates[0]) != self._pair_key(override_pair_candidates[0])

    def _self_review_pair_override_candidates(
        self,
        diagnosis_pair_candidates: list[dict[str, str]],
        review_pair_candidates: list[dict[str, str]],
        review_pair_reason: str,
    ) -> tuple[list[dict[str, str]], str]:
        if not review_pair_reason or len(review_pair_candidates) != 1:
            return diagnosis_pair_candidates, ""

        review_pair = dict(review_pair_candidates[0])
        review_key = self._pair_key(review_pair)
        diagnosis_keys = {self._pair_key(pair) for pair in diagnosis_pair_candidates}
        if len(diagnosis_keys) == 1 and review_key in diagnosis_keys:
            return diagnosis_pair_candidates, ""

        if review_key in diagnosis_keys:
            return (
                [review_pair],
                f"{review_pair_reason}; diagnosis 후보가 여러 개라 self_review 직접 직무 단서로 단일 pair 선택",
            )

        return (
            [review_pair],
            f"{review_pair_reason}; diagnosis 우선 후보와 명확히 충돌해 과거 diagnosis보다 self_review 기준을 우선 적용",
        )

    def _final_candidates_for_pair(
        self,
        pair: dict[str, str],
        knowledge_priority: KnowledgeHardPriority,
    ) -> list[dict[str, str]]:
        candidates = self.taxonomy.children_for_pair(pair["중직무"], pair["소직무"])
        if not knowledge_priority.rows:
            return self._with_optional_dic_unit_candidates(candidates)
        pair_key = (
            normalize_cell(pair.get("중직무", "")).casefold(),
            normalize_cell(pair.get("소직무", "")).casefold(),
        )
        filtered = [
            row
            for row in knowledge_priority.rows
            if (
                normalize_cell(row.get("중직무", "")).casefold(),
                normalize_cell(row.get("소직무", "")).casefold(),
            )
            == pair_key
        ]
        return self._with_optional_dic_unit_candidates(filtered or candidates)

    def _with_optional_dic_unit_candidates(
        self,
        candidates: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        optional_rows: list[dict[str, str]] = []
        existing_keys = {
            tuple(normalize_cell(row.get(column, "")).casefold() for column in TAXONOMY_COLUMNS)
            for row in candidates
        }
        seen_optional_keys: set[tuple[str, ...]] = set()
        for row in candidates:
            if normalize_cell(row.get("중직무", "")).casefold() != "dic":
                continue
            optional = {
                "중직무": row.get("중직무", ""),
                "소직무": row.get("소직무", ""),
                "Device": row.get("Device", ""),
                "단위 직무": "",
                "세부 직무1": "",
                "세부 직무2": "",
            }
            key = tuple(normalize_cell(optional.get(column, "")).casefold() for column in TAXONOMY_COLUMNS)
            if key in existing_keys or key in seen_optional_keys:
                continue
            seen_optional_keys.add(key)
            optional_rows.append(optional)
        if not optional_rows:
            return candidates
        return optional_rows + candidates

    def _diagnosis_priority(
        self,
        diagnosis_context: DiagnosisContext | None,
    ) -> DiagnosisPriority:
        if not diagnosis_context:
            return DiagnosisPriority()

        reasons: list[str] = []
        sub_job = self._best_column_match("소직무", diagnosis_context.job_names)
        hard_sub_job = sub_job if sub_job and _is_hard_diagnosis_match(sub_job[2]) else None
        unit_job = (
            None
            if hard_sub_job
            else self._best_column_match("단위 직무", diagnosis_context.job_names)
        )
        hard_unit_job = unit_job if unit_job and _is_hard_diagnosis_match(unit_job[2]) else None

        major_job = ""
        inferred_sub_job = hard_sub_job[0] if hard_sub_job else ""

        if hard_sub_job:
            reasons.append(f"진단 직무명 '{hard_sub_job[1]}' -> 소직무 '{hard_sub_job[0]}'")
            sub_pairs = self._pairs_for_column_value("소직무", hard_sub_job[0])
            if len(sub_pairs) == 1:
                major_job = sub_pairs[0]["중직무"]
            elif len(sub_pairs) > 1:
                major_pair = self._resolve_pair_by_diagnosis_major_signal(
                    sub_pairs,
                    diagnosis_context.job_names + diagnosis_context.teams,
                )
                if major_pair:
                    major_job = major_pair["중직무"]
                    reasons.append(
                        f"진단 직무명/team의 중직무 직접 단서 -> 중직무 '{major_job}'"
                    )
        elif hard_unit_job:
            unit_pairs = self._pairs_for_column_value("단위 직무", hard_unit_job[0])
            if len(unit_pairs) == 1:
                major_job = unit_pairs[0]["중직무"]
                inferred_sub_job = unit_pairs[0]["소직무"]
                reasons.append(
                    f"진단 직무명 '{hard_unit_job[1]}' -> 단위 직무 '{hard_unit_job[0]}'"
                )

        return DiagnosisPriority(
            major_job=major_job,
            sub_job=inferred_sub_job,
            reason="; ".join(reasons),
        )

    def _diagnosis_pair_candidates(
        self,
        diagnosis_priority: DiagnosisPriority,
    ) -> tuple[list[dict[str, str]], str]:
        pairs = self.taxonomy.pairs()
        major_key = normalize_cell(diagnosis_priority.major_job).casefold()
        sub_key = normalize_cell(diagnosis_priority.sub_job).casefold()
        if not major_key and not sub_key:
            return pairs, ""

        filtered = [
            pair
            for pair in pairs
            if (not major_key or normalize_cell(pair["중직무"]).casefold() == major_key)
            and (not sub_key or normalize_cell(pair["소직무"]).casefold() == sub_key)
        ]
        if filtered:
            return filtered, self._diagnosis_pair_reason(diagnosis_priority)

        if sub_key:
            sub_filtered = [
                pair
                for pair in pairs
                if normalize_cell(pair["소직무"]).casefold() == sub_key
            ]
            if sub_filtered:
                return (
                    sub_filtered,
                    f"diagnosis 우선 적용: 소직무 '{diagnosis_priority.sub_job}' 기준 후보 제한",
                )

        if major_key:
            major_filtered = [
                pair
                for pair in pairs
                if normalize_cell(pair["중직무"]).casefold() == major_key
            ]
            if major_filtered:
                return (
                    major_filtered,
                    f"diagnosis 우선 적용: 중직무 '{diagnosis_priority.major_job}' 기준 후보 제한",
                )

        return pairs, ""

    def _review_pair_candidates(self, context_json: str) -> tuple[list[dict[str, str]], str]:
        pairs = self.taxonomy.pairs()
        try:
            context = json.loads(context_json)
        except json.JSONDecodeError:
            return pairs, ""

        review = normalize_cell(context.get("self_review", ""))
        if not review:
            return pairs, ""

        scored: list[tuple[int, dict[str, str]]] = []
        for pair in pairs:
            major_score = _text_match_score(pair.get("중직무", ""), review)
            sub_score = _sub_job_signal_score(pair.get("소직무", ""), review)
            if major_score and sub_score:
                scored.append((major_score + sub_score, pair))

        if not scored:
            return pairs, ""

        scored.sort(key=lambda item: item[0], reverse=True)
        top_score = scored[0][0]
        winners = [pair for score, pair in scored if score == top_score]
        winner_keys = {self._pair_key(pair) for pair in winners}
        if len(winner_keys) != 1:
            return pairs, ""

        pair = dict(winners[0])
        return (
            [pair],
            "self_review 직접 직무 단서 우선 적용: "
            f"중직무 '{pair['중직무']}', 소직무 '{pair['소직무']}' 기준 stage1 후보 제한",
        )

    def _diagnosis_pair_reason(self, diagnosis_priority: DiagnosisPriority) -> str:
        labels: list[str] = []
        if diagnosis_priority.major_job:
            labels.append(f"중직무 '{diagnosis_priority.major_job}'")
        if diagnosis_priority.sub_job:
            labels.append(f"소직무 '{diagnosis_priority.sub_job}'")
        return f"diagnosis 우선 적용: {', '.join(labels)} 기준 stage1 후보 제한"

    def _resolve_pair_by_diagnosis_major_signal(
        self,
        pairs: list[dict[str, str]],
        source_values: list[str],
    ) -> dict[str, str] | None:
        scored: list[tuple[int, dict[str, str], str]] = []
        for pair in pairs:
            major_job = normalize_cell(pair.get("중직무", ""))
            for source_value in source_values:
                score = _text_match_score(major_job, source_value)
                if score:
                    scored.append((score, pair, source_value))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        top_score = scored[0][0]
        winners = [
            pair
            for score, pair, _source_value in scored
            if score == top_score
        ]
        winner_keys = {
            (
                normalize_cell(pair.get("중직무", "")).casefold(),
                normalize_cell(pair.get("소직무", "")).casefold(),
            )
            for pair in winners
        }
        if len(winner_keys) != 1:
            return None
        return dict(winners[0])

    def _best_column_match(
        self,
        column: str,
        source_values: list[str],
    ) -> tuple[str, str, int] | None:
        scored: list[tuple[int, str, str]] = []
        for taxonomy_value in self._unique_column_values(column):
            for source_value in source_values:
                score = _text_match_score(taxonomy_value, source_value)
                if score:
                    scored.append((score, taxonomy_value, source_value))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        top_score = scored[0][0]
        winners = [
            (taxonomy_value, source_value)
            for score, taxonomy_value, source_value in scored
            if score == top_score
        ]
        winner_keys = {normalize_cell(taxonomy_value).casefold() for taxonomy_value, _ in winners}
        if len(winner_keys) != 1:
            return None
        taxonomy_value, source_value = winners[0]
        return taxonomy_value, source_value, top_score

    def _unique_column_values(self, column: str) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for row in self.taxonomy.rows:
            value = normalize_cell(row.get(column, ""))
            key = value.casefold()
            if not value or key in seen:
                continue
            seen.add(key)
            values.append(value)
        return values

    def _pairs_for_column_value(self, column: str, value: str) -> list[dict[str, str]]:
        wanted = normalize_cell(value).casefold()
        pairs: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in self.taxonomy.rows:
            if normalize_cell(row.get(column, "")).casefold() != wanted:
                continue
            key = (
                normalize_cell(row["중직무"]).casefold(),
                normalize_cell(row["소직무"]).casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"중직무": row["중직무"], "소직무": row["소직무"]})
        return pairs

    def _build_classification_hints(
        self,
        review: str,
        diagnosis_context: DiagnosisContext | None = None,
        knowledge_items: list[JobKnowledge] | None = None,
    ) -> list[str]:
        hints: list[str] = []

        if knowledge_items:
            for item in knowledge_items[: self.config.max_knowledge_hints]:
                hints.append(item.prompt_hint())

        if diagnosis_context:
            hints.extend(
                self._diagnosis_team_major_hints(
                    diagnosis_context,
                    knowledge_items or [],
                )
            )

        return hints[: max(0, self.config.max_knowledge_hints)]

    def _diagnosis_team_major_hints(
        self,
        diagnosis_context: DiagnosisContext,
        knowledge_items: list[JobKnowledge],
    ) -> list[str]:
        hints: list[str] = []
        hints.extend(self._direct_team_major_hints(diagnosis_context))
        hints.extend(self._knowledge_team_major_hints(diagnosis_context, knowledge_items))
        return hints[: self.config.max_knowledge_hints]

    def _direct_team_major_hints(
        self,
        diagnosis_context: DiagnosisContext,
    ) -> list[str]:
        matches: list[tuple[int, str, str]] = []
        for major_job in self._unique_column_values("중직무"):
            for team in diagnosis_context.teams:
                score = _text_match_score(major_job, team)
                if score:
                    matches.append((score, major_job, team))

        matches.sort(key=lambda item: item[0], reverse=True)
        hints: list[str] = []
        seen: set[str] = set()
        for _, major_job, team in matches:
            key = normalize_cell(major_job).casefold()
            if key in seen:
                continue
            seen.add(key)
            hints.append(
                "diagnosis team 직접 단서: "
                f"team '{team}'에 taxonomy 중직무 '{major_job}' 표현이 포함됨. "
                "자동 후보 제한은 아니며 중직무 판단 참고로만 사용."
            )
            if len(hints) >= 3:
                break
        return hints

    def _knowledge_team_major_hints(
        self,
        diagnosis_context: DiagnosisContext,
        knowledge_items: list[JobKnowledge],
    ) -> list[str]:
        hints: list[str] = []
        seen: set[tuple[str, str]] = set()
        for item in knowledge_items[: self.config.max_knowledge_hints]:
            major_job = normalize_cell(item.target_major_job)
            if not major_job:
                continue
            match = self._knowledge_alias_team_match(item, diagnosis_context.teams)
            if not match:
                continue
            alias, team = match
            key = (item.id, major_job.casefold())
            if key in seen:
                continue
            seen.add(key)
            hints.append(
                "diagnosis team 사용자 지식 단서: "
                f"team '{team}'에 사용자 지식[{item.id}] alias '{alias}'가 매칭되어 "
                f"중직무 '{major_job}' 후보를 검토."
            )
        return hints

    def _knowledge_alias_team_match(
        self,
        item: JobKnowledge,
        teams: list[str],
    ) -> tuple[str, str] | None:
        scored: list[tuple[int, str, str]] = []
        for alias in item.aliases:
            for team in teams:
                score = _text_match_score(alias, team)
                if score:
                    scored.append((score, alias, team))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        _, alias, team = scored[0]
        return alias, team

    def _previous_year_prompt_payload(
        self,
        previous_year_context: dict[str, Any] | None,
        *,
        current_review: str = "",
    ) -> dict[str, object] | None:
        if not previous_year_context:
            return None
        if self._current_review_has_sufficient_evidence(current_review):
            return None

        source_classification = previous_year_context.get("classification")
        if not isinstance(source_classification, dict):
            source_classification = previous_year_context

        classification = {
            column: normalize_cell(source_classification.get(column, ""))
            for column in TAXONOMY_COLUMNS
        }
        if not any(classification.values()):
            return None

        payload: dict[str, object] = {
            "year": normalize_cell(previous_year_context.get("year", "")),
            "classification": classification,
            "job_path": _job_path_text(classification),
            "confidence": _safe_float(previous_year_context.get("confidence", 0.0)),
            "needs_review": bool(previous_year_context.get("needs_review", True)),
            "is_soft_continuity_hint": True,
        }
        reason = normalize_cell(previous_year_context.get("reason", ""))
        if reason:
            payload["reason"] = reason[:500]
        return payload

    def _current_review_has_sufficient_evidence(self, review: str) -> bool:
        return len(normalize_cell(review)) >= self.config.previous_year_min_current_review_chars

    def _previous_year_output_payload(
        self,
        previous_year_payload: dict[str, object] | None,
    ) -> dict[str, Any]:
        if not previous_year_payload:
            return {
                "previous_year": "",
                "previous_year_job_path": "",
                "previous_year_confidence": "",
                "previous_year_needs_review": "",
            }
        return {
            "previous_year": previous_year_payload.get("year", ""),
            "previous_year_job_path": previous_year_payload.get("job_path", ""),
            "previous_year_confidence": previous_year_payload.get("confidence", ""),
            "previous_year_needs_review": previous_year_payload.get("needs_review", ""),
        }

    def _retrieve_knowledge(
        self,
        review: str,
        diagnosis_context: DiagnosisContext | None,
    ) -> list[JobKnowledge]:
        if not self.knowledge_store:
            return []
        context = KnowledgeSearchContext(
            self_review=review,
            diagnosis_teams=tuple(diagnosis_context.teams if diagnosis_context else []),
            diagnosis_job_names=tuple(diagnosis_context.job_names if diagnosis_context else []),
            diagnosis_categories=tuple(diagnosis_context.categories if diagnosis_context else []),
        )
        return self.knowledge_store.retrieve_for_context(
            context,
            limit=self.config.max_knowledge_hints,
            review_scope=self.config.knowledge_review_scope,
        )

    def _diagnosis_text(self, diagnosis_context: DiagnosisContext) -> str:
        fields = (
            diagnosis_context.teams
            + diagnosis_context.job_names
            + diagnosis_context.categories
        )
        return " ".join(fields)

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
                completion = self._current_client().chat.completions.create(**payload)
                choice = completion.choices[0]
                content = choice.message.content
                if not content:
                    raise ValueError(self._empty_response_error(choice))
                return content
        raise RuntimeError("unreachable retry state")

    @contextmanager
    def _api_key_scope_for_classification(self):
        api_key = self._next_api_key_for_classification()
        token = self._active_api_key.set(api_key)
        try:
            yield
        finally:
            self._active_api_key.reset(token)

    def _next_api_key_for_classification(self) -> str:
        with self._api_key_lock:
            api_key = self._api_keys[self._next_api_key_index]
            self._next_api_key_index = (self._next_api_key_index + 1) % len(self._api_keys)
            return api_key

    def _current_client(self) -> Any:
        api_key = self._active_api_key.get() or self._api_keys[0]
        return self._client_for_api_key(api_key)

    def _client_for_api_key(self, api_key: str) -> Any:
        client = self._clients_by_api_key.get(api_key)
        if client is None:
            client = self._openai_client_class(
                base_url=self.settings.base_url,
                api_key=api_key,
                timeout=self.settings.timeout_seconds,
            )
            self._clients_by_api_key[api_key] = client
        return client

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

    def _cache_key(
        self,
        context_json: str,
        *,
        cache_identity: dict[str, str] | None = None,
    ) -> str:
        payload = {
            "context": context_json,
            "cache_identity": cache_identity or {},
            "taxonomy_version": self.taxonomy.version_hash(),
            "knowledge_version": self._knowledge_version(),
            "model": self.settings.model,
            "provider_profile": self.settings.provider_profile,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "extra_body": self.settings.extra_body or {},
            "include_team": self.config.include_team_in_prompt,
            "confidence_review_threshold": self.config.confidence_review_threshold,
            "previous_year_min_current_review_chars": self.config.previous_year_min_current_review_chars,
            "diagnosis_hard_match_policy": "exact_or_compact_exact_with_major_tiebreak_v2",
            "near_hard_knowledge_policy": "diagnosis_input_stage1_override_team_v7",
            "previous_year_prompt_policy": "fallback_only_when_current_review_short_v1",
            "self_review_pair_priority_policy": "taxonomy_major_sub_signal_match_v3",
            "diagnosis_self_review_conflict_policy": "self_review_direct_pair_overrides_diagnosis_v1",
            "stage2_pair_recovery_policy": "reason_major_sub_match_v2",
            "dic_optional_unit_candidate_policy": "blank_unit_per_pair_device_v1",
            "knowledge_review_scope": self.config.knowledge_review_scope,
            "prompt_version": PROMPT_VERSION,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_identity_payload(self, row: dict[str, Any]) -> dict[str, str]:
        return {
            "year": normalize_cell(row.get("year", "")),
            "emp_num": normalize_cell(row.get("emp_num", "")),
            "name": normalize_cell(row.get("name", "")),
        }

    def _knowledge_version(self) -> str:
        return self.knowledge_store.version_hash() if self.knowledge_store else ""

    def _review_required_result(self, reason: str) -> dict[str, Any]:
        return {
            **{column: "" for column in TAXONOMY_COLUMNS},
            "confidence": 0.0,
            "reason": "",
            "needs_review": True,
            "ambiguity_reason": "",
            "guardrail_reason": "",
            "diagnosis_priority_reason": "",
            "knowledge_priority_reason": "",
            "previous_year": "",
            "previous_year_job_path": "",
            "previous_year_confidence": "",
            "previous_year_needs_review": "",
            "error": reason,
        }

    def _review_required_result_with_priorities(
        self,
        reason: str,
        applied_priority_reasons: list[str],
        knowledge_priority_reasons: list[str],
    ) -> dict[str, Any]:
        result = self._review_required_result(reason)
        result["diagnosis_priority_reason"] = "; ".join(applied_priority_reasons)
        result["knowledge_priority_reason"] = "; ".join(knowledge_priority_reasons)
        return result


def _compact_text(value: object) -> str:
    return re.sub(r"[\W_]+", "", normalize_cell(value).casefold())


def _reason_mentions_pair_path(reason: str, pair: dict[str, str]) -> bool:
    major_job = normalize_cell(pair.get("중직무", ""))
    sub_job = normalize_cell(pair.get("소직무", ""))
    if not major_job or not sub_job:
        return False

    compact_reason = re.sub(r"\s+", "", normalize_cell(reason).casefold())
    compact_path = re.sub(r"\s+", "", f"{major_job}>{sub_job}".casefold())
    if compact_path in compact_reason:
        return True

    return _text_match_score(major_job, reason) > 0 and _text_match_score(sub_job, reason) > 0


def _sub_job_signal_score(sub_job: object, review: str) -> int:
    direct_score = _text_match_score(sub_job, review)
    if direct_score:
        return direct_score

    scores = [
        _text_match_score(alias, review)
        for alias in _sub_job_signal_aliases(sub_job)
    ]
    return max(scores, default=0)


def _sub_job_signal_aliases(sub_job: object) -> list[str]:
    value = normalize_cell(sub_job)
    aliases: list[str] = []
    seen: set[str] = set()
    generic_tokens = {"공정", "process", "job", "업무", "직무"}

    without_generic = value
    for token in generic_tokens:
        without_generic = re.sub(re.escape(token), " ", without_generic, flags=re.IGNORECASE)

    candidates = [without_generic]
    candidates.extend(re.findall(r"[A-Za-z0-9]+|[가-힣]+", value))
    for candidate in candidates:
        alias = normalize_cell(candidate)
        key = _compact_text(alias)
        if not key or key in generic_tokens or key in seen:
            continue
        if len(key) < 3 and key.isascii():
            continue
        if len(key) < 2:
            continue
        seen.add(key)
        aliases.append(alias)
    return aliases


def _text_match_score(target: object, source: object) -> int:
    target_text = normalize_cell(target).casefold()
    source_text = normalize_cell(source).casefold()
    target_compact = _compact_text(target)
    source_compact = _compact_text(source)
    if not target_compact or not source_compact:
        return 0
    if target_text == source_text or target_compact == source_compact:
        return 1000 + len(target_compact)
    if len(target_compact) >= 2 and target_compact in source_compact:
        return 700 + len(target_compact)
    if len(source_compact) >= 2 and source_compact in target_compact:
        return 600 + len(source_compact)
    return 0


def _diagnosis_job_name_alias_matches(alias: object, diagnosis_job_name: object) -> bool:
    if _is_hard_diagnosis_match(_text_match_score(alias, diagnosis_job_name)):
        return True

    alias_compact = _compact_text(alias)
    job_compact = _compact_text(diagnosis_job_name)
    if not alias_compact or not job_compact:
        return False

    generic_suffixes = ("공정", "직무", "업무", "기술")
    for suffix in generic_suffixes:
        suffix_compact = _compact_text(suffix)
        if not suffix_compact or not job_compact.endswith(suffix_compact):
            continue
        base = job_compact[: -len(suffix_compact)]
        if base and alias_compact == base:
            return True
    return False


def _diagnosis_team_alias_matches(alias: object, diagnosis_team: object) -> bool:
    if _is_hard_diagnosis_match(_text_match_score(alias, diagnosis_team)):
        return True

    alias_tokens = _match_tokens(alias)
    team_tokens = _match_tokens(diagnosis_team)
    if not alias_tokens or not team_tokens:
        return False
    if len(alias_tokens) == 1:
        token = alias_tokens[0]
        return len(token) >= 3 and token in team_tokens
    return all(token in team_tokens for token in alias_tokens)


def _match_tokens(value: object) -> list[str]:
    return [
        token
        for token in re.findall(r"[0-9A-Za-z가-힣]+", normalize_cell(value).casefold())
        if token
    ]


def _is_hard_diagnosis_match(score: int) -> bool:
    return score >= _HARD_DIAGNOSIS_MATCH_MIN_SCORE


def _job_path_text(classification: dict[str, object]) -> str:
    return " > ".join(
        normalize_cell(classification.get(column, ""))
        for column in TAXONOMY_COLUMNS
        if normalize_cell(classification.get(column, ""))
    )


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_year(value: object) -> int | None:
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", normalize_cell(value))
    if not match:
        return None
    return int(match.group(1))


def _year_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(
        r"(?<!\d)(20\d{2})(?!\d)\s*년?\s*(?:~|-|부터|에서|to|through)\s*(20\d{2})(?!\d)",
        normalize_cell(text),
        flags=re.IGNORECASE,
    ):
        start = int(match.group(1))
        end = int(match.group(2))
        if start > end:
            start, end = end, start
        ranges.append((start, end))
    return ranges


def _explicit_years(text: str) -> set[int]:
    return {int(match) for match in re.findall(r"(?<!\d)(20\d{2})(?!\d)", normalize_cell(text))}
