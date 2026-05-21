from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from llm_categorizing.config import LLMSettings
from llm_categorizing.taxonomy import TAXONOMY_COLUMNS, Taxonomy, normalize_cell, normalize_key


DEFAULT_KNOWLEDGE_DB_PATH = "data/output/job_knowledge.sqlite3"
SUPPORTED_KNOWLEDGE_TYPES = {
    "glossary",
    "soft_hint",
    "negative_hint",
    "correction",
    "verified_rule",
}
SUPPORTED_REVIEW_STATUSES = {"draft", "approved", "rejected"}
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)

KNOWLEDGE_NORMALIZATION_SYSTEM_PROMPT = """너는 사내 직무 분류 지식을 정리하는 데이터 관리자다.
사용자가 입력한 자연어 지식을 직무 분류기가 참고할 수 있는 구조화 JSON으로 바꾼다.
입력에 없는 내용을 꾸며내지 말고, 확실하지 않은 target 계층은 빈 문자열로 둔다.
저장된 지식은 LLM 분류 판단의 soft hint로 쓰이며, 자동 보정 rule이 아니다.
응답은 설명 문장 없이 JSON 객체 하나만 출력한다."""


def knowledge_normalization_user_prompt(
    raw_text: str,
    taxonomy_reference_json: str | None = None,
) -> str:
    taxonomy_section = ""
    if taxonomy_reference_json:
        taxonomy_section = f"""
[사용 가능한 taxonomy 참고값]
{taxonomy_reference_json}

target_* 필드는 위 taxonomy 참고값에 존재하는 값만 사용하라.
명확히 매칭되지 않는 target_* 필드는 빈 문자열로 둔다.
"""
    return f"""아래 사용자 입력을 직무 분류 참고 지식으로 정리하라.

[사용자 입력]
{raw_text}
{taxonomy_section}

[knowledge_type 선택 기준]
- glossary: 용어 설명
- soft_hint: 일반 판단 참고
- negative_hint: 특정 단어만으로 오분류하지 말라는 주의
- correction: 과거 오분류 수정 사례
- verified_rule: 사람이 검증한 강한 지식일 때만 선택. 확실하지 않으면 soft_hint.

[출력 JSON 형식]
{{
  "knowledge_type": "soft_hint",
  "title": "한 줄 제목",
  "aliases": ["self_review에서 매칭할 핵심 용어 또는 표기 변형"],
  "applies_when": "이 지식이 적용되는 조건",
  "hint": "분류 모델에게 줄 판단 가이드. 후보를 강제하지 말고 검토 방향을 설명",
  "target_major_job": "명확할 때만 중직무",
  "target_sub_job": "명확할 때만 소직무",
  "target_device": "명확할 때만 Device",
  "target_unit_job": "명확할 때만 단위 직무",
  "target_detail_job_1": "명확할 때만 세부 직무1",
  "target_detail_job_2": "명확할 때만 세부 직무2",
  "priority": 50,
  "confidence": 0.7,
  "validation_errors": []
}}"""


def compact_knowledge_key(value: object) -> str:
    return re.sub(r"[\W_]+", "", normalize_cell(value).casefold(), flags=re.UNICODE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = _strip_thinking_blocks(text).strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response does not contain a JSON object")

    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object")
    return parsed


def _strip_thinking_blocks(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text)


class KnowledgeDraft(BaseModel):
    knowledge_type: str = "soft_hint"
    title: str = ""
    aliases: list[str] = Field(default_factory=list)
    applies_when: str = ""
    hint: str = ""
    target_major_job: str = ""
    target_sub_job: str = ""
    target_device: str = ""
    target_unit_job: str = ""
    target_detail_job_1: str = ""
    target_detail_job_2: str = ""
    priority: int = 50
    confidence: float = 0.5
    validation_errors: list[str] = Field(default_factory=list)

    @field_validator(
        "title",
        "applies_when",
        "hint",
        "target_major_job",
        "target_sub_job",
        "target_device",
        "target_unit_job",
        "target_detail_job_1",
        "target_detail_job_2",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        return normalize_cell(value)

    @field_validator("aliases", mode="before")
    @classmethod
    def _normalize_aliases(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, list):
            raw_items = value
        else:
            raw_items = []

        aliases: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            alias = normalize_cell(item)
            key = normalize_key(alias)
            if not alias or key in seen:
                continue
            seen.add(key)
            aliases.append(alias)
        return aliases[:30]

    @field_validator("knowledge_type", mode="before")
    @classmethod
    def _normalize_knowledge_type(cls, value: object) -> str:
        knowledge_type = normalize_cell(value).casefold()
        if knowledge_type not in SUPPORTED_KNOWLEDGE_TYPES:
            return "soft_hint"
        return knowledge_type

    @field_validator("validation_errors", mode="before")
    @classmethod
    def _normalize_validation_errors(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, list):
            raw_items = value
        else:
            raw_items = []
        result: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            text = normalize_cell(item)
            key = normalize_key(text)
            if not text or key in seen:
                continue
            seen.add(key)
            result.append(text)
        return result[:20]

    @field_validator("priority")
    @classmethod
    def _clamp_priority(cls, value: int) -> int:
        return max(1, min(100, int(value)))

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def with_fallbacks(self, raw_text: str) -> "KnowledgeDraft":
        title = self.title or normalize_cell(raw_text)[:80]
        hint = self.hint or normalize_cell(raw_text)
        aliases = list(self.aliases)
        for field_value in [
            self.target_major_job,
            self.target_sub_job,
            self.target_device,
            self.target_unit_job,
            self.target_detail_job_1,
            self.target_detail_job_2,
        ]:
            if field_value and normalize_key(field_value) not in {normalize_key(item) for item in aliases}:
                aliases.append(field_value)
        return self.model_copy(update={"title": title, "hint": hint, "aliases": aliases[:30]})


@dataclass(frozen=True)
class JobKnowledge:
    id: str
    raw_text: str
    knowledge_type: str
    title: str
    aliases: tuple[str, ...]
    applies_when: str
    hint: str
    target_major_job: str
    target_sub_job: str
    target_device: str
    target_unit_job: str
    target_detail_job_1: str
    target_detail_job_2: str
    priority: int
    confidence: float
    active: bool
    review_status: str
    validation_errors: tuple[str, ...]
    source: str
    created_at: str
    updated_at: str
    match_score: float = 0.0

    @classmethod
    def from_row(cls, row: sqlite3.Row, *, match_score: float = 0.0) -> "JobKnowledge":
        aliases = json.loads(row["aliases_json"] or "[]")
        if not isinstance(aliases, list):
            aliases = []
        validation_errors = json.loads(row["validation_errors_json"] or "[]")
        if not isinstance(validation_errors, list):
            validation_errors = []
        return cls(
            id=row["id"],
            raw_text=row["raw_text"],
            knowledge_type=row["knowledge_type"],
            title=row["title"],
            aliases=tuple(normalize_cell(item) for item in aliases if normalize_cell(item)),
            applies_when=row["applies_when"],
            hint=row["hint"],
            target_major_job=row["target_major_job"],
            target_sub_job=row["target_sub_job"],
            target_device=row["target_device"],
            target_unit_job=row["target_unit_job"],
            target_detail_job_1=row["target_detail_job_1"],
            target_detail_job_2=row["target_detail_job_2"],
            priority=int(row["priority"]),
            confidence=float(row["confidence"]),
            active=bool(row["active"]),
            review_status=row["review_status"],
            validation_errors=tuple(
                normalize_cell(item) for item in validation_errors if normalize_cell(item)
            ),
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            match_score=match_score,
        )

    def prompt_hint(self) -> str:
        label = {
            "glossary": "용어 설명",
            "soft_hint": "판단 참고",
            "negative_hint": "오분류 방지",
            "correction": "수정 사례",
            "verified_rule": "검증 지식",
        }.get(self.knowledge_type, "판단 참고")
        strength = (
            "사람이 검증한 지식이므로 self_review와 충돌하지 않으면 강하게 참고."
            if self.knowledge_type == "verified_rule" or self.review_status == "approved"
            else "자동 보정 rule이 아니라 판단 참고로만 사용."
        )
        parts = [f"사용자 지식[{self.id}] {label} - {self.title}: {self.hint} {strength}"]
        target = self.target_path_text()
        if target:
            parts.append(f"관련 후보 계층: {target}.")
        if self.applies_when:
            parts.append(f"적용 조건: {self.applies_when}.")
        if self.aliases:
            parts.append("매칭 용어: " + ", ".join(self.aliases[:10]) + ".")
        return " ".join(parts)

    def target_path_text(self) -> str:
        values = [
            self.target_major_job,
            self.target_sub_job,
            self.target_device,
            self.target_unit_job,
            self.target_detail_job_1,
            self.target_detail_job_2,
        ]
        return " > ".join(value for value in values if normalize_cell(value))

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "raw_text": self.raw_text,
            "knowledge_type": self.knowledge_type,
            "title": self.title,
            "aliases": list(self.aliases),
            "applies_when": self.applies_when,
            "hint": self.hint,
            "target_major_job": self.target_major_job,
            "target_sub_job": self.target_sub_job,
            "target_device": self.target_device,
            "target_unit_job": self.target_unit_job,
            "target_detail_job_1": self.target_detail_job_1,
            "target_detail_job_2": self.target_detail_job_2,
            "priority": self.priority,
            "confidence": self.confidence,
            "active": self.active,
            "review_status": self.review_status,
            "validation_errors": list(self.validation_errors),
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "match_score": self.match_score,
        }


class JobKnowledgeStore:
    def __init__(self, path: str | Path = DEFAULT_KNOWLEDGE_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id TEXT PRIMARY KEY,
                    raw_text TEXT NOT NULL,
                    knowledge_type TEXT NOT NULL DEFAULT 'soft_hint',
                    title TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    applies_when TEXT NOT NULL,
                    hint TEXT NOT NULL,
                    target_major_job TEXT NOT NULL,
                    target_sub_job TEXT NOT NULL,
                    target_device TEXT NOT NULL,
                    target_unit_job TEXT NOT NULL,
                    target_detail_job_1 TEXT NOT NULL,
                    target_detail_job_2 TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    active INTEGER NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'draft',
                    validation_errors_json TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_entries_active_priority
                ON knowledge_entries(active, priority)
                """
            )
            self._ensure_column(
                connection,
                "knowledge_entries",
                "knowledge_type",
                "TEXT NOT NULL DEFAULT 'soft_hint'",
            )
            self._ensure_column(
                connection,
                "knowledge_entries",
                "review_status",
                "TEXT NOT NULL DEFAULT 'draft'",
            )
            self._ensure_column(
                connection,
                "knowledge_entries",
                "validation_errors_json",
                "TEXT NOT NULL DEFAULT '[]'",
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    classification_id TEXT NOT NULL,
                    knowledge_id TEXT NOT NULL,
                    knowledge_type TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    match_score REAL NOT NULL,
                    final_major_job TEXT NOT NULL,
                    final_sub_job TEXT NOT NULL,
                    final_unit_job TEXT NOT NULL,
                    needs_review INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_usage_knowledge_id
                ON knowledge_usage(knowledge_id, created_at)
                """
            )

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_sql: str,
    ) -> None:
        existing = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name in existing:
            return
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")

    def add(self, raw_text: str, draft: KnowledgeDraft, *, source: str = "user") -> JobKnowledge:
        clean_raw_text = normalize_cell(raw_text)
        if not clean_raw_text:
            raise ValueError("knowledge text is blank")

        clean_draft = draft.with_fallbacks(clean_raw_text)
        now = _utc_now()
        entry_id = uuid.uuid4().hex[:16]
        values = {
            "id": entry_id,
            "raw_text": clean_raw_text,
            "knowledge_type": clean_draft.knowledge_type,
            "title": clean_draft.title,
            "aliases_json": json.dumps(clean_draft.aliases, ensure_ascii=False),
            "applies_when": clean_draft.applies_when,
            "hint": clean_draft.hint,
            "target_major_job": clean_draft.target_major_job,
            "target_sub_job": clean_draft.target_sub_job,
            "target_device": clean_draft.target_device,
            "target_unit_job": clean_draft.target_unit_job,
            "target_detail_job_1": clean_draft.target_detail_job_1,
            "target_detail_job_2": clean_draft.target_detail_job_2,
            "priority": clean_draft.priority,
            "confidence": clean_draft.confidence,
            "active": 1,
            "review_status": "approved" if clean_draft.knowledge_type == "verified_rule" else "draft",
            "validation_errors_json": json.dumps(clean_draft.validation_errors, ensure_ascii=False),
            "source": source,
            "created_at": now,
            "updated_at": now,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO knowledge_entries (
                    id, raw_text, knowledge_type, title, aliases_json, applies_when, hint,
                    target_major_job, target_sub_job, target_device, target_unit_job,
                    target_detail_job_1, target_detail_job_2, priority, confidence,
                    active, review_status, validation_errors_json, source, created_at, updated_at
                ) VALUES (
                    :id, :raw_text, :knowledge_type, :title, :aliases_json, :applies_when, :hint,
                    :target_major_job, :target_sub_job, :target_device, :target_unit_job,
                    :target_detail_job_1, :target_detail_job_2, :priority, :confidence,
                    :active, :review_status, :validation_errors_json, :source, :created_at, :updated_at
                )
                """,
                values,
            )
            row = connection.execute(
                "SELECT * FROM knowledge_entries WHERE id = ?",
                (entry_id,),
            ).fetchone()
        return JobKnowledge.from_row(row)

    def list_recent(self, *, limit: int = 50, include_inactive: bool = True) -> list[JobKnowledge]:
        where = "" if include_inactive else "WHERE active = 1"
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM knowledge_entries
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [JobKnowledge.from_row(row) for row in rows]

    def set_active(self, entry_id: str, active: bool) -> JobKnowledge | None:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE knowledge_entries
                SET active = ?, updated_at = ?
                WHERE id = ?
                """,
                (1 if active else 0, now, entry_id),
            )
            row = connection.execute(
                "SELECT * FROM knowledge_entries WHERE id = ?",
                (entry_id,),
            ).fetchone()
        return JobKnowledge.from_row(row) if row else None

    def update_metadata(
        self,
        entry_id: str,
        *,
        knowledge_type: str | None = None,
        review_status: str | None = None,
    ) -> JobKnowledge | None:
        clean_type = normalize_cell(knowledge_type).casefold() if knowledge_type is not None else None
        clean_status = normalize_cell(review_status).casefold() if review_status is not None else None
        if clean_type is not None and clean_type not in SUPPORTED_KNOWLEDGE_TYPES:
            raise ValueError(f"knowledge_type must be one of: {', '.join(sorted(SUPPORTED_KNOWLEDGE_TYPES))}")
        if clean_status is not None and clean_status not in SUPPORTED_REVIEW_STATUSES:
            raise ValueError(f"review_status must be one of: {', '.join(sorted(SUPPORTED_REVIEW_STATUSES))}")

        assignments: list[str] = []
        values: list[object] = []
        if clean_type is not None:
            assignments.append("knowledge_type = ?")
            values.append(clean_type)
        if clean_status is not None:
            assignments.append("review_status = ?")
            values.append(clean_status)
            if clean_status == "rejected":
                assignments.append("active = ?")
                values.append(0)
        if not assignments:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM knowledge_entries WHERE id = ?",
                    (entry_id,),
                ).fetchone()
            return JobKnowledge.from_row(row) if row else None

        assignments.append("updated_at = ?")
        values.append(_utc_now())
        values.append(entry_id)
        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE knowledge_entries
                SET {', '.join(assignments)}
                WHERE id = ?
                """,
                values,
            )
            row = connection.execute(
                "SELECT * FROM knowledge_entries WHERE id = ?",
                (entry_id,),
            ).fetchone()
        return JobKnowledge.from_row(row) if row else None

    def delete(self, entry_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM knowledge_entries WHERE id = ?", (entry_id,))
        return cursor.rowcount > 0

    def record_usage(
        self,
        *,
        classification_id: str,
        knowledge_items: list[JobKnowledge],
        result: dict[str, Any],
    ) -> None:
        if not knowledge_items:
            return
        now = _utc_now()
        rows = [
            (
                classification_id,
                item.id,
                item.knowledge_type,
                item.review_status,
                item.match_score,
                normalize_cell(result.get("중직무", "")),
                normalize_cell(result.get("소직무", "")),
                normalize_cell(result.get("단위 직무", "")),
                1 if bool(result.get("needs_review", True)) else 0,
                now,
            )
            for item in knowledge_items
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO knowledge_usage (
                    classification_id, knowledge_id, knowledge_type, review_status,
                    match_score, final_major_job, final_sub_job, final_unit_job,
                    needs_review, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def retrieve(self, text: str, *, limit: int = 8) -> list[JobKnowledge]:
        clean_text = normalize_cell(text)
        if not clean_text:
            return []

        text_key = normalize_key(clean_text)
        text_compact = compact_knowledge_key(clean_text)
        text_tokens = self._tokens(clean_text)
        scored: list[tuple[float, JobKnowledge]] = []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM knowledge_entries
                WHERE active = 1 AND review_status != 'rejected'
                ORDER BY priority DESC, created_at DESC
                """
            ).fetchall()

        for row in rows:
            knowledge = JobKnowledge.from_row(row)
            score = self._match_score(knowledge, text_key, text_compact, text_tokens)
            if score <= 0:
                continue
            scored.append((score, JobKnowledge.from_row(row, match_score=score)))

        scored.sort(key=lambda item: (item[0], item[1].priority, item[1].confidence), reverse=True)
        return [knowledge for _, knowledge in scored[:limit]]

    def version_hash(self) -> str:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, knowledge_type, review_status, title, aliases_json, applies_when, hint,
                       target_major_job, target_sub_job, target_device, target_unit_job,
                       target_detail_job_1, target_detail_job_2, priority, confidence,
                       active, validation_errors_json, updated_at
                FROM knowledge_entries
                WHERE active = 1 AND review_status != 'rejected'
                ORDER BY id
                """
            ).fetchall()
        payload = [dict(row) for row in rows]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def _match_score(
        self,
        knowledge: JobKnowledge,
        text_key: str,
        text_compact: str,
        text_tokens: set[str],
    ) -> float:
        score = 0.0
        base = max(1.0, knowledge.priority / 20.0)
        if knowledge.review_status == "approved":
            base += 2.0
        if knowledge.knowledge_type == "verified_rule":
            base += 3.0
        elif knowledge.knowledge_type == "correction":
            base += 1.5

        for alias in knowledge.aliases:
            alias_key = normalize_key(alias)
            alias_compact = compact_knowledge_key(alias)
            if len(alias_compact) < 3:
                if alias_key and alias_key in text_tokens:
                    score += 8.0 + base
                continue
            if alias_key and alias_key in text_key:
                score += 12.0 + base
            elif alias_compact and alias_compact in text_compact:
                score += 10.0 + base

        for target_value in [
            knowledge.target_major_job,
            knowledge.target_sub_job,
            knowledge.target_device,
            knowledge.target_unit_job,
            knowledge.target_detail_job_1,
            knowledge.target_detail_job_2,
        ]:
            target_compact = compact_knowledge_key(target_value)
            if len(target_compact) >= 3 and target_compact in text_compact:
                score += 3.0

        knowledge_tokens = self._tokens(
            " ".join(
                [
                    knowledge.title,
                    knowledge.applies_when,
                    knowledge.hint,
                    knowledge.raw_text,
                ]
            )
        )
        token_overlap = len(text_tokens.intersection(knowledge_tokens))
        if token_overlap >= 2:
            score += min(8.0, token_overlap * 1.5)

        if score > 0:
            score += knowledge.confidence * 2.0
        return score

    def _tokens(self, text: str) -> set[str]:
        tokens = set()
        for token in re.findall(r"[0-9A-Za-z가-힣]+", normalize_cell(text).casefold()):
            if len(token) < 2:
                continue
            tokens.add(token)
        return tokens


class KnowledgeNormalizer:
    def __init__(self, settings: LLMSettings, taxonomy: Taxonomy | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed. Run: pip install -r requirements.txt") from exc

        self.settings = settings
        self.taxonomy = taxonomy
        self.taxonomy_reference_json = taxonomy_reference_json(taxonomy) if taxonomy else None
        self.client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
        )

    def normalize(self, raw_text: str) -> KnowledgeDraft:
        clean_text = normalize_cell(raw_text)
        if not clean_text:
            raise ValueError("knowledge text is blank")

        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": KNOWLEDGE_NORMALIZATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": knowledge_normalization_user_prompt(
                        clean_text,
                        self.taxonomy_reference_json,
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": min(self.settings.max_tokens, 1200),
        }
        if self.settings.use_json_response_format:
            payload["response_format"] = {"type": "json_object"}
        extra_body = _knowledge_normalizer_extra_body(self.settings)
        if extra_body:
            payload["extra_body"] = extra_body

        completion = self.client.chat.completions.create(**payload)
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("empty LLM response while normalizing knowledge")

        try:
            draft = KnowledgeDraft.model_validate(_extract_json_object(content))
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"knowledge normalization failed: {exc}") from exc
        draft = draft.with_fallbacks(clean_text)
        if self.taxonomy:
            draft = validate_draft_against_taxonomy(draft, self.taxonomy)
        return draft


def _knowledge_normalizer_extra_body(settings: LLMSettings) -> dict[str, Any] | None:
    extra_body = copy.deepcopy(settings.extra_body) if settings.extra_body else {}
    if settings.provider_profile != "qwen":
        return extra_body or None

    if "enable_thinking" in extra_body:
        extra_body["enable_thinking"] = False

    chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError("extra_body.chat_template_kwargs must be a JSON object")
    chat_template_kwargs["enable_thinking"] = False
    return extra_body


def taxonomy_reference_json(taxonomy: Taxonomy, *, max_values_per_column: int = 120) -> str:
    reference: dict[str, list[str]] = {}
    for column in TAXONOMY_COLUMNS:
        values: list[str] = []
        seen: set[str] = set()
        for row in taxonomy.rows:
            value = normalize_cell(row.get(column, ""))
            key = normalize_key(value)
            if not value or key in seen:
                continue
            seen.add(key)
            values.append(value)
            if len(values) >= max_values_per_column:
                break
        reference[column] = values
    reference["중직무/소직무"] = [
        f"{pair['중직무']} > {pair['소직무']}" for pair in taxonomy.pairs()[:max_values_per_column]
    ]
    return json.dumps(reference, ensure_ascii=False, indent=2)


def validate_draft_against_taxonomy(draft: KnowledgeDraft, taxonomy: Taxonomy) -> KnowledgeDraft:
    updates: dict[str, object] = {}
    errors = list(draft.validation_errors)
    field_to_column = {
        "target_major_job": "중직무",
        "target_sub_job": "소직무",
        "target_device": "Device",
        "target_unit_job": "단위 직무",
        "target_detail_job_1": "세부 직무1",
        "target_detail_job_2": "세부 직무2",
    }

    for field_name, column in field_to_column.items():
        raw_value = normalize_cell(getattr(draft, field_name))
        if not raw_value:
            continue
        canonical = _canonical_taxonomy_value(taxonomy, column, raw_value)
        if canonical is None:
            updates[field_name] = ""
            errors.append(f"{field_name}='{raw_value}' is not in taxonomy column '{column}'")
        else:
            updates[field_name] = canonical

    candidate = draft.model_copy(update=updates)
    specified = {
        column: normalize_cell(getattr(candidate, field_name))
        for field_name, column in field_to_column.items()
        if normalize_cell(getattr(candidate, field_name))
    }
    if specified and not _taxonomy_has_partial_path(taxonomy, specified):
        for field_name in field_to_column:
            updates[field_name] = ""
        errors.append("target fields do not form a valid taxonomy path; cleared target fields")

    if errors:
        updates["validation_errors"] = _dedupe_errors(errors)
    return draft.model_copy(update=updates)


def _canonical_taxonomy_value(taxonomy: Taxonomy, column: str, value: str) -> str | None:
    target_key = normalize_key(value)
    target_compact = compact_knowledge_key(value)
    for row in taxonomy.rows:
        candidate = normalize_cell(row.get(column, ""))
        if normalize_key(candidate) == target_key:
            return candidate
    for row in taxonomy.rows:
        candidate = normalize_cell(row.get(column, ""))
        if target_compact and compact_knowledge_key(candidate) == target_compact:
            return candidate
    return None


def _taxonomy_has_partial_path(taxonomy: Taxonomy, specified: dict[str, str]) -> bool:
    for row in taxonomy.rows:
        if all(normalize_key(row.get(column, "")) == normalize_key(value) for column, value in specified.items()):
            return True
    return False


def _dedupe_errors(errors: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for error in errors:
        text = normalize_cell(error)
        key = normalize_key(text)
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result[:20]


def fallback_knowledge_draft(raw_text: str) -> KnowledgeDraft:
    clean_text = normalize_cell(raw_text)
    aliases = []
    for phrase in re.findall(r"[0-9A-Za-z]+(?:[\s\-_]+[0-9A-Za-z]+)+", clean_text):
        phrase = normalize_cell(phrase)
        if normalize_key(phrase) not in {normalize_key(item) for item in aliases}:
            aliases.append(phrase)

    for token in re.findall(r"[0-9A-Za-z가-힣][0-9A-Za-z가-힣\-_/]*", clean_text):
        token = normalize_cell(re.sub(r"(과|와|이|가|은|는|을|를|에|에서|으로|로|의|도|만)$", "", token))
        if len(compact_knowledge_key(token)) >= 3 and normalize_key(token) not in {normalize_key(item) for item in aliases}:
            aliases.append(token)
        if len(aliases) >= 12:
            break
    return KnowledgeDraft(
        title=clean_text[:80],
        aliases=aliases,
        applies_when="사용자 입력 지식과 self_review 표현이 관련될 때",
        hint=clean_text,
        priority=50,
        confidence=0.35,
    ).with_fallbacks(clean_text)


def taxonomy_target_from_knowledge(knowledge: JobKnowledge) -> dict[str, str]:
    return {
        "중직무": knowledge.target_major_job,
        "소직무": knowledge.target_sub_job,
        "Device": knowledge.target_device,
        "단위 직무": knowledge.target_unit_job,
        "세부 직무1": knowledge.target_detail_job_1,
        "세부 직무2": knowledge.target_detail_job_2,
    }


def knowledge_from_taxonomy_target(raw_text: str, target: dict[str, object]) -> KnowledgeDraft:
    values = [normalize_cell(target.get(column, "")) for column in TAXONOMY_COLUMNS]
    aliases = [value for value in values if value]
    return KnowledgeDraft(
        title=normalize_cell(raw_text)[:80],
        aliases=aliases,
        hint=normalize_cell(raw_text),
        target_major_job=values[0],
        target_sub_job=values[1],
        target_device=values[2],
        target_unit_job=values[3],
        target_detail_job_1=values[4],
        target_detail_job_2=values[5],
        priority=50,
        confidence=0.5,
    ).with_fallbacks(raw_text)
