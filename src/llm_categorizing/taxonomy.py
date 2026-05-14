from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


EMPLOYEE_COLUMNS = ["year", "team", "emp_num", "name", "self_review"]
TAXONOMY_COLUMNS = ["중직무", "소직무", "Device", "단위 직무", "세부 직무1", "세부 직무2"]


def normalize_cell(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_key(value: object) -> str:
    return normalize_cell(value).casefold()


def read_csv_with_fallback(path: str | Path, encoding: str = "utf-8-sig") -> pd.DataFrame:
    path = Path(path)
    encodings = [encoding, "utf-8-sig", "cp949", "euc-kr", "utf-8"]
    seen: set[str] = set()
    last_error: Exception | None = None

    for enc in encodings:
        if enc in seen:
            continue
        seen.add(enc)
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error:
        raise last_error
    raise ValueError(f"Could not read CSV file: {path}")


def require_columns(df: pd.DataFrame, required_columns: Iterable[str], label: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


@dataclass(frozen=True)
class Taxonomy:
    rows: list[dict[str, str]]

    @classmethod
    def from_csv(cls, path: str | Path, encoding: str = "utf-8-sig") -> "Taxonomy":
        df = read_csv_with_fallback(path, encoding=encoding)
        require_columns(df, TAXONOMY_COLUMNS, "taxonomy CSV")
        return cls.from_rows(df[TAXONOMY_COLUMNS].to_dict(orient="records"))

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, object]]) -> "Taxonomy":
        clean_rows: list[dict[str, str]] = []
        seen: set[tuple[str, ...]] = set()

        for raw_row in rows:
            row = {column: normalize_cell(raw_row.get(column, "")) for column in TAXONOMY_COLUMNS}
            if not any(row.values()):
                continue
            key = tuple(normalize_key(row[column]) for column in TAXONOMY_COLUMNS)
            if key in seen:
                continue
            seen.add(key)
            clean_rows.append(row)

        if not clean_rows:
            raise ValueError("taxonomy CSV has no usable rows")

        return cls(rows=clean_rows)

    def version_hash(self) -> str:
        payload = json.dumps(self.rows, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def pairs(self) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in self.rows:
            key = (normalize_key(row["중직무"]), normalize_key(row["소직무"]))
            if key in seen:
                continue
            seen.add(key)
            result.append({"중직무": row["중직무"], "소직무": row["소직무"]})
        return result

    def canonical_pair(self, major_job: object, sub_job: object) -> dict[str, str] | None:
        wanted = (normalize_key(major_job), normalize_key(sub_job))
        for pair in self.pairs():
            key = (normalize_key(pair["중직무"]), normalize_key(pair["소직무"]))
            if key == wanted:
                return pair
        return None

    def children_for_pair(self, major_job: object, sub_job: object) -> list[dict[str, str]]:
        wanted = (normalize_key(major_job), normalize_key(sub_job))
        children: list[dict[str, str]] = []
        for row in self.rows:
            key = (normalize_key(row["중직무"]), normalize_key(row["소직무"]))
            if key == wanted:
                children.append(dict(row))
        return children

    def major_jobs_for_unit_job(self, unit_job: object) -> list[str]:
        wanted = normalize_key(unit_job)
        if not wanted:
            return []

        result: list[str] = []
        seen: set[str] = set()
        for row in self.rows:
            if normalize_key(row["단위 직무"]) != wanted:
                continue
            major_job = row["중직무"]
            key = normalize_key(major_job)
            if key in seen:
                continue
            seen.add(key)
            result.append(major_job)
        return result

    def rows_for_unit_job(
        self,
        unit_job: object,
        *,
        major_job: object | None = None,
        device: object | None = None,
    ) -> list[dict[str, str]]:
        wanted_unit = normalize_key(unit_job)
        wanted_major = normalize_key(major_job) if major_job is not None else ""
        wanted_device = normalize_key(device) if device is not None else ""
        if not wanted_unit:
            return []

        result: list[dict[str, str]] = []
        for row in self.rows:
            if normalize_key(row["단위 직무"]) != wanted_unit:
                continue
            if wanted_major and normalize_key(row["중직무"]) != wanted_major:
                continue
            if wanted_device and normalize_key(row["Device"]) != wanted_device:
                continue
            result.append(dict(row))
        return result

    def ambiguous_unit_jobs_in_text(self, text: object) -> list[dict[str, Any]]:
        text_key = normalize_key(text)
        if not text_key:
            return []

        result: list[dict[str, Any]] = []
        seen_units: set[str] = set()
        for row in self.rows:
            unit_job = row["단위 직무"]
            unit_key = normalize_key(unit_job)
            if len(unit_key) < 2 or unit_key in seen_units or unit_key not in text_key:
                continue

            major_jobs = self.major_jobs_for_unit_job(unit_job)
            if len(major_jobs) > 1:
                result.append({"단위 직무": unit_job, "중복 중직무": major_jobs})
            seen_units.add(unit_key)
        return result

    def ambiguity_reason_for_row(self, row: dict[str, object]) -> str:
        unit_job = normalize_cell(row.get("단위 직무", ""))
        major_jobs = self.major_jobs_for_unit_job(unit_job)
        if len(major_jobs) <= 1:
            return ""
        joined = ", ".join(major_jobs)
        return f"단위 직무 '{unit_job}'이 여러 중직무({joined})에 존재하므로 self_review의 실제 업무 동사와 산출물 기준으로 검수 필요"

    def annotate_candidates(self, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
        annotated: list[dict[str, Any]] = []
        for row in rows:
            item: dict[str, Any] = dict(row)
            ambiguity_reason = self.ambiguity_reason_for_row(row)
            if ambiguity_reason:
                item["분류주의"] = ambiguity_reason
            annotated.append(item)
        return annotated

    def canonical_path(self, candidate: dict[str, object]) -> dict[str, str] | None:
        wanted = tuple(normalize_key(candidate.get(column, "")) for column in TAXONOMY_COLUMNS)
        for row in self.rows:
            key = tuple(normalize_key(row[column]) for column in TAXONOMY_COLUMNS)
            if key == wanted:
                return dict(row)
        return None

    def format_candidates_json(self, rows: list[dict[str, Any]]) -> str:
        return json.dumps(rows, ensure_ascii=False, indent=2)
