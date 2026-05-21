from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from llm_categorizing.taxonomy import normalize_cell, normalize_key, read_csv_with_fallback, require_columns


DIAGNOSIS_COLUMNS = ["year", "emp_num", "name", "team", "진단 시 직무명", "Category", "항목"]


@dataclass(frozen=True)
class DiagnosisContext:
    year: str
    emp_num: str
    row_count: int
    teams: list[str]
    job_names: list[str]
    categories: list[str]
    evidence_rows: list[dict[str, str]]

    def to_prompt_payload(self) -> dict[str, Any]:
        evidence_rows = [
            {
                "diagnosis_team": row.get("diagnosis_team", ""),
                "diagnosis_job_name": row.get("diagnosis_job_name", ""),
                "category": row.get("category", ""),
            }
            for row in self.evidence_rows
        ]
        return {
            "row_count": self.row_count,
            "diagnosis_teams": self.teams,
            "diagnosis_job_names": self.job_names,
            "categories": self.categories,
            "evidence_rows": [row for row in evidence_rows if any(row.values())],
        }

    def to_output_payload(self) -> dict[str, Any]:
        return {
            "diagnosis_row_count": self.row_count,
            "diagnosis_teams": " | ".join(self.teams),
            "diagnosis_job_names": " | ".join(self.job_names),
            "diagnosis_categories": " | ".join(self.categories),
        }


def load_diagnosis_contexts(
    path: str | Path,
    *,
    encoding: str = "utf-8-sig",
    max_evidence_rows: int = 50,
    max_values_per_field: int = 30,
) -> dict[tuple[str, str], DiagnosisContext]:
    df = read_csv_with_fallback(path, encoding=encoding)
    require_columns(df, DIAGNOSIS_COLUMNS, "diagnosis CSV")

    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for raw_row in df[DIAGNOSIS_COLUMNS].to_dict(orient="records"):
        year = normalize_cell(raw_row.get("year", ""))
        emp_num = normalize_cell(raw_row.get("emp_num", ""))
        if not year or not emp_num:
            continue
        row = {column: normalize_cell(raw_row.get(column, "")) for column in DIAGNOSIS_COLUMNS}
        grouped_rows[(normalize_key(year), normalize_key(emp_num))].append(row)

    contexts: dict[tuple[str, str], DiagnosisContext] = {}
    for key, rows in grouped_rows.items():
        year = rows[0]["year"]
        emp_num = rows[0]["emp_num"]
        contexts[key] = DiagnosisContext(
            year=year,
            emp_num=emp_num,
            row_count=len(rows),
            teams=_unique_values(rows, "team", max_values_per_field),
            job_names=_unique_values(rows, "진단 시 직무명", max_values_per_field),
            categories=_unique_values(rows, "Category", max_values_per_field),
            evidence_rows=_evidence_rows(rows, max_evidence_rows),
        )
    return contexts


def diagnosis_key(year: object, emp_num: object) -> tuple[str, str]:
    return (normalize_key(year), normalize_key(emp_num))


def empty_diagnosis_output_payload() -> dict[str, Any]:
    return {
        "diagnosis_row_count": 0,
        "diagnosis_teams": "",
        "diagnosis_job_names": "",
        "diagnosis_categories": "",
    }


def _unique_values(rows: list[dict[str, str]], column: str, limit: int) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = normalize_cell(row.get(column, ""))
        key = normalize_key(value)
        if not value or key in seen:
            continue
        seen.add(key)
        values.append(value)
        if len(values) >= limit:
            break
    return values


def _evidence_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        item = {
            "diagnosis_team": row.get("team", ""),
            "diagnosis_job_name": row.get("진단 시 직무명", ""),
            "category": row.get("Category", ""),
        }
        if not any(item.values()):
            continue
        key = tuple(normalize_key(value) for value in item.values())
        if key in seen:
            continue
        seen.add(key)
        evidence.append(item)
        if len(evidence) >= limit:
            break
    return evidence
