from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from llm_categorizing.classifier import (
    ClassificationConfig,
    JsonlCache,
    OpenAICompatibleJobClassifier,
)
from llm_categorizing.config import LLMSettings
from llm_categorizing.diagnosis import (
    diagnosis_key,
    empty_diagnosis_output_payload,
    load_diagnosis_contexts,
)
from llm_categorizing.knowledge import DEFAULT_KNOWLEDGE_DB_PATH, JobKnowledgeStore
from llm_categorizing.taxonomy import (
    EMPLOYEE_COLUMNS,
    TAXONOMY_COLUMNS,
    Taxonomy,
    normalize_cell,
    normalize_key,
    read_csv_with_fallback,
    require_columns,
)


DEFAULT_INPUT_PATH = "data/input/employees.csv"
DEFAULT_DIAGNOSIS_PATH = "data/input/diagnosis.csv"
DEFAULT_TAXONOMY_PATH = "data/input/taxonomy.csv"
DEFAULT_OUTPUT_PATH = "data/output/classified_jobs.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify employee self-review rows into a fixed job taxonomy."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"구성원 CSV 경로. 기본값: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--taxonomy",
        default=DEFAULT_TAXONOMY_PATH,
        help=f"직무 taxonomy CSV 경로. 기본값: {DEFAULT_TAXONOMY_PATH}",
    )
    parser.add_argument(
        "--diagnosis",
        default=DEFAULT_DIAGNOSIS_PATH,
        help=f"직무/스킬셋 진단 CSV 경로. 파일이 있으면 자동 사용. 기본값: {DEFAULT_DIAGNOSIS_PATH}",
    )
    parser.add_argument(
        "--no-diagnosis",
        action="store_true",
        help="진단 CSV를 사용하지 않고 self_review만으로 분류",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"분류 결과 CSV 경로. 기본값: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument("--encoding", default="utf-8-sig", help="입력 CSV 기본 인코딩")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N개 행만 처리")
    parser.add_argument("--validate-only", action="store_true", help="스키마만 검증하고 종료")
    parser.add_argument(
        "--include-team-in-prompt",
        action="store_true",
        help="LLM 프롬프트에 team 컬럼 포함. 연도별 팀 정보가 정확할 때만 사용",
    )
    parser.add_argument(
        "--exclude-team-from-prompt",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--include-self-review-output",
        action="store_true",
        help="결과 CSV에 self_review 원문 포함",
    )
    parser.add_argument(
        "--cache-path",
        default="data/output/classification_cache.jsonl",
        help="동일 입력 재호출 방지를 위한 JSONL cache 경로. 빈 문자열이면 비활성화",
    )
    parser.add_argument(
        "--knowledge-db-path",
        default=DEFAULT_KNOWLEDGE_DB_PATH,
        help="사용자 입력 직무 지식 SQLite DB 경로. 빈 문자열이면 비활성화",
    )
    parser.add_argument("--max-review-chars", type=int, default=12000)
    parser.add_argument("--max-diagnosis-rows-per-employee", type=int, default=50)
    parser.add_argument("--max-candidates-per-prompt", type=int, default=300)
    parser.add_argument("--max-knowledge-hints", type=int, default=8)
    parser.add_argument(
        "--knowledge-review-scope",
        choices=["usable", "approved"],
        default="usable",
        help="분류에 사용할 지식 범위. usable은 active draft/approved를 모두 검색하고, approved는 승인 지식만 사용",
    )
    parser.add_argument("--validation-attempts", type=int, default=2)
    parser.add_argument("--api-retry-attempts", type=int, default=5)
    parser.add_argument("--confidence-review-threshold", type=float, default=0.6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    employees = read_csv_with_fallback(args.input, encoding=args.encoding)
    require_columns(employees, EMPLOYEE_COLUMNS, "employee CSV")
    taxonomy = Taxonomy.from_csv(args.taxonomy, encoding=args.encoding)
    diagnosis_contexts = {}
    diagnosis_path = Path(args.diagnosis)
    if not args.no_diagnosis and diagnosis_path.exists():
        diagnosis_contexts = load_diagnosis_contexts(
            diagnosis_path,
            encoding=args.encoding,
            max_evidence_rows=args.max_diagnosis_rows_per_employee,
        )
    elif not args.no_diagnosis:
        print(f"Diagnosis CSV not found. Skipping diagnosis context: {diagnosis_path}")

    if args.limit > 0:
        employees = employees.head(args.limit).copy()
    employees = sort_employees_for_processing(employees)

    print(
        f"Loaded employees={len(employees)}, taxonomy_rows={len(taxonomy.rows)}, "
        f"taxonomy_pairs={len(taxonomy.pairs())}, diagnosis_keys={len(diagnosis_contexts)}, "
        f"taxonomy_version={taxonomy.version_hash()}"
    )

    if args.validate_only:
        print("Validation completed. No LLM calls were made.")
        return 0

    settings = LLMSettings.from_env(role="classification")
    config = ClassificationConfig(
        include_team_in_prompt=args.include_team_in_prompt and not args.exclude_team_from_prompt,
        max_review_chars=args.max_review_chars,
        max_candidates_per_prompt=args.max_candidates_per_prompt,
        max_knowledge_hints=args.max_knowledge_hints,
        knowledge_review_scope=args.knowledge_review_scope,
        validation_attempts=args.validation_attempts,
        api_retry_attempts=args.api_retry_attempts,
        confidence_review_threshold=args.confidence_review_threshold,
    )
    cache = JsonlCache(args.cache_path or None)
    knowledge_store = JobKnowledgeStore(args.knowledge_db_path) if args.knowledge_db_path else None
    classifier = OpenAICompatibleJobClassifier(
        settings=settings,
        taxonomy=taxonomy,
        config=config,
        cache=cache,
        knowledge_store=knowledge_store,
    )

    output_rows: list[dict[str, Any]] = []
    previous_results: dict[tuple[str, int], dict[str, Any]] = {}
    total = len(employees)
    for index, raw_row in employees.iterrows():
        row = {column: raw_row.get(column, "") for column in EMPLOYEE_COLUMNS}
        diagnosis_context = diagnosis_contexts.get(diagnosis_key(row.get("year", ""), row.get("emp_num", "")))
        employee_key = normalize_key(row.get("emp_num", ""))
        year_number = parse_year_number(row.get("year", ""))
        previous_year_context = None
        if employee_key and year_number is not None:
            previous_result = previous_results.get((employee_key, year_number - 1))
            previous_year_context = build_previous_year_context(year_number - 1, previous_result)

        result = classifier.classify_row(
            row,
            diagnosis_context=diagnosis_context,
            previous_year_context=previous_year_context,
        )
        output_rows.append(build_output_row(row, result, args.include_self_review_output))
        if employee_key and year_number is not None:
            previous_results[(employee_key, year_number)] = result

        current = len(output_rows)
        if current == 1 or current == total or current % 10 == 0:
            needs_review_count = sum(1 for item in output_rows if item.get("needs_review") is True)
            print(f"Processed {current}/{total} rows, needs_review={needs_review_count}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote result CSV: {output_path}")
    return 0


def build_output_row(
    source_row: dict[str, Any],
    result: dict[str, Any],
    include_self_review: bool,
) -> dict[str, Any]:
    output = {
        "year": source_row.get("year", ""),
        "team": source_row.get("team", ""),
        "emp_num": source_row.get("emp_num", ""),
        "name": source_row.get("name", ""),
    }
    if include_self_review:
        output["self_review"] = source_row.get("self_review", "")

    for column in TAXONOMY_COLUMNS:
        output[column] = result.get(column, "")

    output.update(
        {
            "confidence": result.get("confidence", 0.0),
            "reason": result.get("reason", ""),
            "needs_review": result.get("needs_review", True),
            "ambiguity_reason": result.get("ambiguity_reason", ""),
            "guardrail_reason": result.get("guardrail_reason", ""),
            "diagnosis_priority_reason": result.get("diagnosis_priority_reason", ""),
            "previous_year": result.get("previous_year", ""),
            "previous_year_job_path": result.get("previous_year_job_path", ""),
            "previous_year_confidence": result.get("previous_year_confidence", ""),
            "previous_year_needs_review": result.get("previous_year_needs_review", ""),
            "used_knowledge_ids": json_list(result.get("used_knowledge_ids", [])),
            "used_knowledge_types": json_list(result.get("used_knowledge_types", [])),
            "used_knowledge_scores": json_list(result.get("used_knowledge_scores", [])),
            "used_knowledge_review_statuses": json_list(
                result.get("used_knowledge_review_statuses", [])
            ),
            "used_knowledge_match_fields": json_list(
                result.get("used_knowledge_match_fields", [])
            ),
            "knowledge_review_scope": result.get("knowledge_review_scope", ""),
            "knowledge_version": result.get("knowledge_version", ""),
            **{
                key: result.get(key, default)
                for key, default in empty_diagnosis_output_payload().items()
            },
            "error": result.get("error", ""),
            "input_truncated": result.get("input_truncated", False),
            "taxonomy_version": result.get("taxonomy_version", ""),
            "model_name": result.get("model_name", ""),
            "classified_at": result.get("classified_at", ""),
        }
    )
    return output


def sort_employees_for_processing(employees: pd.DataFrame) -> pd.DataFrame:
    ordered = employees.copy()
    ordered["_input_order"] = range(len(ordered))
    ordered["_employee_sort_key"] = [
        employee_processing_key(raw_row) for _, raw_row in ordered.iterrows()
    ]
    ordered["_year_sort_key"] = [year_sort_key(value) for value in ordered["year"]]
    ordered = ordered.sort_values(
        by=["_employee_sort_key", "_year_sort_key", "_input_order"],
        kind="mergesort",
    )
    return ordered.drop(columns=["_input_order", "_employee_sort_key", "_year_sort_key"])


def employee_processing_key(row: pd.Series) -> str:
    emp_num = normalize_key(row.get("emp_num", ""))
    if emp_num:
        return emp_num
    return normalize_key(row.get("name", ""))


def year_sort_key(value: object) -> tuple[int, str]:
    parsed = parse_year_number(value)
    if parsed is None:
        return (10**9, normalize_cell(value))
    return (parsed, normalize_cell(value))


def parse_year_number(value: object) -> int | None:
    text = normalize_cell(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        pass
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) >= 4:
        return int(digits[:4])
    return None


def build_previous_year_context(
    previous_year: int,
    previous_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not previous_result or previous_result.get("error"):
        return None
    classification = {
        column: normalize_cell(previous_result.get(column, ""))
        for column in TAXONOMY_COLUMNS
    }
    if not any(classification.values()):
        return None
    return {
        "year": str(previous_year),
        "classification": classification,
        "confidence": previous_result.get("confidence", 0.0),
        "needs_review": previous_result.get("needs_review", True),
        "reason": normalize_cell(previous_result.get("reason", "")),
    }


def json_list(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "")
