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
from llm_categorizing.taxonomy import (
    EMPLOYEE_COLUMNS,
    TAXONOMY_COLUMNS,
    Taxonomy,
    read_csv_with_fallback,
    require_columns,
)


DEFAULT_INPUT_PATH = "data/input/employees.csv"
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
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"분류 결과 CSV 경로. 기본값: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument("--encoding", default="utf-8-sig", help="입력 CSV 기본 인코딩")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N개 행만 처리")
    parser.add_argument("--validate-only", action="store_true", help="스키마만 검증하고 종료")
    parser.add_argument(
        "--exclude-team-from-prompt",
        action="store_true",
        help="LLM 프롬프트에서 team 컬럼도 제외",
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
    parser.add_argument("--max-review-chars", type=int, default=12000)
    parser.add_argument("--max-candidates-per-prompt", type=int, default=300)
    parser.add_argument("--validation-attempts", type=int, default=2)
    parser.add_argument("--api-retry-attempts", type=int, default=3)
    parser.add_argument("--confidence-review-threshold", type=float, default=0.6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    employees = read_csv_with_fallback(args.input, encoding=args.encoding)
    require_columns(employees, EMPLOYEE_COLUMNS, "employee CSV")
    taxonomy = Taxonomy.from_csv(args.taxonomy, encoding=args.encoding)

    if args.limit > 0:
        employees = employees.head(args.limit).copy()

    print(
        f"Loaded employees={len(employees)}, taxonomy_rows={len(taxonomy.rows)}, "
        f"taxonomy_pairs={len(taxonomy.pairs())}, taxonomy_version={taxonomy.version_hash()}"
    )

    if args.validate_only:
        print("Validation completed. No LLM calls were made.")
        return 0

    settings = LLMSettings.from_env()
    config = ClassificationConfig(
        include_team_in_prompt=not args.exclude_team_from_prompt,
        max_review_chars=args.max_review_chars,
        max_candidates_per_prompt=args.max_candidates_per_prompt,
        validation_attempts=args.validation_attempts,
        api_retry_attempts=args.api_retry_attempts,
        confidence_review_threshold=args.confidence_review_threshold,
    )
    cache = JsonlCache(args.cache_path or None)
    classifier = OpenAICompatibleJobClassifier(
        settings=settings,
        taxonomy=taxonomy,
        config=config,
        cache=cache,
    )

    output_rows: list[dict[str, Any]] = []
    total = len(employees)
    for index, raw_row in employees.iterrows():
        row = {column: raw_row.get(column, "") for column in EMPLOYEE_COLUMNS}
        result = classifier.classify_row(row)
        output_rows.append(build_output_row(row, result, args.include_self_review_output))

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
            "error": result.get("error", ""),
            "input_truncated": result.get("input_truncated", False),
            "taxonomy_version": result.get("taxonomy_version", ""),
            "model_name": result.get("model_name", ""),
            "classified_at": result.get("classified_at", ""),
        }
    )
    return output
