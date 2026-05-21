from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from llm_categorizing.taxonomy import TAXONOMY_COLUMNS, normalize_cell, normalize_key, read_csv_with_fallback, require_columns


KEY_COLUMNS = ["year", "emp_num"]


def evaluate_predictions(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    top_confusions: int = 20,
) -> dict[str, Any]:
    require_columns(predictions, KEY_COLUMNS + TAXONOMY_COLUMNS, "prediction CSV")
    require_columns(gold, KEY_COLUMNS + TAXONOMY_COLUMNS, "gold CSV")

    prediction_rows = {
        _row_key(row): row
        for row in predictions.to_dict(orient="records")
        if all(normalize_cell(row.get(column, "")) for column in KEY_COLUMNS)
    }
    gold_rows = [
        row
        for row in gold.to_dict(orient="records")
        if all(normalize_cell(row.get(column, "")) for column in KEY_COLUMNS)
    ]

    matched = 0
    exact_correct = 0
    pair_correct = 0
    column_correct = {column: 0 for column in TAXONOMY_COLUMNS}
    missing_keys: list[str] = []
    needs_review_count = 0
    confusion_counter: Counter[tuple[str, str]] = Counter()

    for gold_row in gold_rows:
        key = _row_key(gold_row)
        prediction_row = prediction_rows.get(key)
        if prediction_row is None:
            missing_keys.append(_display_key(gold_row))
            continue

        matched += 1
        if _truthy(prediction_row.get("needs_review", False)):
            needs_review_count += 1

        column_matches = {
            column: _same_value(prediction_row.get(column, ""), gold_row.get(column, ""))
            for column in TAXONOMY_COLUMNS
        }
        for column, is_match in column_matches.items():
            if is_match:
                column_correct[column] += 1

        if column_matches["중직무"] and column_matches["소직무"]:
            pair_correct += 1
        if all(column_matches.values()):
            exact_correct += 1
        else:
            confusion_counter[(_path_text(gold_row), _path_text(prediction_row))] += 1

    prediction_keys = {_row_key(row) for row in predictions.to_dict(orient="records")}
    gold_keys = {_row_key(row) for row in gold_rows}
    extra_prediction_count = len(prediction_keys - gold_keys)

    return {
        "gold_count": len(gold_rows),
        "prediction_count": len(prediction_rows),
        "matched_count": matched,
        "missing_prediction_count": len(missing_keys),
        "extra_prediction_count": extra_prediction_count,
        "exact_path_accuracy": _ratio(exact_correct, matched),
        "pair_accuracy": _ratio(pair_correct, matched),
        "column_accuracy": {
            column: _ratio(correct, matched) for column, correct in column_correct.items()
        },
        "needs_review_rate": _ratio(needs_review_count, matched),
        "top_confusions": [
            {
                "expected": expected,
                "predicted": predicted,
                "count": count,
            }
            for (expected, predicted), count in confusion_counter.most_common(top_confusions)
        ],
        "missing_keys_sample": missing_keys[:top_confusions],
    }


def evaluate_prediction_files(
    predictions_path: str | Path,
    gold_path: str | Path,
    *,
    encoding: str = "utf-8-sig",
    top_confusions: int = 20,
) -> dict[str, Any]:
    predictions = read_csv_with_fallback(predictions_path, encoding=encoding)
    gold = read_csv_with_fallback(gold_path, encoding=encoding)
    return evaluate_predictions(predictions, gold, top_confusions=top_confusions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate classified job CSV against gold labels.")
    parser.add_argument("--predictions", required=True, help="classified_jobs.csv 경로")
    parser.add_argument("--gold", required=True, help="정답 라벨 CSV 경로")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--top-confusions", type=int, default=20)
    parser.add_argument("--json-output", action="store_true", help="평가 결과를 JSON으로 출력")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_prediction_files(
        args.predictions,
        args.gold,
        encoding=args.encoding,
        top_confusions=args.top_confusions,
    )
    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(_format_summary(result))
    return 0


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return tuple(normalize_key(row.get(column, "")) for column in KEY_COLUMNS)


def _display_key(row: dict[str, Any]) -> str:
    return " / ".join(normalize_cell(row.get(column, "")) for column in KEY_COLUMNS)


def _same_value(left: object, right: object) -> bool:
    return normalize_key(left) == normalize_key(right)


def _truthy(value: object) -> bool:
    return normalize_cell(value).casefold() in {"1", "true", "t", "yes", "y"}


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _path_text(row: dict[str, Any]) -> str:
    return " > ".join(
        normalize_cell(row.get(column, ""))
        for column in TAXONOMY_COLUMNS
        if normalize_cell(row.get(column, ""))
    )


def _format_summary(result: dict[str, Any]) -> str:
    lines = [
        f"gold_count={result['gold_count']}",
        f"prediction_count={result['prediction_count']}",
        f"matched_count={result['matched_count']}",
        f"missing_prediction_count={result['missing_prediction_count']}",
        f"extra_prediction_count={result['extra_prediction_count']}",
        f"exact_path_accuracy={result['exact_path_accuracy']:.4f}",
        f"pair_accuracy={result['pair_accuracy']:.4f}",
        f"needs_review_rate={result['needs_review_rate']:.4f}",
    ]
    if result["top_confusions"]:
        lines.append("top_confusions:")
        for item in result["top_confusions"]:
            lines.append(
                f"- count={item['count']} expected={item['expected']} predicted={item['predicted']}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
