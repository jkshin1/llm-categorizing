from __future__ import annotations

from typing import Any

from llm_categorizing.cli import (
    is_empty_llm_response_error,
    retry_empty_llm_response_rows,
)


def _source_row(year: str) -> dict[str, str]:
    return {
        "year": year,
        "team": "",
        "emp_num": "E0001",
        "name": "A",
        "self_review": f"{year} 업무 수행",
    }


def _successful_result(year: str) -> dict[str, object]:
    return {
        "중직무": "공정",
        "소직무": "Etch공정",
        "Device": "NAND",
        "단위 직무": f"Unit {year}",
        "세부 직무1": "",
        "세부 직무2": "",
        "confidence": 0.91,
        "reason": "retry success",
        "needs_review": False,
        "ambiguity_reason": "",
        "guardrail_reason": "",
        "diagnosis_priority_reason": "",
        "knowledge_priority_reason": "",
        "error": "",
    }


class FakeClassifier:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def classify_row(
        self,
        row: dict[str, Any],
        *,
        diagnosis_context: Any = None,
        previous_year_context: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "row": row,
                "diagnosis_context": diagnosis_context,
                "previous_year_context": previous_year_context,
            }
        )
        return _successful_result(str(row["year"]))


def test_empty_llm_response_detector_matches_detailed_error() -> None:
    assert is_empty_llm_response_error(
        {
            "error": (
                "classification_error: empty LLM response "
                "(finish_reason=stop, message_non_empty_keys=[])"
            )
        }
    )
    assert not is_empty_llm_response_error(
        {"error": "classification_error: LLM unavailable"}
    )
    assert not is_empty_llm_response_error({"error": ""})


def test_retry_empty_llm_response_rows_retries_only_matching_errors() -> None:
    classifier = FakeClassifier()
    source_rows = [_source_row("2024"), _source_row("2025")]
    output_rows = [
        {"error": "classification_error: empty LLM response (finish_reason=stop)"},
        {"error": "classification_error: LLM unavailable"},
    ]

    attempted, classified = retry_empty_llm_response_rows(
        source_rows=source_rows,
        output_rows=output_rows,
        classifier=classifier,  # type: ignore[arg-type]
        diagnosis_contexts={},
        previous_results={},
        include_self_review_output=False,
        min_confidence=0.6,
    )

    assert attempted == 1
    assert classified == 1
    assert len(classifier.calls) == 1
    assert classifier.calls[0]["row"]["year"] == "2024"
    assert output_rows[0]["error"] == ""
    assert output_rows[0]["단위 직무"] == "Unit 2024"
    assert output_rows[1]["error"] == "classification_error: LLM unavailable"


def test_retry_empty_llm_response_rows_updates_previous_year_between_retries() -> None:
    classifier = FakeClassifier()
    source_rows = [_source_row("2023"), _source_row("2024")]
    output_rows = [
        {"error": "classification_error: empty LLM response (finish_reason=stop)"},
        {"error": "classification_error: empty LLM response (finish_reason=stop)"},
    ]

    attempted, classified = retry_empty_llm_response_rows(
        source_rows=source_rows,
        output_rows=output_rows,
        classifier=classifier,  # type: ignore[arg-type]
        diagnosis_contexts={},
        previous_results={},
        include_self_review_output=False,
        min_confidence=0.6,
    )

    assert attempted == 2
    assert classified == 2
    assert classifier.calls[0]["previous_year_context"] is None
    previous_year_context = classifier.calls[1]["previous_year_context"]
    assert previous_year_context["year"] == "2023"
    assert previous_year_context["classification"]["단위 직무"] == "Unit 2023"
