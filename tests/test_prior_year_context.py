import json
from types import MethodType

import pandas as pd

from llm_categorizing.classifier import ClassificationConfig, OpenAICompatibleJobClassifier
from llm_categorizing.cli import (
    build_previous_year_context,
    sort_employees_for_processing,
)
from llm_categorizing.config import LLMSettings
from llm_categorizing.taxonomy import Taxonomy


def _taxonomy() -> Taxonomy:
    return Taxonomy.from_rows(
        [
            {
                "중직무": "공정",
                "소직무": "Etch공정",
                "Device": "NAND",
                "단위 직무": "Chamber",
                "세부 직무1": "Clean",
                "세부 직무2": "",
            }
        ]
    )


def _classifier(taxonomy: Taxonomy) -> OpenAICompatibleJobClassifier:
    return OpenAICompatibleJobClassifier(
        settings=LLMSettings(
            base_url="http://localhost:1/v1",
            api_key="test",
            model="test",
        ),
        taxonomy=taxonomy,
        config=ClassificationConfig(),
    )


def test_sort_employees_for_processing_orders_each_employee_from_past_to_recent() -> None:
    employees = pd.DataFrame(
        [
            {"year": "2024", "team": "", "emp_num": "E0002", "name": "B", "self_review": ""},
            {"year": "2024", "team": "", "emp_num": "E0001", "name": "A", "self_review": ""},
            {"year": "2023", "team": "", "emp_num": "E0001", "name": "A", "self_review": ""},
            {"year": "2023", "team": "", "emp_num": "E0002", "name": "B", "self_review": ""},
        ]
    )

    ordered = sort_employees_for_processing(employees)

    assert list(zip(ordered["emp_num"], ordered["year"])) == [
        ("E0001", "2023"),
        ("E0001", "2024"),
        ("E0002", "2023"),
        ("E0002", "2024"),
    ]


def test_build_previous_year_context_uses_successful_previous_classification_only() -> None:
    previous = {
        "중직무": "공정",
        "소직무": "Etch공정",
        "Device": "NAND",
        "단위 직무": "Chamber",
        "세부 직무1": "Clean",
        "세부 직무2": "",
        "confidence": 0.88,
        "needs_review": False,
        "reason": "전년도 업무",
    }

    context = build_previous_year_context(2023, previous)
    string_context = build_previous_year_context(
        2023,
        {**previous, "confidence": "0.88", "needs_review": "False"},
    )
    failed_context = build_previous_year_context(2023, {**previous, "error": "failed"})
    review_context = build_previous_year_context(2023, {**previous, "needs_review": True})
    low_confidence_context = build_previous_year_context(2023, {**previous, "confidence": 0.59})

    assert context == {
        "year": "2023",
        "classification": {
            "중직무": "공정",
            "소직무": "Etch공정",
            "Device": "NAND",
            "단위 직무": "Chamber",
            "세부 직무1": "Clean",
            "세부 직무2": "",
        },
        "confidence": 0.88,
        "needs_review": False,
        "reason": "전년도 업무",
    }
    assert string_context == context
    assert failed_context is None
    assert review_context is None
    assert low_confidence_context is None


def test_classifier_includes_previous_year_classification_in_prompt_and_output() -> None:
    classifier = _classifier(_taxonomy())
    captured: dict[str, object] = {}

    def fixed_uncached(self, context_json, diagnosis_priority, knowledge_items):
        captured["context"] = json.loads(context_json)
        return {
            "중직무": "공정",
            "소직무": "Etch공정",
            "Device": "NAND",
            "단위 직무": "Chamber",
            "세부 직무1": "Clean",
            "세부 직무2": "",
            "confidence": 0.91,
            "reason": "현재 연도 업무와 전년도 연속성",
            "needs_review": False,
            "ambiguity_reason": "",
            "guardrail_reason": "",
            "diagnosis_priority_reason": "",
            "knowledge_priority_reason": "",
            "error": "",
        }

    classifier._classify_uncached = MethodType(fixed_uncached, classifier)
    previous_context = build_previous_year_context(
        2023,
        {
            "중직무": "공정",
            "소직무": "Etch공정",
            "Device": "NAND",
            "단위 직무": "Chamber",
            "세부 직무1": "Clean",
            "세부 직무2": "",
            "confidence": 0.88,
            "needs_review": False,
            "reason": "전년도 Etch 업무",
        },
    )

    result = classifier.classify_row(
        {
            "year": "2024",
            "team": "",
            "emp_num": "E0001",
            "name": "A",
            "self_review": "Etch chamber 업무 지속",
        },
        previous_year_context=previous_context,
    )

    previous_payload = captured["context"]["previous_year_classification"]
    assert previous_payload["year"] == "2023"
    assert previous_payload["classification"]["중직무"] == "공정"
    assert previous_payload["is_soft_continuity_hint"] is True
    assert result["previous_year"] == "2023"
    assert result["previous_year_job_path"] == "공정 > Etch공정 > NAND > Chamber > Clean"
