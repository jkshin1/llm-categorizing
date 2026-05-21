from types import MethodType

from llm_categorizing.classifier import ClassificationConfig, OpenAICompatibleJobClassifier
from llm_categorizing.cli import build_output_row
from llm_categorizing.config import LLMSettings
from llm_categorizing.diagnosis import DiagnosisContext
from llm_categorizing.models import FinalClassificationResult
from llm_categorizing.taxonomy import Taxonomy


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


def _taxonomy() -> Taxonomy:
    return Taxonomy.from_rows(
        [
            {
                "중직무": "공정",
                "소직무": "Etch공정",
                "Device": "DRAM",
                "단위 직무": "Chamber",
                "세부 직무1": "Clean",
                "세부 직무2": "",
            },
            {
                "중직무": "공정",
                "소직무": "Etch공정",
                "Device": "NAND",
                "단위 직무": "Chamber",
                "세부 직무1": "Clean",
                "세부 직무2": "",
            },
            {
                "중직무": "품질",
                "소직무": "Quality",
                "Device": "Common",
                "단위 직무": "불량분석",
                "세부 직무1": "Failure Analysis",
                "세부 직무2": "",
            },
        ]
    )


def test_diagnosis_priority_infers_pair_from_diagnosis_job_name_only() -> None:
    classifier = _classifier(_taxonomy())
    diagnosis_context = DiagnosisContext(
        year="2025",
        emp_num="E0001",
        row_count=2,
        teams=["DRAM공정 > DPC"],
        job_names=["Etch공정"],
        categories=[],
        evidence_rows=[],
    )

    priority = classifier._diagnosis_priority(diagnosis_context)
    pair_candidates, pair_reason = classifier._diagnosis_pair_candidates(priority)

    assert priority.major_job == "공정"
    assert priority.sub_job == "Etch공정"
    assert pair_candidates == [{"중직무": "공정", "소직무": "Etch공정"}]
    assert "중직무 '공정'" in pair_reason
    assert "Device" not in priority.reason


def test_diagnosis_job_name_can_fall_back_to_unique_unit_job_pair() -> None:
    classifier = _classifier(_taxonomy())
    diagnosis_context = DiagnosisContext(
        year="2025",
        emp_num="E0002",
        row_count=1,
        teams=["품질분석팀"],
        job_names=["불량분석"],
        categories=[],
        evidence_rows=[],
    )

    priority = classifier._diagnosis_priority(diagnosis_context)

    assert priority.major_job == "품질"
    assert priority.sub_job == "Quality"
    assert "단위 직무 '불량분석'" in priority.reason


def test_diagnosis_partial_job_name_does_not_hard_limit_pair() -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "소자",
                "소직무": "PI",
                "Device": "",
                "단위 직무": "공정통합",
                "세부 직무1": "",
                "세부 직무2": "",
            },
            {
                "중직무": "품질",
                "소직무": "QA",
                "Device": "",
                "단위 직무": "불량분석",
                "세부 직무1": "",
                "세부 직무2": "",
            },
        ]
    )
    classifier = _classifier(taxonomy)
    diagnosis_context = DiagnosisContext(
        year="2025",
        emp_num="E0002",
        row_count=1,
        teams=[],
        job_names=["공정"],
        categories=[],
        evidence_rows=[],
    )

    priority = classifier._diagnosis_priority(diagnosis_context)
    pair_candidates, pair_reason = classifier._diagnosis_pair_candidates(priority)

    assert priority.major_job == ""
    assert priority.sub_job == ""
    assert pair_candidates == classifier.taxonomy.pairs()
    assert pair_reason == ""


def test_diagnosis_single_pair_skips_stage1_but_keeps_all_devices_for_llm() -> None:
    classifier = _classifier(_taxonomy())
    diagnosis_context = DiagnosisContext(
        year="2025",
        emp_num="E0001",
        row_count=1,
        teams=["DRAM공정 > DPC"],
        job_names=["Etch공정"],
        categories=[],
        evidence_rows=[],
    )

    def fail_stage1(self, context_json, candidate_pairs=None):
        raise AssertionError("stage1 should be skipped when diagnosis resolves one pair")

    def fixed_stage2(self, context_json, candidates):
        assert [row["Device"] for row in candidates] == ["DRAM", "NAND"]
        return FinalClassificationResult(
            major_job="공정",
            sub_job="Etch공정",
            device="NAND",
            unit_job="Chamber",
            detail_job_1="Clean",
            detail_job_2="",
            confidence=0.92,
            needs_review=False,
            reason="diagnosis_context의 Etch공정/DRAM공정 근거",
        )

    classifier._run_stage1 = MethodType(fail_stage1, classifier)
    classifier._run_stage2 = MethodType(fixed_stage2, classifier)

    result = classifier.classify_row(
        {
            "year": "2025",
            "team": "",
            "emp_num": "E0001",
            "name": "홍길동",
            "self_review": "",
        },
        diagnosis_context=diagnosis_context,
    )

    assert result["중직무"] == "공정"
    assert result["소직무"] == "Etch공정"
    assert result["Device"] == "NAND"
    assert result["needs_review"] is False
    assert "diagnosis 우선 적용" in result["diagnosis_priority_reason"]


def test_diagnosis_team_does_not_force_major_or_device_without_job_name() -> None:
    classifier = _classifier(_taxonomy())
    diagnosis_context = DiagnosisContext(
        year="2025",
        emp_num="E0003",
        row_count=1,
        teams=["DRAM공정 > DPC"],
        job_names=[],
        categories=[],
        evidence_rows=[],
    )

    priority = classifier._diagnosis_priority(diagnosis_context)
    pair_candidates, pair_reason = classifier._diagnosis_pair_candidates(priority)
    hints = classifier._build_classification_hints(
        "",
        diagnosis_context=diagnosis_context,
        knowledge_items=[],
    )

    assert priority.major_job == ""
    assert priority.sub_job == ""
    assert pair_candidates == classifier.taxonomy.pairs()
    assert pair_reason == ""
    assert any("taxonomy 중직무 '공정'" in hint for hint in hints)


def test_output_row_includes_diagnosis_priority_reason() -> None:
    output = build_output_row(
        {"year": "2025", "team": "", "emp_num": "E0001", "name": "홍길동"},
        {"diagnosis_priority_reason": "diagnosis 우선 적용"},
        include_self_review=False,
    )

    assert output["diagnosis_priority_reason"] == "diagnosis 우선 적용"
    assert "diagnosis_items" not in output
