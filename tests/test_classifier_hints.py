from llm_categorizing.classifier import ClassificationConfig, OpenAICompatibleJobClassifier
from llm_categorizing.config import LLMSettings
from llm_categorizing.diagnosis import DiagnosisContext
from llm_categorizing.taxonomy import Taxonomy


def test_classifier_builds_process_and_ambiguity_hints_for_mlm_review() -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "소자",
                "소직무": "Process Integration",
                "Device": "DRAM",
                "단위 직무": "MLM",
                "세부 직무1": "Device",
                "세부 직무2": "",
            },
            {
                "중직무": "공정",
                "소직무": "Module",
                "Device": "DRAM",
                "단위 직무": "MLM",
                "세부 직무1": "Process",
                "세부 직무2": "",
            },
        ]
    )
    classifier = OpenAICompatibleJobClassifier(
        settings=LLMSettings(
            base_url="http://localhost:1/v1",
            api_key="test",
            model="test",
        ),
        taxonomy=taxonomy,
        config=ClassificationConfig(),
    )

    hints = classifier._build_classification_hints(
        "MLM Module 기술적 지원 및 Process Qual, Base Line 변경, DRAM 관련 검토"
    )

    assert any("공정 수행 근거" in hint for hint in hints)
    assert any("여러 중직무" in hint for hint in hints)
    assert any("DRAM 등 제품명" in hint for hint in hints)

    corrected, reason = classifier._apply_process_guardrail(
        taxonomy.rows[0],
        "MLM Module 기술적 지원 및 Process Qual, Base Line 변경, DRAM 관련 검토",
    )

    assert corrected["중직무"] == "공정"
    assert reason.startswith("process_guardrail:")


def test_classifier_uses_diagnosis_guardrail_for_process_job() -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "소자",
                "소직무": "Process Integration",
                "Device": "DRAM",
                "단위 직무": "MLM",
                "세부 직무1": "Device",
                "세부 직무2": "",
            },
            {
                "중직무": "공정",
                "소직무": "Etch공정",
                "Device": "DRAM",
                "단위 직무": "CLN",
                "세부 직무1": "Etch",
                "세부 직무2": "",
            },
        ]
    )
    classifier = OpenAICompatibleJobClassifier(
        settings=LLMSettings(
            base_url="http://localhost:1/v1",
            api_key="test",
            model="test",
        ),
        taxonomy=taxonomy,
        config=ClassificationConfig(),
    )
    diagnosis_context = DiagnosisContext(
        year="2025",
        emp_num="E0001",
        row_count=2,
        teams=["DRAM공정 > DPC"],
        job_names=["Etch공정"],
        categories=["CLN"],
        items=["Chamber clean"],
        evidence_rows=[],
    )

    corrected, reason = classifier._apply_diagnosis_guardrail(taxonomy.rows[0], diagnosis_context)

    assert corrected == taxonomy.rows[1]
    assert reason.startswith("diagnosis_guardrail:")
