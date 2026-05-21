from llm_categorizing.classifier import ClassificationConfig, OpenAICompatibleJobClassifier, extract_json_object
from llm_categorizing.config import LLMSettings
from llm_categorizing.diagnosis import DiagnosisContext
from llm_categorizing.knowledge import JobKnowledgeStore, KnowledgeDraft
from llm_categorizing.taxonomy import Taxonomy


def _classifier(taxonomy: Taxonomy, knowledge_store: JobKnowledgeStore | None = None) -> OpenAICompatibleJobClassifier:
    return OpenAICompatibleJobClassifier(
        settings=LLMSettings(
            base_url="http://localhost:1/v1",
            api_key="test",
            model="test",
        ),
        taxonomy=taxonomy,
        config=ClassificationConfig(),
        knowledge_store=knowledge_store,
    )


def test_classifier_does_not_build_hardcoded_rule_hints() -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "A",
                "소직무": "A1",
                "Device": "",
                "단위 직무": "공통",
                "세부 직무1": "",
                "세부 직무2": "",
            },
            {
                "중직무": "B",
                "소직무": "B1",
                "Device": "",
                "단위 직무": "공통",
                "세부 직무1": "",
                "세부 직무2": "",
            },
        ]
    )
    classifier = _classifier(taxonomy)
    diagnosis_context = DiagnosisContext(
        year="2025",
        emp_num="E0001",
        row_count=1,
        teams=["임의 팀"],
        job_names=["임의 직무"],
        categories=["임의 Category"],
        evidence_rows=[],
    )

    hints = classifier._build_classification_hints(
        "기존 코드에 있던 임의 키워드가 포함된 self_review",
        diagnosis_context=diagnosis_context,
    )

    assert hints == []
    assert not hasattr(classifier, "_apply_process_guardrail")
    assert not hasattr(classifier, "_apply_diagnosis_guardrail")


def test_classifier_json_extraction_ignores_thinking_block() -> None:
    parsed = extract_json_object(
        '<think>{"중직무": "무시"}</think>\n\n{"중직무": "공정", "소직무": "Etch"}'
    )

    assert parsed == {"중직무": "공정", "소직무": "Etch"}


def test_classifier_uses_only_retrieved_user_knowledge_as_hints(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "A",
                "소직무": "A1",
                "Device": "",
                "단위 직무": "Alpha",
                "세부 직무1": "",
                "세부 직무2": "",
            },
            {
                "중직무": "B",
                "소직무": "B1",
                "Device": "",
                "단위 직무": "Beta",
                "세부 직무1": "",
                "세부 직무2": "",
            },
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "AlphaTask 표현은 A 후보 검토에 참고",
        KnowledgeDraft(
            title="AlphaTask 참고 지식",
            aliases=["AlphaTask"],
            hint="AlphaTask 표현이 있으면 A 후보를 검토한다.",
            target_major_job="A",
            priority=80,
            confidence=0.8,
        ),
    )
    classifier = _classifier(taxonomy, store)

    knowledge_items = classifier._retrieve_knowledge("AlphaTask 수행", None)
    hints = classifier._build_classification_hints("AlphaTask 수행", knowledge_items=knowledge_items)

    assert [item.id for item in knowledge_items] == [entry.id]
    assert len(hints) == 1
    assert "AlphaTask 참고 지식" in hints[0]


def test_classifier_does_not_retrieve_knowledge_from_diagnosis_items(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "A",
                "소직무": "A1",
                "Device": "",
                "단위 직무": "Alpha",
                "세부 직무1": "",
                "세부 직무2": "",
            }
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.add(
        "LeakOnlyItem 표현은 A 후보 검토에 참고",
        KnowledgeDraft(
            title="diagnosis 항목 전용 지식",
            aliases=["LeakOnlyItem"],
            match_fields=["diagnosis_item"],
            hint="diagnosis 항목에 LeakOnlyItem이 있으면 A 후보를 검토한다.",
            target_major_job="A",
            priority=80,
            confidence=0.8,
        ),
    )
    classifier = _classifier(taxonomy, store)
    diagnosis_context = DiagnosisContext(
        year="2025",
        emp_num="E0001",
        row_count=1,
        teams=[],
        job_names=[],
        categories=[],
        evidence_rows=[
            {
                "diagnosis_team": "",
                "diagnosis_job_name": "",
                "category": "",
                "item": "LeakOnlyItem",
            }
        ],
    )

    knowledge_items = classifier._retrieve_knowledge("", diagnosis_context)
    hints = classifier._build_classification_hints(
        "",
        diagnosis_context=diagnosis_context,
        knowledge_items=knowledge_items,
    )

    assert knowledge_items == []
    assert hints == []
