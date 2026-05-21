import sqlite3

from llm_categorizing.classifier import ClassificationConfig, OpenAICompatibleJobClassifier
from llm_categorizing.config import LLMSettings
from llm_categorizing.diagnosis import DiagnosisContext
from llm_categorizing.knowledge import (
    JobKnowledgeStore,
    KnowledgeDraft,
    _extract_json_object,
    _knowledge_normalizer_extra_body,
    validate_draft_against_taxonomy,
)
from llm_categorizing.taxonomy import Taxonomy


def test_knowledge_store_retrieves_aliases_and_updates_version(tmp_path) -> None:
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "AlphaTask와 BetaReview가 함께 나오면 A 후보를 우선 검토",
        KnowledgeDraft(
            title="AlphaTask A 후보 단서",
            aliases=["AlphaTask", "BetaReview", "Alpha Task"],
            applies_when="AlphaTask와 BetaReview가 함께 등장",
            hint="A 후보를 우선 검토한다.",
            target_major_job="A",
            priority=80,
            confidence=0.8,
        ),
    )

    version_before = store.version_hash()
    retrieved = store.retrieve("Alpha Task 수행 및 BetaReview 결과 검토", limit=3)

    assert retrieved
    assert retrieved[0].id == entry.id
    assert retrieved[0].match_score > 0

    store.set_active(entry.id, False)

    assert store.retrieve("Alpha Task 수행 및 BetaReview 결과 검토", limit=3) == []
    assert store.version_hash() != version_before


def test_knowledge_normalizer_forces_qwen_thinking_off() -> None:
    settings = LLMSettings(
        base_url="http://localhost:1/v1",
        api_key="test",
        model="Qwen3.6-35B-A3B",
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
        provider_profile="qwen",
    )

    extra_body = _knowledge_normalizer_extra_body(settings)

    assert extra_body == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
    assert settings.extra_body == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}


def test_knowledge_json_extraction_ignores_thinking_block() -> None:
    parsed = _extract_json_object(
        '<think>{"draft": "ignored"}</think>\n\n{"title": "Alpha", "aliases": ["A"]}'
    )

    assert parsed == {"title": "Alpha", "aliases": ["A"]}


def test_classifier_includes_retrieved_knowledge_in_hints(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "소자",
                "소직무": "Process Integration",
                "Device": "",
                "단위 직무": "Alpha",
                "세부 직무1": "Device",
                "세부 직무2": "",
            },
            {
                "중직무": "공정",
                "소직무": "Module",
                "Device": "",
                "단위 직무": "Beta",
                "세부 직무1": "Process",
                "세부 직무2": "",
            },
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.add(
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
    classifier = OpenAICompatibleJobClassifier(
        settings=LLMSettings(
            base_url="http://localhost:1/v1",
            api_key="test",
            model="test",
        ),
        taxonomy=taxonomy,
        config=ClassificationConfig(),
        knowledge_store=store,
    )

    review = "AlphaTask 수행"
    knowledge_items = classifier._retrieve_knowledge(review, None)
    hints = classifier._build_classification_hints(review, knowledge_items=knowledge_items)

    assert any("사용자 지식[" in hint for hint in hints)
    assert any("AlphaTask 표현" in hint for hint in hints)


def test_classifier_retrieves_knowledge_from_diagnosis_team_alias(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
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
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.add(
        "Heraion은 NAND 제품/프로젝트를 의미한다.",
        KnowledgeDraft(
            title="Heraion 제품 alias",
            aliases=["Heraion"],
            hint="Heraion 프로젝트명은 NAND Device 후보를 검토한다.",
            target_device="NAND",
            priority=90,
            confidence=0.9,
        ),
    )
    classifier = OpenAICompatibleJobClassifier(
        settings=LLMSettings(
            base_url="http://localhost:1/v1",
            api_key="test",
            model="test",
        ),
        taxonomy=taxonomy,
        config=ClassificationConfig(),
        knowledge_store=store,
    )
    diagnosis_context = DiagnosisContext(
        year="2025",
        emp_num="E0003",
        row_count=1,
        teams=["Heraion PJT > Heraion TD"],
        job_names=[],
        categories=[],
        items=[],
        evidence_rows=[],
    )

    knowledge_items = classifier._retrieve_knowledge("", diagnosis_context)
    hints = classifier._build_classification_hints("", knowledge_items=knowledge_items)

    assert len(knowledge_items) == 1
    assert "Heraion 제품 alias" in hints[0]
    assert "NAND" in hints[0]


def test_classifier_retrieves_short_team_alias_for_major_job_hint(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "소자",
                "소직무": "Device",
                "Device": "",
                "단위 직무": "TD",
                "세부 직무1": "",
                "세부 직무2": "",
            }
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.add(
        "TD 조직명은 중직무 소자를 의미한다.",
        KnowledgeDraft(
            title="TD 조직 alias",
            aliases=["TD"],
            hint="diagnosis team에 TD가 있으면 중직무 소자 후보를 검토한다.",
            target_major_job="소자",
            priority=90,
            confidence=0.9,
        ),
    )
    classifier = OpenAICompatibleJobClassifier(
        settings=LLMSettings(
            base_url="http://localhost:1/v1",
            api_key="test",
            model="test",
        ),
        taxonomy=taxonomy,
        config=ClassificationConfig(),
        knowledge_store=store,
    )
    diagnosis_context = DiagnosisContext(
        year="2025",
        emp_num="E0004",
        row_count=1,
        teams=["Heraion PJT > Heraion TD"],
        job_names=[],
        categories=[],
        items=[],
        evidence_rows=[],
    )

    knowledge_items = classifier._retrieve_knowledge("", diagnosis_context)
    hints = classifier._build_classification_hints(
        "",
        diagnosis_context=diagnosis_context,
        knowledge_items=knowledge_items,
    )

    assert len(knowledge_items) == 1
    assert any("TD 조직 alias" in hint for hint in hints)
    assert any("alias 'TD'" in hint and "중직무 '소자'" in hint for hint in hints)


def test_validate_draft_against_taxonomy_canonicalizes_and_clears_invalid_path() -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "공정",
                "소직무": "Etch공정",
                "Device": "",
                "단위 직무": "CLN",
                "세부 직무1": "Etch",
                "세부 직무2": "",
            },
            {
                "중직무": "소자",
                "소직무": "Process Integration",
                "Device": "",
                "단위 직무": "Alpha",
                "세부 직무1": "Device",
                "세부 직무2": "",
            },
        ]
    )

    valid = validate_draft_against_taxonomy(
        KnowledgeDraft(
            target_major_job="공정",
            target_sub_job="Etch공정",
            target_unit_job="CLN",
        ),
        taxonomy,
    )
    invalid = validate_draft_against_taxonomy(
        KnowledgeDraft(
            target_major_job="공정",
            target_sub_job="Process Integration",
            target_unit_job="없는단위",
        ),
        taxonomy,
    )

    assert valid.target_major_job == "공정"
    assert valid.target_unit_job == "CLN"
    assert valid.validation_errors == []
    assert invalid.target_major_job == ""
    assert invalid.target_sub_job == ""
    assert invalid.target_unit_job == ""
    assert invalid.validation_errors


def test_knowledge_store_metadata_and_usage_log(tmp_path) -> None:
    db_path = tmp_path / "knowledge.sqlite3"
    store = JobKnowledgeStore(db_path)
    entry = store.add(
        "검증된 공정 지식",
        KnowledgeDraft(
            knowledge_type="correction",
            title="AlphaTask 수정 사례",
            aliases=["AlphaTask"],
            hint="AlphaTask는 A 후보를 우선 검토한다.",
            priority=70,
            confidence=0.8,
        ),
    )

    updated = store.update_metadata(
        entry.id,
        knowledge_type="verified_rule",
        review_status="approved",
    )
    retrieved = store.retrieve("AlphaTask 수행", limit=1)
    store.record_usage(
        classification_id="test-classification",
        knowledge_items=retrieved,
        result={
            "중직무": "공정",
            "소직무": "Etch공정",
            "단위 직무": "CLN",
            "needs_review": False,
        },
    )

    with sqlite3.connect(db_path) as connection:
        usage_count = connection.execute("SELECT COUNT(*) FROM knowledge_usage").fetchone()[0]

    assert updated is not None
    assert updated.knowledge_type == "verified_rule"
    assert updated.review_status == "approved"
    assert retrieved[0].knowledge_type == "verified_rule"
    assert usage_count == 1
