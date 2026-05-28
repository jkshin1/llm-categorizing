import json
import sqlite3

from llm_categorizing.classifier import ClassificationConfig, OpenAICompatibleJobClassifier
from llm_categorizing.config import LLMSettings
from llm_categorizing.diagnosis import DiagnosisContext
from llm_categorizing.knowledge import (
    JobKnowledgeStore,
    KnowledgeSearchContext,
    KnowledgeDraft,
    KNOWLEDGE_NORMALIZATION_SYSTEM_PROMPT,
    _extract_json_object,
    _knowledge_normalizer_extra_body,
    fallback_knowledge_draft,
    knowledge_normalization_user_prompt,
    preserve_explicit_target_scope,
    preserve_raw_knowledge_terms,
    raw_abbreviation_terms,
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


def test_knowledge_store_populates_alias_index_and_revisions(tmp_path) -> None:
    db_path = tmp_path / "knowledge.sqlite3"
    store = JobKnowledgeStore(db_path)
    entry = store.add(
        "AlphaTask와 BetaReview가 함께 나오면 A 후보를 우선 검토",
        KnowledgeDraft(
            aliases=["AlphaTask", "BetaReview"],
            match_fields=["self_review", "diagnosis_team"],
            target_major_job="A",
        ),
    )
    store.update_metadata(entry.id, enforcement_level="near_hard")

    with store._connect() as connection:
        connection.row_factory = sqlite3.Row
        aliases = connection.execute(
            """
            SELECT alias_key, match_field
            FROM knowledge_aliases
            WHERE entry_id = ?
            ORDER BY alias_key, match_field
            """,
            (entry.id,),
        ).fetchall()
        revisions = connection.execute(
            """
            SELECT action, snapshot_json
            FROM knowledge_revisions
            WHERE entry_id = ?
            ORDER BY id
            """,
            (entry.id,),
        ).fetchall()
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert ("alphatask", "self_review") in {
        (row["alias_key"], row["match_field"]) for row in aliases
    }
    assert ("betareview", "diagnosis_team") in {
        (row["alias_key"], row["match_field"]) for row in aliases
    }
    assert [row["action"] for row in revisions] == ["create", "metadata_update"]
    assert json.loads(revisions[-1]["snapshot_json"])["enforcement_level"] == "near_hard"
    assert busy_timeout >= 30000
    assert foreign_keys == 1
    assert journal_mode == "wal"


def test_knowledge_store_backfills_alias_index_for_existing_entries(tmp_path) -> None:
    db_path = tmp_path / "knowledge.sqlite3"
    store = JobKnowledgeStore(db_path)
    entry = store.add(
        "LegacyTask 지식",
        KnowledgeDraft(aliases=["LegacyTask"], match_fields=["diagnosis_team"], target_major_job="A"),
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute("DELETE FROM knowledge_aliases")
        connection.execute("DELETE FROM knowledge_revisions")

    migrated = JobKnowledgeStore(db_path)
    retrieved = migrated.retrieve_for_context(
        KnowledgeSearchContext(diagnosis_teams=("LegacyTask team",)),
        limit=3,
    )

    with sqlite3.connect(db_path) as connection:
        alias_count = connection.execute(
            "SELECT COUNT(*) FROM knowledge_aliases WHERE entry_id = ?",
            (entry.id,),
        ).fetchone()[0]
        revision_count = connection.execute(
            "SELECT COUNT(*) FROM knowledge_revisions WHERE entry_id = ? AND action = 'initial_import'",
            (entry.id,),
        ).fetchone()[0]

    assert alias_count == 1
    assert revision_count == 1
    assert [item.id for item in retrieved] == [entry.id]


def test_knowledge_store_does_not_retrieve_from_target_value_only(tmp_path) -> None:
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.add(
        "Heraion은 NAND 제품 프로젝트를 의미한다.",
        KnowledgeDraft(
            title="Heraion 제품 alias",
            aliases=["Heraion"],
            hint="Heraion 프로젝트명은 NAND Device 후보를 검토한다.",
            target_device="NAND",
            priority=80,
            confidence=0.8,
        ),
    )

    retrieved = store.retrieve("NAND 수율 분석을 수행했지만 프로젝트 alias 언급은 없음", limit=3)

    assert retrieved == []


def test_knowledge_store_uses_fts_as_soft_fallback(tmp_path) -> None:
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "Colosseum 프로젝트는 NAND failure signature 분석 맥락에서 참고한다.",
        KnowledgeDraft(
            aliases=["UnusedAlias"],
            match_fields=["diagnosis_team"],
            hint="NAND failure signature 분석 맥락이면 참고한다.",
            target_device="NAND",
            priority=40,
            confidence=0.6,
        ),
    )

    retrieved = store.retrieve("NAND failure signature 분석 수행", limit=3)

    assert retrieved
    assert retrieved[0].id == entry.id
    assert retrieved[0].match_score > 0


def test_knowledge_store_does_not_use_fts_fallback_for_near_hard_rules(tmp_path) -> None:
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "Colosseum 프로젝트는 NAND failure signature 분석 맥락에서 참고한다.",
        KnowledgeDraft(
            aliases=["UnusedAlias"],
            hint="NAND failure signature 분석 맥락이면 참고한다.",
            target_device="NAND",
        ),
    )
    store.update_metadata(entry.id, enforcement_level="near_hard")

    retrieved = store.retrieve("NAND failure signature 분석 수행", limit=3)

    assert retrieved == []


def test_knowledge_store_infers_match_fields_and_scores_structured_context(tmp_path) -> None:
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    team_entry = store.add(
        "Heraion team은 NAND 제품 프로젝트를 의미한다.",
        KnowledgeDraft(
            title="Heraion team alias",
            aliases=["Heraion"],
            hint="team에 Heraion이 있으면 NAND Device 후보를 검토한다.",
            target_device="NAND",
            priority=70,
            confidence=0.7,
        ),
    )
    review_entry = store.add(
        "Heraion 업무 표현은 리뷰 텍스트에서만 참고한다.",
        KnowledgeDraft(
            title="Heraion review alias",
            aliases=["Heraion"],
            match_fields=["self_review"],
            hint="self_review에서 Heraion이 있으면 일반 참고로만 본다.",
            priority=70,
            confidence=0.7,
        ),
    )

    retrieved = store.retrieve_for_context(
        KnowledgeSearchContext(diagnosis_teams=("Heraion PJT > Heraion TD",)),
        limit=2,
    )

    assert "diagnosis_team" in team_entry.match_fields
    assert retrieved[0].id == team_entry.id
    assert {item.id for item in retrieved} == {team_entry.id, review_entry.id}


def test_knowledge_store_merges_duplicate_raw_text(tmp_path) -> None:
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    first = store.add(
        "TD 조직명은 중직무 소자를 의미한다.",
        KnowledgeDraft(aliases=["TD"], target_major_job="소자", priority=50),
    )
    second = store.add(
        "TD 조직명은 중직무 소자를 의미한다.",
        KnowledgeDraft(aliases=["TD", "Technology Development"], target_major_job="소자", priority=80),
        source="txt_import",
    )

    entries = store.list_recent(limit=10)

    assert first.id == second.id
    assert len(entries) == 1
    assert "Technology Development" in entries[0].aliases
    assert entries[0].priority == 80


def test_knowledge_store_marks_conflicting_targets(tmp_path) -> None:
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.add(
        "Heraion은 NAND 제품을 의미한다.",
        KnowledgeDraft(aliases=["Heraion"], target_device="NAND"),
    )

    conflicting = store.add(
        "Heraion은 DRAM 제품을 의미한다.",
        KnowledgeDraft(aliases=["Heraion"], target_device="DRAM"),
    )

    assert conflicting.conflicts
    assert conflicting.review_status == "draft"
    assert any("potential conflict" in error for error in conflicting.validation_errors)


def test_knowledge_store_can_clear_false_positive_conflicts(tmp_path) -> None:
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.add(
        "Heraion은 NAND 제품을 의미한다.",
        KnowledgeDraft(aliases=["Heraion"], target_device="NAND"),
    )
    conflicting = store.add(
        "Heraion은 DRAM 제품도 맥락에 따라 의미할 수 있다.",
        KnowledgeDraft(aliases=["Heraion"], target_device="DRAM"),
    )

    cleared = store.update_metadata(conflicting.id, clear_conflicts=True)
    approved = store.update_metadata(
        conflicting.id,
        knowledge_type="verified_rule",
        review_status="approved",
    )

    assert cleared is not None
    assert cleared.conflicts == ()
    assert cleared.validation_errors == ()
    assert approved is not None
    assert approved.conflicts == ()
    assert approved.validation_errors == ()
    assert approved.review_status == "approved"


def test_knowledge_store_can_retrieve_only_approved_entries(tmp_path) -> None:
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    draft = store.add(
        "AlphaTask 초안 지식",
        KnowledgeDraft(aliases=["AlphaTask"], hint="초안 지식"),
    )
    approved = store.add(
        "AlphaTask 검증 지식",
        KnowledgeDraft(
            knowledge_type="verified_rule",
            aliases=["AlphaTask"],
            hint="검증 지식",
        ),
    )

    usable = store.retrieve("AlphaTask 수행", limit=5)
    approved_only = store.retrieve("AlphaTask 수행", limit=5, review_scope="approved")

    assert {item.id for item in usable} == {draft.id, approved.id}
    assert [item.id for item in approved_only] == [approved.id]


def test_knowledge_store_exports_and_imports_approved_ndjson(tmp_path) -> None:
    source_store = JobKnowledgeStore(tmp_path / "source.sqlite3")
    draft = source_store.add(
        "AlphaTask 초안 지식",
        KnowledgeDraft(aliases=["AlphaTask"], hint="초안 지식"),
    )
    approved = source_store.add(
        "BetaTask 검증 지식",
        KnowledgeDraft(
            knowledge_type="verified_rule",
            aliases=["BetaTask"],
            match_fields=["diagnosis_team"],
            hint="검증 지식",
            target_major_job="B",
            priority=80,
            confidence=0.8,
        ),
    )
    export_path = tmp_path / "approved.ndjson"

    count = source_store.export_ndjson(export_path)
    imported = JobKnowledgeStore(tmp_path / "imported.sqlite3").import_ndjson(export_path)

    assert count == 1
    assert draft.raw_text not in export_path.read_text(encoding="utf-8")
    assert len(imported) == 1
    assert imported[0].raw_text == approved.raw_text
    assert imported[0].review_status == "approved"
    assert imported[0].enforcement_level == "strong"
    assert imported[0].target_major_job == "B"


def test_near_hard_enforcement_is_strongest_hint(tmp_path) -> None:
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    normal = store.add(
        "AlphaTask는 일반 참고 지식이다.",
        KnowledgeDraft(
            title="AlphaTask 일반",
            aliases=["AlphaTask"],
            hint="AlphaTask는 일반 참고로만 본다.",
            target_major_job="A",
            priority=90,
            confidence=0.9,
        ),
    )
    near_hard = store.add(
        "AlphaTask는 검증된 준하드룰 지식이다.",
        KnowledgeDraft(
            title="AlphaTask 준하드",
            aliases=["AlphaTask"],
            hint="AlphaTask가 있으면 B 후보를 사실상 우선한다.",
            target_major_job="B",
            priority=50,
            confidence=0.5,
        ),
    )

    updated = store.update_metadata(near_hard.id, enforcement_level="near_hard")
    retrieved = store.retrieve("AlphaTask 수행", limit=2)

    assert updated is not None
    assert updated.enforcement_level == "near_hard"
    assert updated.knowledge_type == "verified_rule"
    assert updated.review_status == "approved"
    assert "준하드룰" in updated.prompt_hint()
    assert [item.id for item in retrieved] == [near_hard.id, normal.id]


def test_strong_enforcement_preserves_approved_status(tmp_path) -> None:
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "BetaTask는 검증 지식이다.",
        KnowledgeDraft(aliases=["BetaTask"], hint="BetaTask는 검증 지식"),
    )

    strong = store.update_metadata(entry.id, enforcement_level="strong")
    soft = store.update_metadata(entry.id, enforcement_level="soft")

    assert strong is not None
    assert strong.enforcement_level == "strong"
    assert strong.knowledge_type == "verified_rule"
    assert strong.review_status == "approved"
    assert soft is not None
    assert soft.enforcement_level == "soft"
    assert soft.knowledge_type == "soft_hint"
    assert soft.review_status == "draft"


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


def test_knowledge_prompt_forbids_arbitrary_abbreviation_expansion() -> None:
    prompt = knowledge_normalization_user_prompt("TD는 중직무 소자 후보")

    assert "약어" in KNOWLEDGE_NORMALIZATION_SYSTEM_PROMPT
    assert "입력에 명시되지 않은 약어 풀어쓰기" in KNOWLEDGE_NORMALIZATION_SYSTEM_PROMPT
    assert "aliases에는 사용자 입력 문자열에 실제 등장한 표기만 넣는다" in prompt
    assert "약어의 풀네임" in prompt
    assert "Device 항목/컬럼만 지정하면 target_device만" in KNOWLEDGE_NORMALIZATION_SYSTEM_PROMPT
    assert 'target_sub_job="Device"' in prompt


def test_preserve_raw_knowledge_terms_removes_llm_abbreviation_expansion() -> None:
    draft = KnowledgeDraft(
        title="TD Technology Development 지식",
        aliases=["TD", "Technology Development"],
        applies_when="Technology Development 조직명과 매칭될 때",
        hint="TD는 Technology Development 의미로 보고 소자 후보를 검토한다.",
        target_major_job="소자",
        target_sub_job="Technology Development",
        priority=80,
        confidence=0.8,
    )

    preserved = preserve_raw_knowledge_terms("TD 조직명은 중직무 소자 후보를 우선 검토한다.", draft)

    assert "TD" in preserved.aliases
    assert "Technology Development" not in preserved.aliases
    assert preserved.title == "TD 조직명은 중직무 소자 후보를 우선 검토한다."
    assert preserved.hint == "TD 조직명은 중직무 소자 후보를 우선 검토한다."
    assert preserved.applies_when == "사용자 입력 지식과 현재 입력이 명확히 관련될 때"
    assert preserved.target_major_job == "소자"
    assert preserved.target_sub_job == ""
    assert any("removed LLM-generated aliases" in error for error in preserved.validation_errors)
    assert any("cleared target_sub_job" in error for error in preserved.validation_errors)


def test_preserve_explicit_device_only_target_scope_clears_inferred_path() -> None:
    draft = KnowledgeDraft(
        title="Colosseum NAND 지식",
        aliases=["Colosseum"],
        match_fields=["diagnosis_team"],
        applies_when="Colosseum 프로젝트",
        hint="Colosseum이면 NAND Device로 분류",
        target_major_job="소자",
        target_sub_job="Device",
        target_device="NAND",
        target_unit_job="소자개발",
        target_detail_job_1="Device Characterization",
        confidence=0.8,
    )

    preserved = preserve_explicit_target_scope(
        "팀 명칭에 'Colosseum'이 포함된 경우 Device항목은 무조건 'NAND'로 분류해야 합니다.",
        draft,
    )

    assert preserved.target_major_job == ""
    assert preserved.target_sub_job == ""
    assert preserved.target_device == "NAND"
    assert preserved.target_unit_job == ""
    assert preserved.target_detail_job_1 == ""
    assert any("input only constrained Device target" in error for error in preserved.validation_errors)


def test_preserve_explicit_device_only_target_scope_keeps_explicit_major_scope() -> None:
    draft = KnowledgeDraft(
        target_major_job="소자",
        target_sub_job="Device",
        target_device="NAND",
    )

    preserved = preserve_explicit_target_scope(
        "중직무는 소자, 소직무는 Device, Device 항목은 NAND로 분류한다.",
        draft,
    )

    assert preserved.target_major_job == "소자"
    assert preserved.target_sub_job == "Device"
    assert preserved.target_device == "NAND"


def test_fallback_knowledge_draft_keeps_short_abbreviation_alias() -> None:
    draft = fallback_knowledge_draft("TD 조직명은 중직무 소자 후보를 우선 검토한다.")

    assert "TD" in draft.aliases
    assert raw_abbreviation_terms("M0C ETCH와 TD") == ["M0C ETCH", "M0C", "ETCH", "TD"]
    assert raw_abbreviation_terms("2025년 기준") == []


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
