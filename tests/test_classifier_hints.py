import json
from types import MethodType

from llm_categorizing.classifier import (
    ClassificationConfig,
    JsonlCache,
    OpenAICompatibleJobClassifier,
    extract_json_object,
)
from llm_categorizing.config import LLMSettings
from llm_categorizing.diagnosis import DiagnosisContext
from llm_categorizing.knowledge import JobKnowledgeStore, KnowledgeDraft
from llm_categorizing.models import FinalClassificationResult, Stage1Result
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


def test_near_hard_knowledge_limits_taxonomy_candidates(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "A",
                "소직무": "A1",
                "Device": "DRAM",
                "단위 직무": "Alpha",
                "세부 직무1": "",
                "세부 직무2": "",
            },
            {
                "중직무": "B",
                "소직무": "B1",
                "Device": "NAND",
                "단위 직무": "Beta",
                "세부 직무1": "",
                "세부 직무2": "",
            },
            {
                "중직무": "B",
                "소직무": "B1",
                "Device": "DRAM",
                "단위 직무": "Gamma",
                "세부 직무1": "",
                "세부 직무2": "",
            },
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "HardAlias는 NAND 후보를 강제한다.",
        KnowledgeDraft(
            title="HardAlias 준하드룰",
            aliases=["HardAlias"],
            hint="HardAlias가 있으면 NAND 후보를 우선한다.",
            target_device="NAND",
        ),
    )
    store.update_metadata(entry.id, enforcement_level="near_hard")
    classifier = _classifier(taxonomy, store)

    knowledge_items = classifier._retrieve_knowledge("self_review는 DRAM 업무지만 HardAlias 포함", None)
    priority = classifier._near_hard_knowledge_priority(knowledge_items)
    pairs, reason = classifier._knowledge_pair_candidates(priority)
    final_candidates = classifier._final_candidates_for_pair(pairs[0], priority)

    assert len(pairs) == 1
    assert pairs == [{"중직무": "B", "소직무": "B1"}]
    assert "준하드룰" in reason
    assert len(final_candidates) == 1
    assert final_candidates[0]["Device"] == "NAND"


def test_self_review_direct_pair_overrides_project_near_hard_major_hint(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "공정",
                "소직무": "ETCH공정",
                "Device": "NAND",
                "단위 직무": "Chamber",
                "세부 직무1": "Etch",
                "세부 직무2": "",
            },
            {
                "중직무": "공정",
                "소직무": "ETCH공정",
                "Device": "DRAM",
                "단위 직무": "Chamber",
                "세부 직무1": "Etch",
                "세부 직무2": "",
            },
            {
                "중직무": "소자",
                "소직무": "Device",
                "Device": "NAND",
                "단위 직무": "Cell",
                "세부 직무1": "M0C",
                "세부 직무2": "",
            },
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "Colosseum 프로젝트는 NAND Device 후보를 검토한다.",
        KnowledgeDraft(
            title="Colosseum NAND Device 준하드룰",
            aliases=["Colosseum"],
            hint="Colosseum 프로젝트는 Device=NAND로 본다.",
            target_major_job="소자",
            target_sub_job="Device",
            target_device="NAND",
        ),
    )
    store.update_metadata(entry.id, review_status="approved", enforcement_level="near_hard")
    classifier = _classifier(taxonomy, store)

    def fail_stage1(self, context_json, candidate_pairs=None):
        raise AssertionError("self_review direct pair should select the pair before stage1")

    def fixed_stage2(self, context_json, candidates):
        assert {row["중직무"] for row in candidates} == {"공정"}
        assert {row["소직무"] for row in candidates} == {"ETCH공정"}
        return FinalClassificationResult(
            major_job="공정",
            sub_job="ETCH공정",
            device="NAND",
            unit_job="Chamber",
            detail_job_1="Etch",
            detail_job_2="",
            confidence=0.92,
            needs_review=False,
            reason="M0C ETCH 공정/장비/Capa 개선 근거",
        )

    classifier._run_stage1 = MethodType(fail_stage1, classifier)
    classifier._run_stage2 = MethodType(fixed_stage2, classifier)

    result = classifier.classify_row(
        {
            "year": "2022",
            "team": "",
            "emp_num": "E0001",
            "name": "홍길동",
            "self_review": (
                "Colosseum Plug Etch UPH 공정 전환, GT2 28Chamber Capa 확보, "
                "Plug Etch APC Set up을 통한 공정 안정화"
            ),
        }
    )

    assert result["중직무"] == "공정"
    assert result["소직무"] == "ETCH공정"
    assert "self_review 직접 직무 단서 우선 적용" in result["diagnosis_priority_reason"]
    assert "준하드룰 지식은 선택된 중직무/소직무 내부" in result["knowledge_priority_reason"]


def test_stage2_reason_can_recover_available_pair_after_stage1_mispick() -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "공정",
                "소직무": "Etch공정",
                "Device": "DRAM",
                "단위 직무": "Chamber",
                "세부 직무1": "Etch",
                "세부 직무2": "",
            },
            {
                "중직무": "소자",
                "소직무": "Device",
                "Device": "NAND",
                "단위 직무": "소자개발",
                "세부 직무1": "Device",
                "세부 직무2": "",
            },
        ]
    )
    classifier = _classifier(taxonomy)

    def wrong_stage1(self, context_json, candidate_pairs=None):
        return Stage1Result(
            major_job="소자",
            sub_job="Device",
            confidence=0.8,
            needs_review=False,
            reason="잘못된 stage1 선택",
        )

    stage2_major_jobs: list[str] = []

    def recovering_stage2(self, context_json, candidates):
        stage2_major_jobs.append(candidates[0]["중직무"])
        if candidates[0]["중직무"] == "소자":
            return FinalClassificationResult(
                major_job="소자",
                sub_job="Device",
                device="NAND",
                unit_job="소자개발",
                detail_job_1="Device",
                detail_job_2="",
                confidence=0.35,
                needs_review=True,
                reason=(
                    "self_review의 핵심 업무는 M0C ETCH 공정 조건/장비/Capa/APC 개선임. "
                    "제공된 후보 목록에 공정 관련 계층이 없어 NAND Device 후보를 선택함."
                ),
            )
        return FinalClassificationResult(
            major_job="공정",
            sub_job="Etch공정",
            device="DRAM",
            unit_job="Chamber",
            detail_job_1="Etch",
            detail_job_2="",
            confidence=0.91,
            needs_review=False,
            reason="공정 > Etch공정 후보로 재검토해 Etch 공정 업무로 분류",
        )

    classifier._run_stage1 = MethodType(wrong_stage1, classifier)
    classifier._run_stage2 = MethodType(recovering_stage2, classifier)

    result = classifier.classify_row(
        {
            "year": "2022",
            "team": "",
            "emp_num": "E0001",
            "name": "홍길동",
            "self_review": "M0C ETCH Set Up, Chamber Capa, APC CD 개선",
        }
    )

    assert stage2_major_jobs == ["소자", "공정"]
    assert result["중직무"] == "공정"
    assert result["소직무"] == "Etch공정"
    assert result["needs_review"] is False
    assert "stage2 재시도" in result["guardrail_reason"]


def test_classifier_applies_global_hint_cap(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "공정",
                "소직무": "A1",
                "Device": "",
                "단위 직무": "Alpha",
                "세부 직무1": "",
                "세부 직무2": "",
            },
            {
                "중직무": "소자",
                "소직무": "B1",
                "Device": "",
                "단위 직무": "Beta",
                "세부 직무1": "",
                "세부 직무2": "",
            },
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    for index in range(4):
        store.add(
            f"Alias{index} 표현은 후보 검토에 참고",
            KnowledgeDraft(
                title=f"Alias{index} 참고 지식",
                aliases=[f"Alias{index}"],
                hint=f"Alias{index} 표현을 참고한다.",
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
        config=ClassificationConfig(max_knowledge_hints=2),
        knowledge_store=store,
    )
    diagnosis_context = DiagnosisContext(
        year="2025",
        emp_num="E0001",
        row_count=1,
        teams=["공정팀 소자팀"],
        job_names=[],
        categories=[],
        evidence_rows=[],
    )

    knowledge_items = classifier._retrieve_knowledge(
        "Alias0 Alias1 Alias2 Alias3 수행",
        diagnosis_context,
    )
    hints = classifier._build_classification_hints(
        "Alias0 Alias1 Alias2 Alias3 수행",
        diagnosis_context=diagnosis_context,
        knowledge_items=knowledge_items,
    )

    assert len(hints) == 2


def test_cache_key_includes_confidence_review_threshold() -> None:
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
    settings = LLMSettings(
        base_url="http://localhost:1/v1",
        api_key="test",
        model="test",
    )
    low_threshold = OpenAICompatibleJobClassifier(
        settings=settings,
        taxonomy=taxonomy,
        config=ClassificationConfig(confidence_review_threshold=0.6),
    )
    high_threshold = OpenAICompatibleJobClassifier(
        settings=settings,
        taxonomy=taxonomy,
        config=ClassificationConfig(confidence_review_threshold=0.8),
    )

    assert low_threshold._cache_key('{"self_review": "x"}') != high_threshold._cache_key(
        '{"self_review": "x"}'
    )


def test_cache_value_includes_source_identity(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "공정",
                "소직무": "Etch공정",
                "Device": "NAND",
                "단위 직무": "Chamber",
                "세부 직무1": "",
                "세부 직무2": "",
            }
        ]
    )
    cache_path = tmp_path / "classification_cache.jsonl"
    classifier = OpenAICompatibleJobClassifier(
        settings=LLMSettings(
            base_url="http://localhost:1/v1",
            api_key="test",
            model="test",
        ),
        taxonomy=taxonomy,
        config=ClassificationConfig(),
        cache=JsonlCache(cache_path),
    )

    def fixed_uncached(self, context_json, diagnosis_priority, knowledge_items):
        return {
            "중직무": "공정",
            "소직무": "Etch공정",
            "Device": "NAND",
            "단위 직무": "Chamber",
            "세부 직무1": "",
            "세부 직무2": "",
            "confidence": 0.9,
            "reason": "cache identity test",
            "needs_review": False,
            "ambiguity_reason": "",
            "guardrail_reason": "",
            "diagnosis_priority_reason": "",
            "knowledge_priority_reason": "",
            "error": "",
        }

    classifier._classify_uncached = MethodType(fixed_uncached, classifier)

    result = classifier.classify_row(
        {
            "year": "2026",
            "team": "",
            "emp_num": "E1234",
            "name": "홍길동",
            "self_review": "Etch 공정 개선",
        }
    )

    cache_item = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[0])
    assert result["year"] == "2026"
    assert result["emp_num"] == "E1234"
    assert result["name"] == "홍길동"
    assert cache_item["value"]["year"] == "2026"
    assert cache_item["value"]["emp_num"] == "E1234"
    assert cache_item["value"]["name"] == "홍길동"
    assert classifier._cache_key(
        '{"self_review": "x"}',
        cache_identity={"year": "2026", "emp_num": "E1234", "name": "홍길동"},
    ) != classifier._cache_key(
        '{"self_review": "x"}',
        cache_identity={"year": "2026", "emp_num": "E9999", "name": "김철수"},
    )


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
