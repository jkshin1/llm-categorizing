from types import MethodType

from llm_categorizing.classifier import ClassificationConfig, OpenAICompatibleJobClassifier
from llm_categorizing.cli import build_output_row
from llm_categorizing.config import LLMSettings
from llm_categorizing.diagnosis import DiagnosisContext
from llm_categorizing.knowledge import JobKnowledgeStore, KnowledgeDraft
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


def test_diagnosis_job_name_uses_major_signal_to_break_duplicate_sub_job() -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "공정",
                "소직무": "Etch공정",
                "Device": "DRAM",
                "단위 직무": "Chamber",
                "세부 직무1": "",
                "세부 직무2": "",
            },
            {
                "중직무": "소자",
                "소직무": "Etch공정",
                "Device": "DRAM",
                "단위 직무": "Device Etch",
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
        teams=[],
        job_names=["Etch공정"],
        categories=[],
        evidence_rows=[],
    )

    priority = classifier._diagnosis_priority(diagnosis_context)
    pair_candidates, pair_reason = classifier._diagnosis_pair_candidates(priority)

    assert priority.major_job == "공정"
    assert priority.sub_job == "Etch공정"
    assert pair_candidates == [{"중직무": "공정", "소직무": "Etch공정"}]
    assert "중직무 직접 단서" in priority.reason
    assert "diagnosis 우선 적용" in pair_reason


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


def test_diagnosis_job_name_takes_precedence_over_conflicting_near_hard_knowledge(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
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
                "중직무": "소자",
                "소직무": "TD",
                "Device": "NAND",
                "단위 직무": "Cell",
                "세부 직무1": "Integration",
                "세부 직무2": "",
            },
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "TD는 소자 후보를 우선한다.",
        KnowledgeDraft(
            title="TD 소자 준하드룰",
            aliases=["TD"],
            hint="diagnosis team에 TD가 있으면 소자 후보를 우선한다.",
            target_major_job="소자",
            target_sub_job="TD",
            match_fields=["diagnosis_team"],
        ),
    )
    store.update_metadata(entry.id, enforcement_level="near_hard")
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
        emp_num="E0001",
        row_count=1,
        teams=["Heraion TD"],
        job_names=["Etch공정"],
        categories=[],
        evidence_rows=[],
    )

    def fail_stage1(self, context_json, candidate_pairs=None):
        raise AssertionError("diagnosis job name should select the pair before stage1")

    def fixed_stage2(self, context_json, candidates):
        assert candidates == [taxonomy.rows[0]]
        return FinalClassificationResult(
            major_job="공정",
            sub_job="Etch공정",
            device="DRAM",
            unit_job="Chamber",
            detail_job_1="Clean",
            detail_job_2="",
            confidence=0.93,
            needs_review=False,
            reason="진단 직무명 Etch공정이 우선",
        )

    classifier._run_stage1 = MethodType(fail_stage1, classifier)
    classifier._run_stage2 = MethodType(fixed_stage2, classifier)

    result = classifier.classify_row(
        {
            "year": "2025",
            "team": "",
            "emp_num": "E0001",
            "name": "홍길동",
            "self_review": "TD 관련 업무",
        },
        diagnosis_context=diagnosis_context,
    )

    assert result["중직무"] == "공정"
    assert result["소직무"] == "Etch공정"
    assert "diagnosis 우선 적용" in result["diagnosis_priority_reason"]
    assert "diagnosis 직무명 우선 적용" in result["knowledge_priority_reason"]


def test_near_hard_diagnosis_job_name_mapping_overrides_legacy_diagnosis_pair(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "공정",
                "소직무": "Photo공정",
                "Device": "",
                "단위 직무": "Photo",
                "세부 직무1": "",
                "세부 직무2": "",
            },
            {
                "중직무": "DIC",
                "소직무": "OPC",
                "Device": "",
                "단위 직무": "OPC",
                "세부 직무1": "",
                "세부 직무2": "",
            },
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "2021~2024년 진단 시 직무명 Photo공정은 OPC 단서가 있으면 현재 DIC > OPC로 보정한다.",
        KnowledgeDraft(
            title="Photo공정 과거 OPC 매핑",
            aliases=["Photo", "OPC", "CROPC", "MASK"],
            match_fields=["diagnosis_job_name", "self_review"],
            hint="2021~2024년 Photo공정 진단명과 OPC 업무 단서가 함께 있으면 DIC > OPC 후보를 우선한다.",
            target_major_job="DIC",
            target_sub_job="OPC",
        ),
    )
    store.update_metadata(entry.id, enforcement_level="near_hard")
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
        year="2022",
        emp_num="E0002",
        row_count=1,
        teams=[],
        job_names=["Photo공정"],
        categories=[],
        evidence_rows=[],
    )

    def fail_stage1(self, context_json, candidate_pairs=None):
        raise AssertionError("near-hard legacy mapping should select the pair before stage1")

    def fixed_stage2(self, context_json, candidates):
        assert candidates == [taxonomy.rows[1]]
        return FinalClassificationResult(
            major_job="DIC",
            sub_job="OPC",
            device="",
            unit_job="OPC",
            detail_job_1="",
            detail_job_2="",
            confidence=0.94,
            needs_review=False,
            reason="Photo공정 과거 진단명과 OPC/CROPC/MASK 업무 단서가 함께 있어 DIC > OPC로 보정",
        )

    classifier._run_stage1 = MethodType(fail_stage1, classifier)
    classifier._run_stage2 = MethodType(fixed_stage2, classifier)

    result = classifier.classify_row(
        {
            "year": "2022",
            "team": "",
            "emp_num": "E0002",
            "name": "홍길동",
            "self_review": "OPC 조건 확보, CROPC 적용, MASK 확보 등 OPC 인프라 업무 수행",
        },
        diagnosis_context=diagnosis_context,
    )

    assert result["중직무"] == "DIC"
    assert result["소직무"] == "OPC"
    assert "준하드룰 지식이 과거 diagnosis 직무명/team 보정" in result["diagnosis_priority_reason"]
    assert "diagnosis 직무명/team 보정으로 stage1 후보 제한" in result["knowledge_priority_reason"]


def test_near_hard_diagnosis_team_mapping_overrides_legacy_diagnosis_pair(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "공정",
                "소직무": "Photo공정",
                "Device": "DRAM",
                "단위 직무": "IPT",
                "세부 직무1": "",
                "세부 직무2": "",
            },
            {
                "중직무": "DIC",
                "소직무": "OPC",
                "Device": "DRAM",
                "단위 직무": "OPC Technology",
                "세부 직무1": "",
                "세부 직무2": "",
            },
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "2021~2024년 diagnosis team에 DRAM OPC 또는 OPC가 포함되면 현재 DIC > OPC 업무로 보정한다.",
        KnowledgeDraft(
            title="DRAM OPC team 과거 OPC 매핑",
            aliases=["OPC", "DRAM OPC"],
            match_fields=["diagnosis_team"],
            hint="diagnosis team이 DRAM OPC이면 DIC > OPC 후보를 우선한다.",
            target_major_job="DIC",
            target_sub_job="OPC",
        ),
    )
    store.update_metadata(entry.id, enforcement_level="near_hard")
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
        year="2022",
        emp_num="E0006",
        row_count=1,
        teams=["DRAM OPC"],
        job_names=["Photo공정"],
        categories=[],
        evidence_rows=[],
    )

    def fail_stage1(self, context_json, candidate_pairs=None):
        raise AssertionError("near-hard team mapping should select the pair before stage1")

    def fixed_stage2(self, context_json, candidates):
        assert candidates == [taxonomy.rows[1]]
        return FinalClassificationResult(
            major_job="DIC",
            sub_job="OPC",
            device="DRAM",
            unit_job="OPC Technology",
            detail_job_1="",
            detail_job_2="",
            confidence=0.94,
            needs_review=False,
            reason="diagnosis team DRAM OPC와 OPC 업무 단서가 있어 DIC > OPC로 보정",
        )

    classifier._run_stage1 = MethodType(fail_stage1, classifier)
    classifier._run_stage2 = MethodType(fixed_stage2, classifier)

    result = classifier.classify_row(
        {
            "year": "2022",
            "team": "",
            "emp_num": "E0006",
            "name": "홍길동",
            "self_review": "IPT, ISO, BLC, MT0, NFC 구간 OPC 업무 수행",
        },
        diagnosis_context=diagnosis_context,
    )

    assert result["중직무"] == "DIC"
    assert result["소직무"] == "OPC"
    assert "준하드룰 지식이 과거 diagnosis 직무명/team 보정" in result["diagnosis_priority_reason"]
    assert "diagnosis 직무명/team 보정으로 stage1 후보 제한" in result["knowledge_priority_reason"]


def test_near_hard_legacy_mapping_requires_matching_diagnosis_job_name(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "공정",
                "소직무": "Etch공정",
                "Device": "",
                "단위 직무": "Etch",
                "세부 직무1": "",
                "세부 직무2": "",
            },
            {
                "중직무": "DIC",
                "소직무": "OPC",
                "Device": "",
                "단위 직무": "OPC",
                "세부 직무1": "",
                "세부 직무2": "",
            },
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "2021~2024년 진단 시 직무명 Photo공정은 OPC 단서가 있으면 현재 DIC > OPC로 보정한다.",
        KnowledgeDraft(
            title="Photo공정 과거 OPC 매핑",
            aliases=["Photo공정", "OPC", "CROPC", "MASK"],
            match_fields=["diagnosis_job_name", "self_review"],
            hint="2021~2024년 Photo공정 진단명과 OPC 업무 단서가 함께 있으면 DIC > OPC 후보를 우선한다.",
            target_major_job="DIC",
            target_sub_job="OPC",
        ),
    )
    store.update_metadata(entry.id, enforcement_level="near_hard")
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
        year="2022",
        emp_num="E0003",
        row_count=1,
        teams=[],
        job_names=["Etch공정"],
        categories=[],
        evidence_rows=[],
    )

    def fail_stage1(self, context_json, candidate_pairs=None):
        raise AssertionError("diagnosis job name should select the pair before stage1")

    def fixed_stage2(self, context_json, candidates):
        assert candidates == [taxonomy.rows[0]]
        return FinalClassificationResult(
            major_job="공정",
            sub_job="Etch공정",
            device="",
            unit_job="Etch",
            detail_job_1="",
            detail_job_2="",
            confidence=0.9,
            needs_review=False,
            reason="진단 직무명 Etch공정이 우선",
        )

    classifier._run_stage1 = MethodType(fail_stage1, classifier)
    classifier._run_stage2 = MethodType(fixed_stage2, classifier)

    result = classifier.classify_row(
        {
            "year": "2022",
            "team": "",
            "emp_num": "E0003",
            "name": "홍길동",
            "self_review": "OPC 조건 확보, CROPC 적용, MASK 확보 업무도 일부 수행",
        },
        diagnosis_context=diagnosis_context,
    )

    assert result["중직무"] == "공정"
    assert result["소직무"] == "Etch공정"
    assert "diagnosis 직무명 우선 적용" in result["knowledge_priority_reason"]


def test_near_hard_legacy_mapping_does_not_use_partial_diagnosis_job_name_alias(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "DIC",
                "소직무": "DTCO",
                "Device": "NAND",
                "단위 직무": "SPICE Modeling",
                "세부 직무1": "",
                "세부 직무2": "",
            },
            {
                "중직무": "DIC",
                "소직무": "OPC",
                "Device": "NAND",
                "단위 직무": "OPC Technology",
                "세부 직무1": "",
                "세부 직무2": "",
            },
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "2021~2024년 diagnosis 직무명이 Photo공정이고 self_review에 OPC/SOM/Recipe/Model이 핵심 업무로 나타나면 OPC 업무로 본다.",
        KnowledgeDraft(
            title="Photo공정 과거 OPC 매핑",
            aliases=["Photo공정", "OPC", "SOM", "Recipe", "Model"],
            match_fields=["diagnosis_job_name", "self_review"],
            hint="Photo공정 진단명과 OPC/SOM/Recipe/Model 업무 단서가 함께 있으면 DIC > OPC 후보를 우선한다.",
            target_major_job="DIC",
            target_sub_job="OPC",
        ),
    )
    store.update_metadata(entry.id, enforcement_level="near_hard")
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
        year="2022",
        emp_num="E0005",
        row_count=1,
        teams=["Device Tech Solution > DTCO"],
        job_names=["SPICE Modeling"],
        categories=["미기원-Modeling"],
        evidence_rows=[],
    )

    def fail_stage1(self, context_json, candidate_pairs=None):
        raise AssertionError("diagnosis job name should select the DTCO pair before stage1")

    def fixed_stage2(self, context_json, candidates):
        assert candidates == [taxonomy.rows[0]]
        return FinalClassificationResult(
            major_job="DIC",
            sub_job="DTCO",
            device="NAND",
            unit_job="SPICE Modeling",
            detail_job_1="",
            detail_job_2="",
            confidence=0.91,
            needs_review=False,
            reason="진단 직무명 SPICE Modeling과 DTCO team 근거",
        )

    classifier._run_stage1 = MethodType(fail_stage1, classifier)
    classifier._run_stage2 = MethodType(fixed_stage2, classifier)

    result = classifier.classify_row(
        {
            "year": "2022",
            "team": "",
            "emp_num": "E0005",
            "name": "홍길동",
            "self_review": "DTCO/회로 최적화/SPICE Modeling 수행 및 RB NAND 검토",
        },
        diagnosis_context=diagnosis_context,
    )

    assert result["중직무"] == "DIC"
    assert result["소직무"] == "DTCO"
    assert "diagnosis 우선 적용" in result["diagnosis_priority_reason"]
    assert "준하드룰 지식이 과거 diagnosis 직무명/team 보정" not in result["diagnosis_priority_reason"]


def test_near_hard_legacy_mapping_does_not_override_outside_year_range(tmp_path) -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "공정",
                "소직무": "Photo공정",
                "Device": "",
                "단위 직무": "Photo",
                "세부 직무1": "",
                "세부 직무2": "",
            },
            {
                "중직무": "DIC",
                "소직무": "OPC",
                "Device": "",
                "단위 직무": "OPC",
                "세부 직무1": "",
                "세부 직무2": "",
            },
        ]
    )
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")
    entry = store.add(
        "2021~2024년 진단 시 직무명 Photo공정은 OPC 단서가 있으면 현재 DIC > OPC로 보정한다.",
        KnowledgeDraft(
            title="Photo공정 과거 OPC 매핑",
            aliases=["Photo공정", "OPC", "CROPC", "MASK"],
            match_fields=["diagnosis_job_name", "self_review"],
            hint="2021~2024년 Photo공정 진단명과 OPC 업무 단서가 함께 있으면 DIC > OPC 후보를 우선한다.",
            target_major_job="DIC",
            target_sub_job="OPC",
        ),
    )
    store.update_metadata(entry.id, enforcement_level="near_hard")
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
        teams=[],
        job_names=["Photo공정"],
        categories=[],
        evidence_rows=[],
    )

    def fail_stage1(self, context_json, candidate_pairs=None):
        raise AssertionError("diagnosis job name should select the pair before stage1")

    def fixed_stage2(self, context_json, candidates):
        assert candidates == [taxonomy.rows[0]]
        return FinalClassificationResult(
            major_job="공정",
            sub_job="Photo공정",
            device="",
            unit_job="Photo",
            detail_job_1="",
            detail_job_2="",
            confidence=0.9,
            needs_review=False,
            reason="2025년은 과거 Photo공정 보정 범위 밖이므로 진단 직무명 유지",
        )

    classifier._run_stage1 = MethodType(fail_stage1, classifier)
    classifier._run_stage2 = MethodType(fixed_stage2, classifier)

    result = classifier.classify_row(
        {
            "year": "2025",
            "team": "",
            "emp_num": "E0004",
            "name": "홍길동",
            "self_review": "OPC 조건 확보, CROPC 적용, MASK 확보 업무 수행",
        },
        diagnosis_context=diagnosis_context,
    )

    assert result["중직무"] == "공정"
    assert result["소직무"] == "Photo공정"
    assert "diagnosis 직무명 우선 적용" in result["knowledge_priority_reason"]


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
