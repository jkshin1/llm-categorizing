from llm_categorizing.prompts import (
    DECISION_RULES,
    ORGANIZATION_CONTEXT,
    PROCESS_DEVICE_SIGNAL_RULES,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    correction_prompt,
    stage1_user_prompt,
    stage2_user_prompt,
)


def test_system_prompt_includes_future_technology_research_context() -> None:
    assert PROMPT_VERSION == "job-classification-v18-dic-optional-unit"
    assert "SK하이닉스 미래기술연구원" in SYSTEM_PROMPT
    assert "DRAM/NAND Flash 혁신" in ORGANIZATION_CONTEXT
    assert "DRAM/NAND 선행 제품 개발" in ORGANIZATION_CONTEXT
    assert "soft context" in ORGANIZATION_CONTEXT
    assert "조직명만으로 Device, 중직무, 소직무를 확정하지 않는다" in ORGANIZATION_CONTEXT
    assert "준하드룰" in SYSTEM_PROMPT
    assert "현재 self_review와 명확히 충돌하면 self_review" in SYSTEM_PROMPT
    assert "DIC는 단위 직무까지 항상 채우지 않는다" in SYSTEM_PROMPT


def test_prompts_prioritize_etch_process_work_over_device_words() -> None:
    assert "M0C ETCH" in PROCESS_DEVICE_SIGNAL_RULES
    assert "공정 > Etch공정" in PROCESS_DEVICE_SIGNAL_RULES
    assert "소자 > Device" in PROCESS_DEVICE_SIGNAL_RULES
    assert "EBI" in PROCESS_DEVICE_SIGNAL_RULES
    assert "WT" in PROCESS_DEVICE_SIGNAL_RULES
    assert PROCESS_DEVICE_SIGNAL_RULES in DECISION_RULES
    assert "Etch module 개선" in SYSTEM_PROMPT


def test_prompts_allow_dic_blank_unit_when_unit_signal_is_weak() -> None:
    assert "DIC 단위 직무 선택 규칙" in DECISION_RULES
    assert "필수 확정 범위가 중직무, 소직무, Device까지" in DECISION_RULES
    assert "단위 직무 근거가 약하면" in DECISION_RULES
    assert "억지로 선택하지 않는다" in DECISION_RULES


def test_stage_prompts_use_stable_json_boundaries_and_output_rules() -> None:
    stage1 = stage1_user_prompt(
        '{"self_review": "DRAM 선행 공정 개발"}',
        '[{"중직무": "공정", "소직무": "Etch공정"}]',
    )
    stage2 = stage2_user_prompt(
        '{"self_review": "NAND 선행 소자 개발"}',
        '[{"중직무": "소자", "소직무": "Device", "Device": "NAND"}]',
    )

    assert "BEGIN_EMPLOYEE_CONTEXT_JSON" in stage1
    assert "END_EMPLOYEE_CONTEXT_JSON" in stage1
    assert "BEGIN_CANDIDATE_PAIRS_JSON" in stage1
    assert "BEGIN_FINAL_CANDIDATES_JSON" in stage2
    assert "Markdown, 코드블록, 설명 문장, <think>...</think>를 출력하지 않는다" in stage1
    assert "후보 값을 번역/요약/정규화하지 말고 그대로 복사한다" in stage2


def test_correction_prompt_reinforces_plain_json_only() -> None:
    prompt = correction_prompt("bad output", "```json\n{}\n```")

    assert "순수 JSON 객체 하나만" in prompt
    assert "Markdown, 코드블록, 설명 문장, <think>...</think>를 출력하지 않는다" in prompt
