from __future__ import annotations

import json


PROMPT_VERSION = "job-classification-v3-diagnosis-context"


DECISION_RULES = """[분류 판단 규칙]
- 중직무는 self_review에 나타난 실제 수행 업무의 동사, 산출물, 검증 활동 기준으로 판단한다.
- Device 값 또는 제품명(DRAM/NAND/Logic/Cell 등)은 대상 제품 정보일 수 있으며, 중직무 판단의 직접 근거로 사용하지 않는다.
- MLM처럼 여러 중직무에 중복 존재하는 단위 직무는 단독 근거로 쓰지 말고 주변 업무 표현으로 구분한다.
- Process Qual, 공정 조건 평가, Scheme 개발, Base Line 변경, Process Flow 검토, Reticle, MTS, PLR, 단위공정 Tuning, Low-k IMD 평가는 공정 수행 근거로 강하게 본다.
- 소자는 소자 구조/전기적 특성/Device characterization/Reliability/Cell 특성/소자 원인 분석이 핵심일 때 선택한다.
- 공정 실행/평가/qual/baseline 변경 단서와 제품명 또는 Device명이 함께 나오면 제품명보다 공정 수행 단서를 우선한다.
- diagnosis_context는 해당 연도 구성원의 직무/스킬셋 자가진단 데이터이며, self_review보다 구조화된 강한 보조 근거로 사용한다.
- diagnosis_context에 여러 evidence_rows가 있으면 같은 구성원의 같은 연도에 여러 진단 항목이 존재한다는 의미이며, 반복되는 team/job/category/item 단서를 더 강하게 본다.
- diagnosis_team에 "DRAM공정 > DPC"처럼 공정 조직 단서가 있으면 중직무 공정 근거로 강하게 본다.
- diagnosis_job_name이 "Etch공정"이면 중직무 공정, 소직무 Etch공정 근거로 강하게 본다.
- diagnosis_context의 Category가 CLN/MLM 등 taxonomy의 단위 직무와 맞으면 단위 직무 판단 근거로 사용한다.
- 후보에 '분류주의'가 있으면 해당 후보는 taxonomy상 중복 의미가 있으므로 needs_review를 true로 두거나 confidence를 보수적으로 낮춘다.

[오분류 방지 예시]
self_review에 "MLM Module 기술적 지원", "Process Qual", "Base Line 변경", "Via First Scheme",
"Low-k IMD 평가", "Process Flow 검토", "단위공정 Tuning"이 함께 나타나면
DRAM 또는 Lucy Base Line 같은 제품/라인 단어가 있어도 중직무는 소자보다 공정 후보를 우선 검토한다."""


SYSTEM_PROMPT = """너는 사내 직무 분류기다.
반드시 제공된 후보 목록 안에서만 직무를 선택해야 한다.
후보 목록에 없는 직무명, 계층, 조합을 새로 만들면 안 된다.
판단 근거는 self_review 문장에 있는 업무 내용이어야 한다.
name과 emp_num은 개인정보이므로 제공되지 않는다.
team은 설정에 따라 제공되지 않을 수 있으며, 제공되더라도 self_review의 업무 내용을 우선해야 한다.
확실하지 않으면 가장 가까운 후보를 고르되 confidence를 낮게 주고 needs_review를 true로 둔다.
응답은 설명 문장 없이 JSON 객체 하나만 출력한다."""


def employee_context(
    *,
    year: str,
    team: str,
    self_review: str,
    include_team: bool,
    input_truncated: bool,
    classification_hints: list[str] | None = None,
    diagnosis_context: dict[str, object] | None = None,
) -> str:
    payload = {
        "year": year,
        "self_review": self_review,
        "input_truncated": input_truncated,
    }
    if include_team:
        payload["team"] = team
    if classification_hints:
        payload["classification_hints"] = classification_hints
    if diagnosis_context:
        payload["diagnosis_context"] = diagnosis_context
    return json.dumps(payload, ensure_ascii=False, indent=2)


def stage1_user_prompt(context_json: str, candidate_pairs_json: str) -> str:
    return f"""아래 구성원의 성과리뷰를 기준으로 중직무와 소직무를 선택하라.

{DECISION_RULES}

[구성원 입력]
{context_json}

[선택 가능한 후보 목록]
{candidate_pairs_json}

[출력 JSON 형식]
{{
  "중직무": "후보 목록의 중직무 값",
  "소직무": "후보 목록의 소직무 값",
  "confidence": 0.0,
  "reason": "self_review의 어떤 표현 때문에 이 후보를 골랐는지 짧게 설명",
  "needs_review": true
}}"""


def stage2_user_prompt(context_json: str, full_candidates_json: str) -> str:
    return f"""아래 구성원의 성과리뷰를 기준으로 최종 직무 계층 하나를 선택하라.
반드시 후보 목록의 객체 하나와 동일한 계층 조합을 출력해야 한다.
후보 객체에서 빈 문자열("")인 값은 출력에서도 빈 문자열로 유지한다.
후보 객체의 '분류주의' 필드는 참고 정보이며, 출력 JSON에는 포함하지 않는다.

{DECISION_RULES}

[구성원 입력]
{context_json}

[선택 가능한 최종 후보 목록]
{full_candidates_json}

[출력 JSON 형식]
{{
  "중직무": "후보 목록의 중직무 값",
  "소직무": "후보 목록의 소직무 값",
  "Device": "후보 목록의 Device 값",
  "단위 직무": "후보 목록의 단위 직무 값",
  "세부 직무1": "후보 목록의 세부 직무1 값",
  "세부 직무2": "후보 목록의 세부 직무2 값",
  "confidence": 0.0,
  "reason": "self_review의 어떤 표현 때문에 이 후보를 골랐는지 짧게 설명",
  "needs_review": true
}}"""


def correction_prompt(error: str, previous_output: str) -> str:
    return f"""이전 응답은 유효하지 않다.

[검증 오류]
{error}

[이전 응답]
{previous_output}

후보 목록에 존재하는 값만 사용해서 JSON 객체 하나만 다시 출력하라."""
