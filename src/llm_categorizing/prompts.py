from __future__ import annotations

import json


PROMPT_VERSION = "job-classification-v8-prior-year-context"


DECISION_RULES = """[분류 판단 규칙]
- 반드시 제공된 후보 목록 안에서만 선택한다.
- 후보 목록에 없는 직무명, 계층, 조합을 새로 만들지 않는다.
- 판단 근거는 self_review, diagnosis_context, classification_hints에 제공된 내용으로 제한한다.
- diagnosis_context가 있으면 year+emp_num으로 매칭된 진단 당시 데이터다. 진단 당시 직무명은 중직무/소직무 판단의 우선 근거로 사용한다.
- diagnosis_context의 team에는 사내 프로젝트명/제품 alias가 들어갈 수 있다. 해당 의미는 classification_hints에 제공된 사용자 지식이 있을 때만 그 지식에 따라 해석한다.
- classification_hints에는 diagnosis team에 직접 포함된 taxonomy 중직무 표현이나 사용자 지식 alias 매칭 정보가 들어갈 수 있다. 이는 자동 보정 rule이 아니라 판단 참고다.
- classification_hints가 있으면 사용자가 저장한 지식 DB에서 검색된 참고 지식이다. self_review 또는 diagnosis_context와 충돌하지 않는 범위에서만 참고한다.
- previous_year_classification이 있으면 같은 구성원의 직전 연도 분류 결과다. 직무 연속성 참고로 사용하되, 현재 연도 self_review/diagnosis_context와 충돌하면 현재 연도 근거를 우선한다.
- 코드에 내장된 직무별 키워드 규칙은 없으므로, 특정 용어만으로 사내 도메인 규칙을 임의 생성하지 않는다.
- 확실하지 않으면 가장 가까운 후보를 고르되 confidence를 낮게 주고 needs_review를 true로 둔다."""


SYSTEM_PROMPT = """너는 사내 직무 분류기다.
반드시 제공된 후보 목록 안에서만 직무를 선택해야 한다.
후보 목록에 없는 직무명, 계층, 조합을 새로 만들면 안 된다.
판단 근거는 self_review와 year+emp_num으로 매칭된 diagnosis_context에 있는 내용이어야 한다.
name과 emp_num은 개인정보이므로 제공되지 않는다.
team은 설정에 따라 제공되지 않을 수 있으며, 제공되더라도 self_review의 업무 내용을 우선해야 한다.
diagnosis_context 안의 직무명은 진단 당시 데이터이므로, 제공되면 중직무/소직무 판단에 우선 활용한다.
diagnosis_context 안의 team은 조직/프로젝트/제품 alias 단서일 수 있다. team의 사내 의미는 classification_hints에 있는 사용자 지식으로 해석하고, 근거 없는 alias 의미를 새로 만들지 않는다.
classification_hints의 diagnosis team 단서는 후보를 강제하지 않는 참고 정보이며, 제공된 후보 목록 안에서만 최종 판단한다.
previous_year_classification은 같은 구성원의 직전 연도 분류 결과다. 현재 연도 근거가 약하고 직무가 이어지는 정황이면 참고하되, 현재 연도 근거를 덮어쓰는 자동 보정 rule로 사용하지 않는다.
사전에 하드코딩된 직무별 키워드 규칙은 사용하지 않는다.
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
    previous_year_classification: dict[str, object] | None = None,
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
    if previous_year_classification:
        payload["previous_year_classification"] = previous_year_classification
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
  "reason": "self_review, diagnosis_context 또는 previous_year_classification의 어떤 근거 때문에 이 후보를 골랐는지 짧게 설명",
  "needs_review": true
}}"""


def stage2_user_prompt(context_json: str, full_candidates_json: str) -> str:
    return f"""아래 구성원의 성과리뷰를 기준으로 최종 직무 계층 하나를 선택하라.
반드시 후보 목록의 객체 하나와 동일한 계층 조합을 출력해야 한다.
후보 객체에서 빈 문자열("")인 값은 출력에서도 빈 문자열로 유지한다.

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
  "reason": "self_review, diagnosis_context 또는 previous_year_classification의 어떤 근거 때문에 이 후보를 골랐는지 짧게 설명",
  "needs_review": true
}}"""


def correction_prompt(error: str, previous_output: str) -> str:
    return f"""이전 응답은 유효하지 않다.

[검증 오류]
{error}

[이전 응답]
{previous_output}

후보 목록에 존재하는 값만 사용해서 JSON 객체 하나만 다시 출력하라."""
