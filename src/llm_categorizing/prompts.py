from __future__ import annotations

import json


SYSTEM_PROMPT = """너는 사내 직무 분류기다.
반드시 제공된 후보 목록 안에서만 직무를 선택해야 한다.
후보 목록에 없는 직무명, 계층, 조합을 새로 만들면 안 된다.
판단 근거는 self_review 문장에 있는 업무 내용이어야 한다.
name과 emp_num은 개인정보이므로 제공되지 않는다.
확실하지 않으면 가장 가까운 후보를 고르되 confidence를 낮게 주고 needs_review를 true로 둔다.
응답은 설명 문장 없이 JSON 객체 하나만 출력한다."""


def employee_context(
    *,
    year: str,
    team: str,
    self_review: str,
    include_team: bool,
    input_truncated: bool,
) -> str:
    payload = {
        "year": year,
        "self_review": self_review,
        "input_truncated": input_truncated,
    }
    if include_team:
        payload["team"] = team
    return json.dumps(payload, ensure_ascii=False, indent=2)


def stage1_user_prompt(context_json: str, candidate_pairs_json: str) -> str:
    return f"""아래 구성원의 성과리뷰를 기준으로 중직무와 소직무를 선택하라.

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
