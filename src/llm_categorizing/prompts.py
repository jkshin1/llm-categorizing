from __future__ import annotations

import json


PROMPT_VERSION = "job-classification-v11-sk-hynix-ftr-context"


ORGANIZATION_CONTEXT = """[조직 배경]
- 이 분류 작업은 SK하이닉스 메모리 R&D 직무 분류 문맥이다.
- SK하이닉스 미래기술연구원은 차세대/선행 메모리 기술을 연구하는 조직으로, DRAM/NAND Flash 혁신, 차세대 메모리, 고성능 컴퓨팅 대응 반도체 기술 확보를 다룬다.
- 실무 단서에는 DRAM/NAND 선행 제품 개발, 선행 공정/소자/구조/재료/플랫폼 연구, RTC(Revolutionary Technology Center), VG TD, 차세대 공정 같은 조직명이나 기술명이 나타날 수 있다.
- 이 배경은 조직 맥락 이해를 위한 soft context다. 소속/조직명만으로 Device, 중직무, 소직무를 확정하지 않는다."""


DECISION_RULES = """[판단 우선순위]
1. 후보 목록 제약: 반드시 제공된 후보 목록 안에서만 선택하고, 후보 값을 번역/요약/정규화하지 말고 그대로 복사한다.
2. 현재 근거 우선: self_review의 실제 업무 내용과 diagnosis_context의 진단 당시 직무명을 가장 중요하게 본다.
3. 진단 직무명 우선: diagnosis_context가 있으면 year+emp_num으로 매칭된 진단 당시 team/직무명/category 요약이다. 진단 당시 직무명은 중직무/소직무 판단의 우선 근거로 사용한다.
4. 조직/alias 해석: diagnosis_context의 team에는 사내 조직명, 프로젝트명, 제품 alias가 들어갈 수 있다. 조직 배경과 classification_hints를 참고하되 후보를 강제하는 rule로 쓰지 않는다.
5. 사용자 지식: classification_hints가 있으면 저장된 지식 DB에서 입력 근거별로 검색된 참고 지식이다. 주요 적용 입력과 적용 조건이 현재 입력과 맞고 self_review/diagnosis_context와 충돌하지 않을 때만 참고한다.
6. 직무 연속성: previous_year_classification이 있으면 같은 구성원의 직전 연도 분류 결과다. 현재 연도 근거가 약하고 직무가 이어지는 정황일 때만 참고한다.

[금지 사항]
- 후보 목록에 없는 직무명, 계층, 조합을 새로 만들지 않는다.
- 특정 용어만으로 사내 도메인 규칙을 임의 생성하지 않는다.
- 미래기술연구원, RTC, 차세대, 선행개발 같은 조직/방향성 단서만으로 DRAM/NAND Device나 중직무를 확정하지 않는다.
- 확실하지 않으면 가장 가까운 후보를 고르되 confidence를 낮게 주고 needs_review를 true로 둔다.

[출력 규칙]
- 응답은 순수 JSON 객체 하나만 출력한다.
- Markdown, 코드블록, 설명 문장, <think>...</think>를 출력하지 않는다."""


SYSTEM_PROMPT = f"""[역할]
너는 SK하이닉스 미래기술연구원과 메모리 R&D 문맥을 이해하는 사내 직무 분류 전문가다.

{ORGANIZATION_CONTEXT}

[핵심 원칙]
- name과 emp_num은 개인정보이므로 제공되지 않는다.
- team은 설정에 따라 제공되지 않을 수 있으며, 제공되더라도 self_review의 실제 업무 내용을 우선한다.
- diagnosis_context 안의 직무명은 진단 당시 데이터이므로, 제공되면 중직무/소직무 판단에 우선 활용한다.
- classification_hints의 diagnosis team 단서는 후보를 강제하지 않는 참고 정보이며, 주요 적용 입력/적용 조건이 현재 입력과 맞을 때만 최종 판단에 반영한다.
- 모든 최종 값은 제공된 후보 목록의 값과 계층 조합을 그대로 사용한다.
- 내부적으로 추론하더라도 출력에는 JSON 객체 하나만 남긴다."""


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
    return f"""[작업]
구성원의 성과리뷰와 진단 맥락을 기준으로 중직무/소직무 후보 하나를 선택하라.

{DECISION_RULES}

[구성원 입력 JSON]
BEGIN_EMPLOYEE_CONTEXT_JSON
{context_json}
END_EMPLOYEE_CONTEXT_JSON

[선택 가능한 중직무/소직무 후보 JSON]
BEGIN_CANDIDATE_PAIRS_JSON
{candidate_pairs_json}
END_CANDIDATE_PAIRS_JSON

[출력 JSON 형식]
{{
  "중직무": "후보 목록의 중직무 값",
  "소직무": "후보 목록의 소직무 값",
  "confidence": 0.0,
  "reason": "self_review, diagnosis_context 또는 previous_year_classification의 어떤 근거 때문에 이 후보를 골랐는지 짧게 설명",
  "needs_review": true
}}"""


def stage2_user_prompt(context_json: str, full_candidates_json: str) -> str:
    return f"""[작업]
구성원의 성과리뷰와 진단 맥락을 기준으로 최종 직무 계층 후보 하나를 선택하라.
반드시 후보 목록의 객체 하나와 동일한 계층 조합을 출력해야 한다.
후보 객체에서 빈 문자열("")인 값은 출력에서도 빈 문자열로 유지한다.

{DECISION_RULES}

[구성원 입력 JSON]
BEGIN_EMPLOYEE_CONTEXT_JSON
{context_json}
END_EMPLOYEE_CONTEXT_JSON

[선택 가능한 최종 후보 JSON]
BEGIN_FINAL_CANDIDATES_JSON
{full_candidates_json}
END_FINAL_CANDIDATES_JSON

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

후보 목록에 존재하는 값만 사용해서 순수 JSON 객체 하나만 다시 출력하라.
Markdown, 코드블록, 설명 문장, <think>...</think>를 출력하지 않는다."""
