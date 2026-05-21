# LLM 직무 카테고라이징

사내 OpenAI-compatible LLM API를 사용해 구성원별 `self_review` 문장을 고정된 직무 taxonomy로 분류하는 CLI 프로젝트입니다.

## 입력 CSV

구성원 CSV 필수 컬럼:

```text
year,team,emp_num,name,self_review
```

직무 taxonomy CSV 필수 컬럼:

```text
중직무,소직무,Device,단위 직무,세부 직무1,세부 직무2
```

taxonomy는 위 컬럼 순서대로 상위에서 하위로 계층화된 값이어야 합니다. LLM 결과가 taxonomy에 존재하지 않는 조합이면 결과를 채택하지 않고 `needs_review=True`로 처리합니다.

직무/스킬셋 진단 CSV 선택 컬럼:

```text
year,emp_num,name,team,진단 시 직무명,Category,항목
```

진단 CSV는 같은 `year + emp_num`에 여러 행이 있어도 됩니다. 여러 행은 한 구성원의 해당 연도 진단 근거로 묶어서 사용합니다. 단, `항목` 값은 분류 프롬프트, 지식 검색 컨텍스트, 결과 CSV에 넣지 않습니다.

## Windows 10 셋업

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

`.env` 파일을 열어 사내 LLM endpoint를 설정합니다.

```text
LLM_ENDPOINT_PROFILE=internal
INTERNAL_LLM_BASE_URL=https://your-internal-llm-endpoint/v1
INTERNAL_LLM_API_KEY=replace-me
INTERNAL_LLM_MODEL=shared-default-model
LLM_PROVIDER_PROFILE=auto
LLM_TIMEOUT_SECONDS=300
```

분류 판단 모델과 지식 저장/메모리 정리 모델은 역할별 환경변수로 따로 지정할 수 있습니다. 역할별 값이 있으면 우선 사용하고, 없으면 위 공통 `LLM_*`/`INTERNAL_LLM_*`/`ALIBABA_*` 값을 fallback으로 사용합니다.

```text
CLASSIFICATION_LLM_ENDPOINT_PROFILE=internal
CLASSIFICATION_INTERNAL_LLM_MODEL=glm-5.1
CLASSIFICATION_LLM_PROVIDER_PROFILE=glm
CLASSIFICATION_LLM_GLM_MAX_TOKENS=2048
CLASSIFICATION_LLM_GLM_EXTRA_BODY_JSON=

KNOWLEDGE_LLM_ENDPOINT_PROFILE=internal
KNOWLEDGE_INTERNAL_LLM_MODEL=Qwen3.6-35B-A3B
KNOWLEDGE_LLM_PROVIDER_PROFILE=qwen
KNOWLEDGE_LLM_QWEN_DISABLE_THINKING=1
KNOWLEDGE_LLM_MAX_TOKENS=1200
```

Alibaba/DashScope OpenAI-compatible API를 쓰려면 endpoint profile을 바꿉니다.

```text
LLM_ENDPOINT_PROFILE=alibaba
ALIBABA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ALIBABA_API_KEY=replace-me
ALIBABA_MODEL=qwen-plus
LLM_PROVIDER_PROFILE=qwen
LLM_QWEN_DISABLE_THINKING=0
```

Alibaba endpoint에서 역할별 모델만 나누려면 아래처럼 `CLASSIFICATION_ALIBABA_MODEL`, `KNOWLEDGE_ALIBABA_MODEL`을 사용하세요.

```text
CLASSIFICATION_LLM_ENDPOINT_PROFILE=alibaba
CLASSIFICATION_ALIBABA_MODEL=glm-5.1
CLASSIFICATION_LLM_PROVIDER_PROFILE=glm

KNOWLEDGE_LLM_ENDPOINT_PROFILE=alibaba
KNOWLEDGE_ALIBABA_MODEL=Qwen3.6-35B-A3B
KNOWLEDGE_LLM_PROVIDER_PROFILE=qwen
KNOWLEDGE_LLM_QWEN_DISABLE_THINKING=1
```

사내 endpoint가 OpenAI의 `response_format={"type":"json_object"}` 옵션을 지원하지 않으면 `LLM_USE_JSON_RESPONSE_FORMAT=0`을 유지하세요.

`LLM_PROVIDER_PROFILE=auto`는 모델명에 따라 `qwen`, `glm`, `generic` 프로파일을 자동 선택합니다. 역할별 모델명이 사내 별칭이라 자동 인식이 어렵다면 `CLASSIFICATION_LLM_PROVIDER_PROFILE`, `KNOWLEDGE_LLM_PROVIDER_PROFILE`로 직접 지정하세요.

```text
LLM_PROVIDER_PROFILE=qwen
LLM_PROVIDER_PROFILE=glm
LLM_PROVIDER_PROFILE=generic
```

Qwen3/Qwen3.6 계열 모델은 분류 품질을 우선해 기본적으로 thinking mode를 사용합니다. Qwen 프로파일은 기본적으로 아래 옵션을 요청 body에 추가합니다.

```text
LLM_QWEN_DISABLE_THINKING=0
```

사내에서 `qwen3.6-35b-a3b`를 쓸 때는 `LLM_PROVIDER_PROFILE=auto` 또는 `qwen`으로 두면 분류 판단 호출은 thinking mode로 동작합니다. 이때 `LLM_MAX_TOKENS`를 비워두면 Qwen thinking용 기본값 `LLM_QWEN_THINKING_MAX_TOKENS=4096`을 사용합니다.

```text
LLM_QWEN_DISABLE_THINKING=0
LLM_QWEN_THINKING_MAX_TOKENS=4096
```

그래도 최종 JSON이 비면 thinking 토큰이 많이 소모된 것이므로 `8192`까지 올려 테스트하세요.

```text
LLM_QWEN_THINKING_MAX_TOKENS=8192
```

사내 endpoint가 Roo Code처럼 별도 body 옵션을 요구하면 `.env`에 JSON 객체로 추가할 수 있습니다.

```text
LLM_QWEN_EXTRA_BODY_JSON={"chat_template_kwargs":{"enable_thinking":true}}
```

지식 입력 페이지의 메모리/지식 저장 정리 호출은 분류 호출과 달리 hallucination을 줄이는 것이 더 중요하므로, Qwen 프로파일에서도 코드가 `enable_thinking=false`를 별도로 요청합니다.

GLM 계열은 Qwen 전용 extra body를 보내지 않습니다. GLM에 별도 옵션이 필요할 때만 GLM 전용 값을 넣으세요.

```text
LLM_GLM_MAX_TOKENS=2048
LLM_GLM_EXTRA_BODY_JSON=
```

## 실행

기본 경로:

```text
구성원 CSV: data/input/employees.csv
직무/스킬셋 진단 CSV: data/input/diagnosis.csv
직무 taxonomy CSV: data/input/taxonomy.csv
분류 결과 CSV: data/output/classified_jobs.csv
```

위 경로에 파일을 두면 경로 인자 없이 실행할 수 있습니다. 진단 CSV는 선택 파일이라 없으면 자동으로 건너뜁니다.

스키마와 taxonomy만 검증:

```bat
python classify_jobs.py --validate-only
```

실제 분류:

```bat
python classify_jobs.py
```

처음에는 일부 행만 테스트하는 것을 권장합니다.

```bat
python classify_jobs.py --limit 20 --output data\output\classified_jobs_sample.csv
```

다른 경로를 써야 할 때만 인자로 덮어쓰면 됩니다.

```bat
python classify_jobs.py ^
  --input examples\employees_sample.csv ^
  --diagnosis examples\diagnosis_sample.csv ^
  --taxonomy examples\taxonomy_sample.csv ^
  --output data\output\classified_jobs_sample.csv
```

진단 데이터를 일부러 쓰지 않으려면:

```bat
python classify_jobs.py --no-diagnosis
```

사용자 입력 지식 DB를 쓰지 않으려면:

```bat
python classify_jobs.py --knowledge-db-path ""
```

정답 라벨 CSV가 있으면 분류 결과를 수치로 평가할 수 있습니다. 정답 CSV는 `year`,
`emp_num`과 taxonomy 6개 컬럼을 포함해야 합니다.

```bat
python evaluate_classification.py ^
  --predictions data\output\classified_jobs.csv ^
  --gold data\input\classified_jobs_gold.csv
```

## 지식 입력 페이지

분류 판단에 필요한 도메인 지식은 간단한 로컬 페이지에서 추가할 수 있습니다. 입력된 원문은 그대로 저장하고, LLM이 제목/매칭 용어/적용 조건/힌트/관련 taxonomy 계층으로 정리한 구조화 데이터도 함께 SQLite에 저장합니다.

```bat
python serve_knowledge.py
```

브라우저에서 아래 주소를 열어 지식을 입력합니다.

```text
http://127.0.0.1:8765
```

기본 DB 경로는 `data/output/job_knowledge.sqlite3`입니다. 다른 경로를 쓰려면 페이지 서버와 분류 실행 모두 같은 경로를 지정하세요.

```bat
python serve_knowledge.py --knowledge-db-path data\output\job_knowledge.sqlite3
python classify_jobs.py --knowledge-db-path data\output\job_knowledge.sqlite3
```

검증 완료된 지식만 분류에 쓰고 싶으면 아래처럼 실행합니다.

```bash
python classify_jobs.py --knowledge-review-scope approved
```

지식 입력 페이지는 기본적으로 `data/input/taxonomy.csv`가 있으면 LLM 정리 시 taxonomy 참고값을 함께 전달하고, 저장 전 target 값이 실제 taxonomy에 있는지 검증합니다. 다른 taxonomy를 쓰려면:

```bat
python serve_knowledge.py --taxonomy examples\taxonomy_sample.csv
```

LLM 환경변수가 아직 없지만 UI만 시험하려면 fallback draft 저장을 허용할 수 있습니다. 운영에서는 LLM 정리 결과를 쓰는 것을 권장합니다.

```bat
python serve_knowledge.py --allow-fallback-normalizer
```

저장된 지식은 처음에는 `soft_hint`/`draft`로 취급됩니다. 페이지에서 `승격`을 누르면 `verified_rule`/`approved`로 바뀌며, 분류 prompt에서 더 강한 참고 지식으로 전달됩니다. 그래도 자동 보정 rule은 아니므로, 실제 정확도 개선 여부는 샘플 재분류와 수동 검수로 확인해야 합니다.

긴 텍스트 파일은 페이지의 `TXT 줄 단위 가져오기`에서 업로드할 수 있습니다. 빈 줄은 제외하고 줄바꿈 1줄을 지식 1개로 저장합니다. 기본 제한은 한 번에 최대 100줄, 줄당 최대 4,000자, 요청 본문 최대 1MB입니다.

## 개인정보 처리 기준

- LLM 프롬프트에는 `name`, `emp_num`을 보내지 않습니다.
- 진단 CSV도 `name`, `emp_num`은 조인 키/출력 매핑에만 쓰고 LLM 프롬프트에는 보내지 않습니다.
- 기본적으로 `team`도 LLM에 전달하지 않습니다. 현시점 팀명이 과거 연도와 맞지 않으면 오분류 bias가 생길 수 있기 때문입니다.
- 단, 진단 CSV의 `team`은 진단 당시 팀 정보라서 LLM 프롬프트에 보조 근거로 사용합니다.
- 연도별 팀 정보가 정확해서 분류 단서로 쓰고 싶을 때만 `--include-team-in-prompt`를 사용합니다.
- 결과 CSV에는 기본적으로 `self_review` 원문을 저장하지 않습니다.
- 원문까지 결과에 포함해야 할 때만 `--include-self-review-output`을 사용합니다.
- `data/input/*`, `data/output/*`, `.env`는 git ignore 처리되어 있습니다.

## 출력 컬럼

기본 출력:

```text
year,team,emp_num,name,
중직무,소직무,Device,단위 직무,세부 직무1,세부 직무2,
confidence,reason,needs_review,ambiguity_reason,guardrail_reason,diagnosis_priority_reason,knowledge_priority_reason,
previous_year,previous_year_job_path,previous_year_confidence,previous_year_needs_review,
used_knowledge_ids,used_knowledge_types,used_knowledge_scores,
used_knowledge_review_statuses,used_knowledge_enforcement_levels,
used_knowledge_match_fields,knowledge_review_scope,knowledge_version,
diagnosis_row_count,diagnosis_teams,diagnosis_job_names,diagnosis_categories,
error,input_truncated,taxonomy_version,model_name,classified_at
```

`needs_review=True`가 되는 주요 경우:

- `self_review`가 비어 있음
- LLM 결과가 taxonomy에 없는 조합임
- confidence가 `--confidence-review-threshold`보다 낮음
- API 호출 또는 JSON 파싱 실패
- 후보가 너무 많아 안전하게 분류할 수 없음

## 분류 방식

1. 진단 CSV가 있으면 `year + emp_num`으로 매칭된 `team`, `진단 시 직무명`을 먼저 봅니다.
2. `진단 시 직무명`이 taxonomy의 `소직무`와 정확히 매칭되면 해당 `중직무`/`소직무` 후보를 우선 사용합니다. `소직무` 정확 매칭이 없지만 taxonomy의 `단위 직무`와 유일하게 정확 매칭되면 그 row의 `중직무`/`소직무`를 사용합니다. 부분 문자열 매칭은 후보 제한에 쓰지 않습니다.
3. diagnosis의 `team`은 후보를 강제로 제한하지 않습니다. 다만 team에 taxonomy `중직무` 값이 직접 포함되면 soft hint로 전달하고, `TD -> 소자`, `Heraion -> NAND` 같은 사내 조직/프로젝트/제품 alias는 지식 DB에서 검색된 `classification_hints`로 LLM에 전달합니다.
4. 실행 순서는 구성원별 `year` 오름차순입니다. 같은 구성원의 직전 연도 결과가 있고 `needs_review=False`, confidence가 검토 threshold 이상이면 `previous_year_classification`으로 prompt에 넣어 직무 연속성 참고 정보로 사용합니다.
5. diagnosis로 후보가 하나로 좁혀지지 않으면 남은 `중직무`/`소직무` 후보 중 하나를 LLM이 선택합니다.
6. 선택된 pair의 하위 taxonomy row만 후보로 넣어 최종 계층을 선택합니다.
7. 최종 결과가 taxonomy CSV의 row와 정확히 일치하는지 검증합니다.
8. 동일 입력은 `data/output/classification_cache.jsonl`에 캐시해 재실행 비용과 결과 흔들림을 줄입니다.

## 정확도 개선 로직

- 코드에 내장된 직무별 키워드 룰은 사용하지 않습니다.
- 입력 CSV의 `self_review`와 선택 입력인 `diagnosis_context`는 LLM에 근거 데이터로 전달합니다. `diagnosis_context`에는 diagnosis `team`, `진단 시 직무명`, `Category` 요약만 넣고 `항목` 값은 넣지 않습니다.
- diagnosis의 `진단 시 직무명`은 taxonomy에 실제 존재하는 값과 정확히 매칭될 때만 `중직무`/`소직무` 후보 제한에 사용합니다. 적용 여부는 `diagnosis_priority_reason` 컬럼에 기록됩니다. 부분 문자열 단서는 prompt 참고 정보로만 남기고 hard restrict에는 쓰지 않습니다.
- diagnosis의 `team`은 후보 제한 rule로 쓰지 않고, 직접 보이는 taxonomy 중직무 표현과 지식 DB에 저장된 alias/제품 지식을 LLM 판단 근거로 전달합니다. `TD` 같은 2글자 alias도 team에서 독립 token으로 매칭되면 지식 검색에 사용합니다.
- 직전 연도 결과는 같은 `emp_num`의 `year-1` 결과가 있고 오류가 없으며 `needs_review=False`, confidence가 검토 threshold 이상일 때만 사용합니다. 현재 연도 self_review/diagnosis와 충돌하면 현재 연도 근거를 우선하도록 prompt에 명시합니다.
- 사용자가 지식 입력 페이지로 추가한 지식은 `self_review`, diagnosis `team`, 진단 직무명, category를 분리해서 검색한 뒤 점수가 높은 일부만 `classification_hints`에 넣습니다. target taxonomy 값만으로 지식을 검색하지 않고 alias 매칭이 있을 때 target/본문 overlap을 boost로만 사용합니다. diagnosis `항목` 값은 지식 검색에도 사용하지 않습니다. 결과 CSV의 `used_knowledge_ids`, `used_knowledge_types`, `used_knowledge_scores`, `used_knowledge_enforcement_levels`, `used_knowledge_match_fields`, `knowledge_version`으로 어떤 지식이 쓰였는지 추적할 수 있습니다.
- 지식 DB에는 `knowledge_type`, `review_status`, `enforcement_level`, `match_fields`, `conflicts`를 저장합니다. `승격`은 `strong`, `준하드룰`은 `near_hard`로 저장됩니다. `near_hard` 지식은 현재 입력에 매칭되면 해당 target과 맞는 taxonomy row로 stage1/stage2 후보를 먼저 제한합니다. 최종 후보가 1개로 좁혀지면 LLM 호출 없이 해당 row를 선택하며, 적용 여부는 `knowledge_priority_reason`에 기록합니다.
- 같은 raw 지식이 다시 들어오면 새 row를 만들지 않고 기존 row에 alias/source/priority를 병합합니다. 같은 alias가 서로 다른 target을 가리키면 `conflicts`와 검증 경고로 표시해 사람이 확인할 수 있게 합니다.
- 분류 시 검색된 지식은 `knowledge_usage` 테이블에 `classification_id`, `knowledge_id`, `match_score`, 최종 분류 결과와 함께 기록되어 나중에 어떤 지식이 실제 분류에 자주 쓰였는지 점검할 수 있습니다.
- taxonomy 중복 단위직무를 후보에 별도 주의 정보로 주입하던 로직도 분류 판단에서는 제거했습니다. `ambiguity_reason` 컬럼은 과거 출력 스키마 호환을 위해 남아 있지만 새 분류에서는 빈 값입니다.
- 기존 룰 기반 자동 보정은 제거했습니다. `guardrail_reason` 컬럼은 과거 출력 스키마 호환을 위해 남아 있지만 새 분류에서는 빈 값입니다.
- 프롬프트 버전을 cache key에 포함해, 프롬프트 개선 전의 기존 cache 결과가 재사용되지 않도록 했습니다.

## GLM-5 운영 튜닝

사내 GLM-5 endpoint가 혼잡해서 `classification_error: empty LLM response`가 많이 발생하면 `.env`에서 대기 시간을 늘리세요.

```text
LLM_TIMEOUT_SECONDS=300
```

기본 API 재시도 횟수는 5회입니다. 더 오래 기다려도 되는 배치 작업이면 실행 시 재시도를 늘릴 수 있습니다.

```bat
python classify_jobs.py --api-retry-attempts 8
```

## 운영 팁

- 처음부터 전수 자동화하지 말고, 200~500건 샘플을 사람이 검수해 직무별 confusion pattern을 확인하세요.
- `confidence < 0.6` 또는 `needs_review=True`인 건은 운영 반영 전에 수동 검수하는 흐름을 권장합니다.
