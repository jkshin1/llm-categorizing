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

진단 CSV는 같은 `year + emp_num`에 여러 행이 있어도 됩니다. 여러 행은 한 구성원의 해당 연도 진단 근거로 묶어서 사용합니다.

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
INTERNAL_LLM_BASE_URL=https://your-internal-llm-endpoint/v1
INTERNAL_LLM_API_KEY=replace-me
INTERNAL_LLM_MODEL=your-internal-model-name
LLM_TIMEOUT_SECONDS=300
```

사내 endpoint가 OpenAI의 `response_format={"type":"json_object"}` 옵션을 지원하지 않으면 `LLM_USE_JSON_RESPONSE_FORMAT=0`을 유지하세요.

Qwen3/Qwen3.6 계열 모델은 thinking mode 때문에 OpenAI-compatible API의 `message.content`가 빈 값으로 내려오는 경우가 있습니다. 이 프로젝트는 모델명에 `qwen`이 들어가면 기본적으로 아래 옵션을 요청 body에 추가합니다.

```text
LLM_QWEN_DISABLE_THINKING=1
```

사내 endpoint가 Roo Code처럼 별도 body 옵션을 요구하면 `.env`에 JSON 객체로 추가할 수 있습니다.

```text
LLM_EXTRA_BODY_JSON={"chat_template_kwargs":{"enable_thinking":false}}
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
confidence,reason,needs_review,ambiguity_reason,guardrail_reason,
diagnosis_row_count,diagnosis_teams,diagnosis_job_names,diagnosis_categories,diagnosis_items,
error,input_truncated,taxonomy_version,model_name,classified_at
```

`needs_review=True`가 되는 주요 경우:

- `self_review`가 비어 있음
- LLM 결과가 taxonomy에 없는 조합임
- confidence가 `--confidence-review-threshold`보다 낮음
- `MLM`처럼 동일한 `단위 직무`가 여러 `중직무`에 중복 존재함
- API 호출 또는 JSON 파싱 실패
- 후보가 너무 많아 안전하게 분류할 수 없음

## 분류 방식

1. `중직무`/`소직무` 후보 중 하나를 먼저 선택합니다.
2. 선택된 pair의 하위 taxonomy row만 후보로 넣어 최종 계층을 선택합니다.
3. 최종 결과가 taxonomy CSV의 row와 정확히 일치하는지 검증합니다.
4. 진단 CSV가 있으면 `year + emp_num` 기준으로 여러 행을 묶어 team/job/category/item 근거로 함께 사용합니다.
5. 동일 입력은 `data/output/classification_cache.jsonl`에 캐시해 재실행 비용과 결과 흔들림을 줄입니다.

## 정확도 개선 로직

- 입력 CSV 컬럼을 추가하지 않고 `self_review` 문장 안의 업무 단서를 자동 감지해 프롬프트 참고 정보로 전달합니다.
- `DRAM`, `NAND`, `Logic` 같은 제품/Device 용어는 중직무 직접 근거로 쓰지 않도록 프롬프트에서 제한합니다.
- `Process Qual`, `Base Line`, `Scheme`, `Process Flow`, `Reticle`, `MTS`, `PLR`, `단위공정 Tuning`, `Low-k IMD` 등은 공정 수행 근거로 우선 검토하게 했습니다.
- taxonomy에서 동일한 `단위 직무`가 여러 `중직무`에 존재하면 후보에 `분류주의`를 붙이고, 결과 CSV의 `ambiguity_reason`에 검수 사유를 남깁니다.
- 강한 공정 수행 단서가 있고 동일 `단위 직무`가 소자/공정에 중복될 때, 동일 Device의 공정 후보가 유일하면 `guardrail_reason`을 남기고 공정 후보로 보정합니다.
- 진단 데이터의 `team`에 `DRAM공정 > DPC`, `진단 시 직무명`에 `Etch공정`, `Category`에 `CLN`/`MLM`처럼 taxonomy와 맞는 단서가 있으면 강한 보조 근거로 사용합니다.
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
