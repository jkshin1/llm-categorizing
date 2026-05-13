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
```

사내 endpoint가 OpenAI의 `response_format={"type":"json_object"}` 옵션을 지원하지 않으면 `LLM_USE_JSON_RESPONSE_FORMAT=0`을 유지하세요.

## 실행

스키마와 taxonomy만 검증:

```bat
python classify_jobs.py ^
  --input examples\employees_sample.csv ^
  --taxonomy examples\taxonomy_sample.csv ^
  --output data\output\classified_jobs.csv ^
  --validate-only
```

실제 분류:

```bat
python classify_jobs.py ^
  --input data\input\employees.csv ^
  --taxonomy data\input\taxonomy.csv ^
  --output data\output\classified_jobs.csv
```

처음에는 일부 행만 테스트하는 것을 권장합니다.

```bat
python classify_jobs.py ^
  --input data\input\employees.csv ^
  --taxonomy data\input\taxonomy.csv ^
  --output data\output\classified_jobs_sample.csv ^
  --limit 20
```

## 개인정보 처리 기준

- LLM 프롬프트에는 `name`, `emp_num`을 보내지 않습니다.
- 기본적으로 `team`, `year`, `self_review`만 LLM에 전달합니다.
- 팀 정보도 제외하려면 `--exclude-team-from-prompt`를 사용합니다.
- 결과 CSV에는 기본적으로 `self_review` 원문을 저장하지 않습니다.
- 원문까지 결과에 포함해야 할 때만 `--include-self-review-output`을 사용합니다.
- `data/input/*`, `data/output/*`, `.env`는 git ignore 처리되어 있습니다.

## 출력 컬럼

기본 출력:

```text
year,team,emp_num,name,
중직무,소직무,Device,단위 직무,세부 직무1,세부 직무2,
confidence,reason,needs_review,error,input_truncated,taxonomy_version,model_name,classified_at
```

`needs_review=True`가 되는 주요 경우:

- `self_review`가 비어 있음
- LLM 결과가 taxonomy에 없는 조합임
- confidence가 `--confidence-review-threshold`보다 낮음
- API 호출 또는 JSON 파싱 실패
- 후보가 너무 많아 안전하게 분류할 수 없음

## 분류 방식

1. `중직무`/`소직무` 후보 중 하나를 먼저 선택합니다.
2. 선택된 pair의 하위 taxonomy row만 후보로 넣어 최종 계층을 선택합니다.
3. 최종 결과가 taxonomy CSV의 row와 정확히 일치하는지 검증합니다.
4. 동일 입력은 `data/output/classification_cache.jsonl`에 캐시해 재실행 비용과 결과 흔들림을 줄입니다.

## 운영 팁

- taxonomy에는 직무명만 두는 것보다 `직무설명`, `대표업무예시`, `포함키워드`, `제외키워드` 같은 보조 설명을 별도로 관리하는 편이 정확도 개선에 유리합니다. 현재 코드는 고정 taxonomy row 검증을 우선으로 두고 있습니다.
- 처음부터 전수 자동화하지 말고, 200~500건 샘플을 사람이 검수해 직무별 confusion pattern을 확인하세요.
- `confidence < 0.6` 또는 `needs_review=True`인 건은 운영 반영 전에 수동 검수하는 흐름을 권장합니다.
