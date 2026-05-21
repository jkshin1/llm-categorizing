import json
from pathlib import Path

from llm_categorizing.diagnosis import DiagnosisContext, diagnosis_key, load_diagnosis_contexts


def test_load_diagnosis_contexts_groups_multiple_rows(tmp_path: Path) -> None:
    path = tmp_path / "diagnosis.csv"
    path.write_text(
        "\n".join(
            [
                "year,emp_num,name,team,진단 시 직무명,Category,항목",
                "2025,E0001,홍길동,DRAM공정 > DPC,Etch공정,CLN,Chamber clean",
                "2025,E0001,홍길동,DRAM공정 > DPC,Etch공정,MLM,Via scheme",
            ]
        ),
        encoding="utf-8-sig",
    )

    contexts = load_diagnosis_contexts(path)
    context = contexts[diagnosis_key("2025", "E0001")]

    assert context.row_count == 2
    assert context.teams == ["DRAM공정 > DPC"]
    assert context.job_names == ["Etch공정"]
    assert context.categories == ["CLN", "MLM"]
    assert len(context.evidence_rows) == 2

    prompt_payload = context.to_prompt_payload()
    serialized_prompt_payload = json.dumps(prompt_payload, ensure_ascii=False)

    assert "items" not in prompt_payload
    assert "item" not in serialized_prompt_payload
    assert "Chamber clean" not in serialized_prompt_payload
    assert "Via scheme" not in serialized_prompt_payload


def test_diagnosis_prompt_payload_sanitizes_manual_evidence_items() -> None:
    context = DiagnosisContext(
        year="2025",
        emp_num="E0001",
        row_count=1,
        teams=["TeamA"],
        job_names=["JobA"],
        categories=["CategoryA"],
        evidence_rows=[
            {
                "diagnosis_team": "TeamA",
                "diagnosis_job_name": "JobA",
                "category": "CategoryA",
                "item": "PromptLeakItem",
            }
        ],
    )

    payload = context.to_prompt_payload()
    serialized_payload = json.dumps(payload, ensure_ascii=False)

    assert payload["evidence_rows"] == [
        {
            "diagnosis_team": "TeamA",
            "diagnosis_job_name": "JobA",
            "category": "CategoryA",
        }
    ]
    assert "PromptLeakItem" not in serialized_payload
