from pathlib import Path

from llm_categorizing.diagnosis import diagnosis_key, load_diagnosis_contexts


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
