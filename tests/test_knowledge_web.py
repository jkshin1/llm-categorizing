import pytest

from llm_categorizing.knowledge import JobKnowledgeStore
from llm_categorizing.knowledge_web import import_text_chunks, split_import_text


def test_split_import_text_uses_non_blank_lines_as_chunks() -> None:
    chunks = split_import_text(
        """
        Heraion은 NAND 제품 프로젝트를 의미한다.

        TD는 중직무 소자 후보를 우선 검토한다.
        """
    )

    assert chunks == [
        "Heraion은 NAND 제품 프로젝트를 의미한다.",
        "TD는 중직무 소자 후보를 우선 검토한다.",
    ]


def test_split_import_text_enforces_line_limits() -> None:
    with pytest.raises(ValueError, match="too long"):
        split_import_text("A" * 11, max_chunk_chars=10)

    with pytest.raises(ValueError, match="too many"):
        split_import_text("a\nb\nc", max_chunks=2)


def test_import_text_chunks_stores_each_line_as_separate_entry(tmp_path) -> None:
    store = JobKnowledgeStore(tmp_path / "knowledge.sqlite3")

    result = import_text_chunks(
        store,
        "Heraion은 NAND 제품 프로젝트를 의미한다.\nTD는 중직무 소자 후보를 우선 검토한다.",
        normalizer=None,
        taxonomy=None,
        use_llm=False,
        allow_fallback_normalizer=True,
    )

    assert result["created_count"] == 2
    assert result["chunk_count"] == 2
    assert result["errors"] == []
    assert [item["source"] for item in result["items"]] == ["txt_import", "txt_import"]
    assert len(store.list_recent(limit=10)) == 2
