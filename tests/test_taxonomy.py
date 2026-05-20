from llm_categorizing.taxonomy import Taxonomy


def test_taxonomy_deduplicates_and_canonicalizes_case() -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "소자",
                "소직무": "Process Integration",
                "Device": "Memory",
                "단위 직무": "공정통합",
                "세부 직무1": "DRAM PI",
                "세부 직무2": "Cell",
            },
            {
                "중직무": "소자",
                "소직무": "Process Integration",
                "Device": "Memory",
                "단위 직무": "공정통합",
                "세부 직무1": "DRAM PI",
                "세부 직무2": "Cell",
            },
        ]
    )

    assert len(taxonomy.rows) == 1
    assert taxonomy.canonical_pair("소자", "process integration") == {
        "중직무": "소자",
        "소직무": "Process Integration",
    }


def test_canonical_path_returns_original_taxonomy_values() -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "소자",
                "소직무": "Device",
                "Device": "Logic",
                "단위 직무": "소자개발",
                "세부 직무1": "Device Characterization",
                "세부 직무2": "Reliability",
            }
        ]
    )

    result = taxonomy.canonical_path(
        {
            "중직무": "소자",
            "소직무": "device",
            "Device": "logic",
            "단위 직무": "소자개발",
            "세부 직무1": "device characterization",
            "세부 직무2": "reliability",
        }
    )

    assert result == taxonomy.rows[0]


def test_taxonomy_can_query_duplicate_unit_job_rows_without_adding_classification_rules() -> None:
    taxonomy = Taxonomy.from_rows(
        [
            {
                "중직무": "소자",
                "소직무": "Process Integration",
                "Device": "DRAM",
                "단위 직무": "MLM",
                "세부 직무1": "Device",
                "세부 직무2": "",
            },
            {
                "중직무": "공정",
                "소직무": "Module",
                "Device": "DRAM",
                "단위 직무": "MLM",
                "세부 직무1": "Process",
                "세부 직무2": "",
            },
        ]
    )

    assert taxonomy.major_jobs_for_unit_job("mlm") == ["소자", "공정"]
    assert taxonomy.rows_for_unit_job("mlm", major_job="공정", device="dram") == [taxonomy.rows[1]]
