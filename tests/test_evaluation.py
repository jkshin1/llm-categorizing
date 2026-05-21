import pandas as pd

from llm_categorizing.evaluation import evaluate_predictions


def test_evaluate_predictions_reports_accuracy_and_confusions() -> None:
    predictions = pd.DataFrame(
        [
            {
                "year": "2025",
                "emp_num": "E0001",
                "중직무": "공정",
                "소직무": "Etch",
                "Device": "NAND",
                "단위 직무": "Chamber",
                "세부 직무1": "Clean",
                "세부 직무2": "",
                "needs_review": "False",
            },
            {
                "year": "2025",
                "emp_num": "E0002",
                "중직무": "품질",
                "소직무": "Quality",
                "Device": "Common",
                "단위 직무": "불량분석",
                "세부 직무1": "FA",
                "세부 직무2": "",
                "needs_review": "True",
            },
        ]
    )
    gold = pd.DataFrame(
        [
            {
                "year": "2025",
                "emp_num": "E0001",
                "중직무": "공정",
                "소직무": "Etch",
                "Device": "NAND",
                "단위 직무": "Chamber",
                "세부 직무1": "Clean",
                "세부 직무2": "",
            },
            {
                "year": "2025",
                "emp_num": "E0002",
                "중직무": "품질",
                "소직무": "Quality",
                "Device": "Common",
                "단위 직무": "불량분석",
                "세부 직무1": "Root Cause",
                "세부 직무2": "",
            },
        ]
    )

    result = evaluate_predictions(predictions, gold)

    assert result["matched_count"] == 2
    assert result["exact_path_accuracy"] == 0.5
    assert result["pair_accuracy"] == 1.0
    assert result["needs_review_rate"] == 0.5
    assert result["top_confusions"] == [
        {
            "expected": "품질 > Quality > Common > 불량분석 > Root Cause",
            "predicted": "품질 > Quality > Common > 불량분석 > FA",
            "count": 1,
        }
    ]
