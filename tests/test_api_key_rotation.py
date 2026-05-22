from types import MethodType

from llm_categorizing.classifier import ClassificationConfig, JsonlCache, OpenAICompatibleJobClassifier
from llm_categorizing.config import LLMSettings
from llm_categorizing.taxonomy import Taxonomy


def _taxonomy() -> Taxonomy:
    return Taxonomy.from_rows(
        [
            {
                "중직무": "A",
                "소직무": "A1",
                "Device": "",
                "단위 직무": "Alpha",
                "세부 직무1": "",
                "세부 직무2": "",
            }
        ]
    )


def _successful_result() -> dict[str, object]:
    return {
        "중직무": "A",
        "소직무": "A1",
        "Device": "",
        "단위 직무": "Alpha",
        "세부 직무1": "",
        "세부 직무2": "",
        "confidence": 0.91,
        "reason": "test",
        "needs_review": False,
        "ambiguity_reason": "",
        "guardrail_reason": "",
        "diagnosis_priority_reason": "",
        "knowledge_priority_reason": "",
        "error": "",
    }


def test_classifier_rotates_api_keys_between_uncached_classifications() -> None:
    classifier = OpenAICompatibleJobClassifier(
        settings=LLMSettings(
            base_url="http://localhost:1/v1",
            api_key="key-1",
            api_keys=("key-1", "key-2"),
            model="test",
        ),
        taxonomy=_taxonomy(),
        config=ClassificationConfig(),
        cache=JsonlCache(None),
    )
    seen_api_keys: list[str | None] = []

    def fixed_uncached(self, context_json, diagnosis_priority, knowledge_items):
        seen_api_keys.append(self._active_api_key.get())
        return _successful_result()

    classifier._classify_uncached = MethodType(fixed_uncached, classifier)

    first_row = {
        "year": "2025",
        "team": "",
        "emp_num": "E0001",
        "name": "A",
        "self_review": "first classification",
    }
    classifier.classify_row(first_row)
    classifier.classify_row(first_row)
    classifier.classify_row({**first_row, "self_review": "second classification"})
    classifier.classify_row({**first_row, "self_review": "third classification"})

    assert seen_api_keys == ["key-1", "key-2", "key-1"]
