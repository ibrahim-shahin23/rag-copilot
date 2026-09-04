from application.tools import DraftItemTool
from domain.ports import LLMProvider


class _AlwaysFailsJSON(LLMProvider):
    """Simulates the extractive-fallback provider: returns non-JSON text,
    so DraftItemTool must fall through to the deterministic drafter."""
    name = "fake-non-json-llm"

    def complete(self, system_prompt, user_prompt):
        return "This is not JSON, just an excerpt echo."


class _ReturnsValidJSON(LLMProvider):
    name = "fake-json-llm"

    def complete(self, system_prompt, user_prompt):
        return '{"question": "Q?", "options": ["a", "b"], "correct_index": 0}'


class _ReturnsMalformedJSON(LLMProvider):
    name = "fake-malformed-llm"

    def complete(self, system_prompt, user_prompt):
        return '{"question": "Q?", "options": []}'  # missing correct_index, empty options


def test_llm_path_used_when_llm_returns_valid_json():
    tool = DraftItemTool(llm=_ReturnsValidJSON())
    result = tool("FR-1 requires at least two input formats.", "spec.md")
    assert result == {"question": "Q?", "options": ["a", "b"], "correct_index": 0}


def test_falls_back_to_deterministic_drafter_on_non_json_llm_output():
    tool = DraftItemTool(llm=_AlwaysFailsJSON())
    result = tool("Ingestion requires at least 2 input formats.", "spec.md")
    assert result is not None
    assert "____" in result["question"]
    assert "2" in result["options"]
    correct = result["options"][result["correct_index"]]
    assert correct == "2"


def test_falls_back_to_deterministic_drafter_on_malformed_json():
    tool = DraftItemTool(llm=_ReturnsMalformedJSON())
    result = tool("At least 25 Q/A pairs are required.", "spec.md")
    assert result is not None
    assert result["options"][result["correct_index"]] == "25"


def test_deterministic_drafter_returns_none_when_no_number_present():
    tool = DraftItemTool(llm=_AlwaysFailsJSON())
    result = tool("This sentence has no digits in it at all.", "spec.md")
    assert result is None


def test_deterministic_drafter_options_are_distinct():
    tool = DraftItemTool(llm=_AlwaysFailsJSON())
    result = tool("FR-8 requires at least two roles.", "spec.md")
    assert len(set(result["options"])) == len(result["options"])