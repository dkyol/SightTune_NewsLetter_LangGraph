"""Unit tests for individual nodes — LLM calls are mocked."""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from src.agent import reviewer_node, topic_planner_node


def make_state(**overrides):
    base = {
        "theme":             "piano music technology",
        "topics":            ["topic A", "topic B", "topic C"],
        "current_index":     0,
        "current_research":  "Some research content",
        "current_draft":     "Draft article text here.",
        "revision_count":    0,
        "articles":          [],
        "research_messages": [],
        "messages":          [],
        "output":            None,
    }
    return {**base, **overrides}


# ── reviewer_node ─────────────────────────────────────────────────────────────

@patch("src.agent.llm")
def test_reviewer_approves(mock_llm):
    mock_llm.invoke.return_value = AIMessage(content="APPROVED")
    state  = make_state()
    result = reviewer_node(state)
    assert len(result["articles"]) == 1
    assert result["current_draft"] is None
    assert result["current_index"] == 1
    assert result["revision_count"] == 0

@patch("src.agent.llm")
def test_reviewer_requests_revision(mock_llm):
    mock_llm.invoke.return_value = AIMessage(content="REVISION: needs a stronger opening")
    state  = make_state()
    result = reviewer_node(state)
    assert len(result["articles"]) == 0
    assert result["revision_count"] == 1
    assert result["current_draft"].startswith("REVISION:")

@patch("src.agent.llm")
def test_reviewer_force_approves_at_max_revisions(mock_llm):
    state  = make_state(revision_count=2)
    result = reviewer_node(state)
    # Should approve without calling LLM
    mock_llm.invoke.assert_not_called()
    assert len(result["articles"]) == 1
    assert result["revision_count"] == 0

@patch("src.agent.llm")
def test_reviewer_handles_approved_with_punctuation(mock_llm):
    mock_llm.invoke.return_value = AIMessage(content="APPROVED.")
    state  = make_state()
    result = reviewer_node(state)
    assert len(result["articles"]) == 1


# ── topic_planner_node ────────────────────────────────────────────────────────

@patch("src.agent.llm")
def test_topic_planner_parses_list(mock_llm):
    mock_llm.invoke.return_value = AIMessage(
        content='["Event A", "Store B spotlight", "Person C profile"]'
    )
    state  = make_state(topics=[], articles=[], current_index=0)
    result = topic_planner_node(state)
    assert result["topics"] == ["Event A", "Store B spotlight", "Person C profile"]
    assert result["current_index"] == 0
    assert result["articles"] == []

@patch("src.agent.llm")
def test_topic_planner_fallback_parsing(mock_llm):
    mock_llm.invoke.return_value = AIMessage(content="1. Event A\n2. Store B\n3. Person C")
    state  = make_state(topics=[], articles=[])
    result = topic_planner_node(state)
    assert len(result["topics"]) == 3
