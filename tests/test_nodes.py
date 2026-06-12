"""Unit tests for individual nodes — LLM calls are mocked."""
from unittest.mock import patch

from langchain_core.messages import AIMessage

from src.agent import _screen_relevance, reviewer_node, topic_planner_node


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
def test_reviewer_drops_article_at_max_revisions(mock_llm):
    # Nothing is force-approved: at the revision limit the article is dropped, not shipped.
    state  = make_state(revision_count=3)
    result = reviewer_node(state)
    mock_llm.invoke.assert_not_called()        # no LLM call needed to drop
    assert len(result["articles"]) == 0        # article NOT added
    assert result["current_index"] == 1        # advanced to next topic
    assert result["revision_count"] == 0

@patch("src.agent.llm")
def test_reviewer_handles_approved_with_punctuation(mock_llm):
    mock_llm.invoke.return_value = AIMessage(content="APPROVED.")
    state  = make_state()
    result = reviewer_node(state)
    assert len(result["articles"]) == 1


# ── _screen_relevance (relevance gate) ────────────────────────────────────────

@patch("src.agent.llm")
def test_relevance_keeps_only_on_theme(mock_llm):
    mock_llm.invoke.return_value = AIMessage(content="[1]")
    result = _screen_relevance(
        "piano music technology",
        ["On-theme piano AI learning tool", "Generic data-marketing consultancy"],
    )
    assert result == ["On-theme piano AI learning tool"]

@patch("src.agent.llm")
def test_relevance_fails_open_on_bad_response(mock_llm):
    mock_llm.invoke.return_value = AIMessage(content="not a list")
    topics = ["topic A", "topic B"]
    assert _screen_relevance("piano music technology", topics) == topics

@patch("src.agent.llm")
def test_relevance_fails_open_when_all_rejected(mock_llm):
    # A model that rejects everything must not empty the issue.
    mock_llm.invoke.return_value = AIMessage(content="[]")
    topics = ["topic A", "topic B"]
    assert _screen_relevance("piano music technology", topics) == topics


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
