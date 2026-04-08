"""Unit tests for graph routing logic — no LLM calls, no API spend."""
import pytest
from src.agent import route_after_reviewer, should_continue_research


def make_state(**overrides):
    base = {
        "theme":             "piano music technology",
        "topics":            ["topic A", "topic B", "topic C"],
        "current_index":     0,
        "current_research":  None,
        "current_draft":     None,
        "revision_count":    0,
        "articles":          [],
        "research_messages": [],
        "messages":          [],
        "output":            None,
    }
    return {**base, **overrides}


# ── route_after_reviewer ──────────────────────────────────────────────────────

def test_routes_to_researcher_after_approval():
    state = make_state(articles=["a1"], current_draft=None)
    assert route_after_reviewer(state) == "researcher"

def test_routes_to_writer_on_revision():
    state = make_state(articles=[], current_draft="REVISION: needs more detail")
    assert route_after_reviewer(state) == "writer"

def test_routes_to_compiler_when_all_done():
    state = make_state(articles=["a1", "a2", "a3"])
    assert route_after_reviewer(state) == "newsletter_compiler"

def test_routes_to_researcher_after_second_approval():
    state = make_state(articles=["a1", "a2"], current_draft=None)
    assert route_after_reviewer(state) == "researcher"


# ── should_continue_research ──────────────────────────────────────────────────

class FakeAIMessage:
    def __init__(self, has_tool_calls=False):
        self.tool_calls = [{"name": "web_search", "args": {"query": "test"}}] if has_tool_calls else []

def test_continues_research_when_tool_calls_present():
    msg   = FakeAIMessage(has_tool_calls=True)
    state = make_state(messages=[msg], research_messages=[msg])
    assert should_continue_research(state) == "tools"

def test_moves_to_writer_when_no_tool_calls():
    msg   = FakeAIMessage(has_tool_calls=False)
    state = make_state(messages=[msg], research_messages=[])
    assert should_continue_research(state) == "writer"

def test_stops_at_max_iterations():
    # 5 messages each with tool_calls = hit the cap
    msgs  = [FakeAIMessage(has_tool_calls=True)] * 5
    last  = FakeAIMessage(has_tool_calls=True)
    state = make_state(messages=[last], research_messages=msgs)
    assert should_continue_research(state) == "writer"
