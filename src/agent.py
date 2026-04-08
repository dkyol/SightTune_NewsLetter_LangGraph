import ast
from datetime import date
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from src.state import NewsletterState
from src.tools import RESEARCH_TOOLS
from src.email_template import build_email, load_logo_b64

MAX_RESEARCH_ITERATIONS = 5
MAX_REVISIONS = 2

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
researcher_llm = llm.bind_tools(RESEARCH_TOOLS)
_tool_node = ToolNode(RESEARCH_TOOLS)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def topic_planner_node(state: NewsletterState) -> NewsletterState:
    today = date.today().strftime("%B %d, %Y")
    response = llm.invoke([
        SystemMessage(content=(
            f"You are a newsletter editor. Today's date is {today}.\n"
            "Given a theme, generate exactly 3 distinct topics that have happened "
            "or are actively developing within the LAST 60 DAYS. "
            "Do not suggest anything older than 60 days. "
            "Mix types: a recent event, a brand/store spotlight, and a person to profile.\n"
            "Return a Python list of 3 strings only — no explanation, no numbering, just the list."
        )),
        HumanMessage(content=f"Theme: {state['theme']}")
    ])
    try:
        topics = ast.literal_eval(response.content.strip())
    except Exception:
        topics = [l.strip("- *1234567890.") for l in response.content.strip().splitlines() if l.strip()][:3]
    print(f"Topics: {topics}")
    return {
        **state,
        "topics":            topics,
        "current_index":     0,
        "revision_count":    0,
        "articles":          [],
        "research_messages": [],
        "messages":          [response],
    }


def researcher_node(state: NewsletterState) -> NewsletterState:
    topic      = state["topics"][state["current_index"]]
    today      = date.today().strftime("%B %d, %Y")
    prior_msgs = state.get("research_messages", [])
    iteration  = sum(1 for m in prior_msgs if hasattr(m, "tool_calls") and m.tool_calls) + 1
    print(f"  Researching [{state['current_index']+1}/3]: {topic!r}  (iter {iteration})")

    system_msg = SystemMessage(content=(
        f"You are a research agent for a newsletter. Today's date is {today}.\n"
        "Your research MUST focus on news and developments from the last 30 days only.\n"
        "When forming search queries, include the current month and year.\n"
        "Tool priority:\n"
        "  1. web_search   — use FIRST (fastest, cheapest)\n"
        "  2. tavily_search — only if web_search results are incomplete\n"
        "  3. deep_research — only for complex multi-step topics\n"
        "Return a concise summary with key facts, quotes, sources, and the date of each fact."
    ))
    human_msg = HumanMessage(content=f"Theme: {state['theme']}\nTopic: {topic}")
    response = researcher_llm.invoke([system_msg, human_msg] + prior_msgs)

    return {
        **state,
        "current_research":  response.content,
        "research_messages": prior_msgs + [response],
        "messages":          [response],
    }


def research_tool_node(state: NewsletterState) -> NewsletterState:
    result    = _tool_node.invoke(state)
    tool_msgs = result.get("messages", [])
    print(f"    Tool returned {len(tool_msgs)} result(s)")
    return {
        **state,
        "messages":          tool_msgs,
        "research_messages": state.get("research_messages", []) + tool_msgs,
    }


def writer_node(state: NewsletterState) -> NewsletterState:
    topic        = state["topics"][state["current_index"]]
    draft_context = state.get("current_draft") or ""
    feedback = (
        f"\n\nReviewer feedback to address:\n{draft_context}"
        if draft_context.startswith("REVISION:") else ""
    )
    print(f"  Writing article for: {topic!r}")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a newsletter writer. Write a short, engaging article (150-200 words) "
            "based on the research. Use a clear headline, 2-3 paragraphs, and a punchy "
            "closing sentence. No bullet points."
        )),
        HumanMessage(content=f"Topic: {topic}\n\nResearch:\n{state['current_research']}{feedback}")
    ])
    return {**state, "current_draft": response.content, "messages": [response]}


def reviewer_node(state: NewsletterState) -> NewsletterState:
    topic = state["topics"][state["current_index"]]
    print(f"  Reviewing article for: {topic!r}  (revision #{state['revision_count']})")

    if state["revision_count"] >= MAX_REVISIONS:
        print("    Max revisions reached — force approving")
        updated = state["articles"] + [state["current_draft"]]
        return {
            **state,
            "articles":          updated,
            "current_draft":     None,
            "current_research":  None,
            "current_index":     state["current_index"] + 1,
            "revision_count":    0,
            "research_messages": [],
            "messages":          [],
        }

    response = llm.invoke([
        SystemMessage(content=(
            "You are a newsletter editor. Review the article.\n"
            "If it is acceptable (on-topic, engaging, roughly 150-200 words), "
            "reply with exactly the single word: APPROVED\n"
            "If it needs a specific fix, reply with: REVISION: <one sentence of feedback>\n"
            "Do not add any other text to your response."
        )),
        HumanMessage(content=f"Topic: {topic}\n\nArticle:\n{state['current_draft']}")
    ])
    verdict    = response.content.strip()
    is_approved = verdict.upper() == "APPROVED" or verdict.upper().startswith("APPROVED")
    print(f"    Verdict: {'APPROVED' if is_approved else verdict[:60]}")

    if is_approved:
        updated = state["articles"] + [state["current_draft"]]
        print(f"    Articles done: {len(updated)}/3")
        return {
            **state,
            "articles":          updated,
            "current_draft":     None,
            "current_research":  None,
            "current_index":     state["current_index"] + 1,
            "revision_count":    0,
            "research_messages": [],
            "messages":          [response],
        }
    return {**state, "current_draft": verdict, "revision_count": state["revision_count"] + 1, "messages": [response]}


def newsletter_compiler_node(state: NewsletterState) -> NewsletterState:
    today    = date.today().strftime("%B %d, %Y")
    logo_b64 = load_logo_b64()
    html     = build_email(state["theme"], state["articles"], today, logo_b64)
    response = AIMessage(content=html)
    print("Newsletter compiled!")
    return {**state, "output": html, "messages": [response]}


# ── Routing ───────────────────────────────────────────────────────────────────

def should_continue_research(state: NewsletterState) -> str:
    last_msg        = state["messages"][-1]
    prior_msgs      = state.get("research_messages", [])
    tool_calls_made = sum(1 for m in prior_msgs if hasattr(m, "tool_calls") and m.tool_calls)
    if tool_calls_made >= MAX_RESEARCH_ITERATIONS:
        print(f"  Max research iterations reached — moving to writer")
        return "writer"
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "writer"


def route_after_reviewer(state: NewsletterState) -> str:
    if len(state["articles"]) < 3:
        if (state.get("current_draft") or "").startswith("REVISION:"):
            return "writer"
        return "researcher"
    return "newsletter_compiler"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(NewsletterState)

    builder.add_node("topic_planner",       topic_planner_node)
    builder.add_node("researcher",          researcher_node)
    builder.add_node("research_tools",      research_tool_node)
    builder.add_node("writer",              writer_node)
    builder.add_node("reviewer",            reviewer_node)
    builder.add_node("newsletter_compiler", newsletter_compiler_node)

    builder.set_entry_point("topic_planner")
    builder.add_edge("topic_planner", "researcher")
    builder.add_conditional_edges("researcher", should_continue_research, {
        "tools":  "research_tools",
        "writer": "writer",
    })
    builder.add_edge("research_tools", "researcher")
    builder.add_edge("writer",         "reviewer")
    builder.add_conditional_edges("reviewer", route_after_reviewer, {
        "researcher":           "researcher",
        "writer":               "writer",
        "newsletter_compiler":  "newsletter_compiler",
    })
    builder.add_edge("newsletter_compiler", END)

    return builder.compile(checkpointer=MemorySaver())
