import ast
import json
from datetime import date, timedelta
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.email_template import build_email, load_logo_b64
from src.state import NewsletterState
from src.tools import RESEARCH_TOOLS, run_topic_search

MAX_RESEARCH_ITERATIONS = 5
MAX_REVISIONS = 2

LOGS_DIR = Path(__file__).parent.parent / "logs"

ANGLE_TEMPLATES = [
    "{theme} emerging news {month} {year}",
    "{theme} niche community spotlight {year}",
    "{theme} performer profile interview {year}",
    "{theme} new technology research {year}",
    "{theme} independent business startup {year}",
]


def _load_past_topics(months: int = 6) -> list[str]:
    """Return topics covered in the last N months from history.jsonl."""
    history_path = LOGS_DIR / "history.jsonl"
    if not history_path.exists():
        return []
    cutoff = date.today() - timedelta(days=months * 30)
    topics = []
    with open(history_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                entry_date = date.fromisoformat(entry.get("date", ""))
                if entry_date >= cutoff:
                    topics.extend(entry.get("topics", []))
            except (json.JSONDecodeError, ValueError):
                continue
    return topics


def _generate_search_queries(theme: str, today: str, history_context: str) -> list[str]:
    """Ask the LLM to generate diverse, angle-driven search queries. Falls back to templates."""
    month, year = today.split(" ")[0], today.split(" ")[-1]
    try:
        response = llm.invoke([
            SystemMessage(content=(
                "You are an editorial researcher. Generate 4 search queries that would surface "
                "underreported, niche, or emerging stories — not the dominant mainstream headlines.\n"
                "Each query should target a different angle: emerging technology, community/grassroots, "
                "person profile, and independent business.\n"
                f"{history_context}\n"
                "Return a Python list of 4 query strings only — no explanation."
            )),
            HumanMessage(content=f"Theme: {theme}\nCurrent month: {month} {year}"),
        ])
        queries = ast.literal_eval(_strip_code_fence(response.content))
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:4]
    except Exception:
        pass
    # Fallback to hardcoded templates
    return [
        t.format(theme=theme, month=month, year=year)
        for t in ANGLE_TEMPLATES[:4]
    ]


def _parse_research(content: str) -> tuple[str, list[str]]:
    """Split researcher output into (summary, sources). Falls back to (full text, []) if unparseable."""
    if "SOURCES:" not in content:
        return content.strip(), []
    parts = content.split("SOURCES:", 1)
    summary = parts[0].replace("SUMMARY:", "").strip()
    source_lines = [
        line.strip().lstrip("- ").strip()
        for line in parts[1].strip().splitlines()
        if line.strip() and line.strip() != "-"
    ]
    return summary, source_lines


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences (```python ... ```) if present."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _select_topics(theme: str, today: str, search_results: str, history_context: str) -> list[str]:
    """Ask the LLM to pick 3 novel topics from combined search results."""
    response = llm.invoke([
        SystemMessage(content=(
            f"You are a newsletter editor. Today's date is {today}.\n"
            "From the search results below, select exactly 3 distinct, interesting topics "
            "that are underreported or novel — avoid the biggest mainstream headlines.\n"
            "Mix types: one emerging technology or trend, one person or brand spotlight, "
            "one community or grassroots story.\n"
            f"{history_context}\n"
            "Return a Python list of 3 strings only — no markdown, no code fences, no explanation."
        )),
        HumanMessage(content=f"Theme: {theme}\n\nSearch results:\n{search_results}"),
    ])
    content = _strip_code_fence(response.content)
    try:
        topics = ast.literal_eval(content)
        if isinstance(topics, list) and len(topics) >= 3:
            return topics[:3]
    except Exception:
        pass
    return [line.strip("- *1234567890.") for line in content.splitlines() if line.strip()][:3]

llm            = None
researcher_llm = None
_tool_node     = None


def _init_llm():
    """Initialise LLM clients — called from build_graph() so env vars are loaded first."""
    global llm, researcher_llm, _tool_node
    llm            = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    researcher_llm = llm.bind_tools(RESEARCH_TOOLS)
    _tool_node     = ToolNode(RESEARCH_TOOLS)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def topic_planner_node(state: NewsletterState) -> NewsletterState:
    today = date.today().strftime("%B %d, %Y")
    theme = state["theme"]

    # 1. Load past topics to avoid repetition
    past_topics = _load_past_topics(months=6)
    history_context = (
        "Previously covered topics (do not repeat similar ones):\n"
        + "\n".join(f"- {t}" for t in past_topics)
        if past_topics else ""
    )

    # 2. LLM generates angle-driven search queries (falls back to templates)
    queries = _generate_search_queries(theme, today, history_context)
    print(f"Topic search queries: {queries}")

    # 3. Run searches and combine results
    combined_results = []
    for query in queries:
        try:
            result = run_topic_search(query)
            combined_results.append(f"Query: {query}\n{result}")
        except Exception as e:
            print(f"  Search failed for {query!r}: {e}")
    search_results = "\n\n---\n\n".join(combined_results)

    # 4. LLM selects 3 novel topics from the real search results
    topics = _select_topics(theme, today, search_results, history_context)
    print(f"Topics selected: {topics}")

    return {
        **state,
        "topics":               topics,
        "topic_search_results": search_results,
        "current_index":        0,
        "revision_count":       0,
        "articles":             [],
        "research_messages":    [],
        "messages":             [],
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
        "Return your response in exactly this format:\n"
        "SUMMARY:\n"
        "<concise prose summary with key facts, quotes, and dates>\n\n"
        "SOURCES:\n"
        "- <url> | <one-line description>\n"
        "- <url> | <one-line description>\n"
        "Include every URL that contributed a meaningful fact or quote."
    ))
    human_msg = HumanMessage(content=f"Theme: {state['theme']}\nTopic: {topic}")
    response = researcher_llm.invoke([system_msg, human_msg] + prior_msgs)

    summary, sources = _parse_research(response.content)
    print(f"    Sources found: {len(sources)}")

    return {
        **state,
        "current_research":  summary,
        "current_sources":   sources,
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
    topic         = state["topics"][state["current_index"]]
    draft_context = state.get("current_draft") or ""
    feedback = (
        f"\n\nReviewer feedback to address:\n{draft_context}"
        if draft_context.startswith("REVISION:") else ""
    )
    sources       = state.get("current_sources") or []
    sources_block = "\n".join(f"- {s}" for s in sources)
    print(f"  Writing article for: {topic!r}  ({len(sources)} sources)")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a newsletter writer. Write a short, engaging article (150-200 words) "
            "based on the research. Use a clear headline, 2-3 paragraphs, and a punchy "
            "closing sentence. No bullet points.\n"
            "You MUST reference at least one specific source URL inline (e.g. 'according to [Publication](url)')."
        )),
        HumanMessage(content=(
            f"Topic: {topic}\n\n"
            f"Research:\n{state['current_research']}\n\n"
            f"Sources:\n{sources_block}"
            f"{feedback}"
        ))
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
            "current_sources":   None,
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
            "current_sources":   None,
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
        print("  Max research iterations reached — moving to writer")
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
    _init_llm()
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
