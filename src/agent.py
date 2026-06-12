import ast
import json
import re
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
MAX_REVISIONS = 3
EVIDENCE_CHAR_CAP = 4000  # cap raw evidence passed to writer/reviewer to control tokens

LOGS_DIR = Path(__file__).parent.parent / "logs"

_URL_RE = re.compile(r"https?://[^\s'\"<>)\]}]+")


def _extract_urls(text: str) -> list[str]:
    """Pull http(s) URLs out of raw tool output, trimming trailing punctuation."""
    return [u.rstrip(".,;") for u in _URL_RE.findall(text or "")]


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


# Guardrail #1: reject fabricated / placeholder source URLs.
_PLACEHOLDER_HOSTS = (
    "example.com", "example.org", "example.net", "example.edu",
    "localhost", "test.com", "yoursite", "yourdomain", "url.com", "domain.com",
)


def _is_valid_source_url(url: str) -> bool:
    """True only for real-looking http(s) URLs — drops example.com and placeholders."""
    u = (url or "").strip().lower()
    if not u.startswith(("http://", "https://")):
        return False
    if any(host in u for host in _PLACEHOLDER_HOSTS):
        return False
    return "." in u.split("//", 1)[-1]  # must have a dotted host


def _filter_urls(urls: list[str]) -> list[str]:
    return [u for u in urls if _is_valid_source_url(u)]


# Guardrail #2: keep topics on-brand and exclude reputationally risky subjects.
_UNSAFE_TOPIC_TERMS = (
    "allegation", "misconduct", "sexual", "assault", "harassment", "abuse",
    "lawsuit", "sued", "arrested", "indicted", "convicted", "fraud", "scandal",
    "controversy", "crime", "criminal", "death", "died", "obituary",
    "racist", "racism", "politic", "election",
)


def _is_safe_topic(topic: str) -> bool:
    """Drop topics that are off-brand or carry defamation / negative-news risk."""
    t = (topic or "").lower()
    return not any(term in t for term in _UNSAFE_TOPIC_TERMS)


def _fallback_topics(theme: str) -> list[str]:
    """Safe, on-theme, claim-free topics used to backfill if screening removes some."""
    return [
        f"How emerging {theme} tools are changing the way people learn and practice",
        f"Broader trends shaping {theme} and what they mean for everyday musicians",
        f"Why accessibility and community matter in the world of {theme}",
    ]

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


def _screen_relevance(theme: str, topics: list[str]) -> list[str]:
    """Keep only topics genuinely and primarily about the theme — drops forced/tangential ones.

    Fails open: if the screen errors or returns nothing usable, all topics are kept so a
    bad LLM response can't empty the issue.
    """
    if not topics:
        return []
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(topics, 1))
    try:
        response = llm.invoke([
            SystemMessage(content=(
                f"You are an editor for a publication strictly about {theme}.\n"
                "For each numbered candidate, decide if it is GENUINELY and PRIMARILY about "
                f"{theme}. Reject a topic when:\n"
                f"  - its connection to {theme} is forced, tangential, or superficial;\n"
                "  - the subject is a general company, product, or service not specifically "
                f"about {theme} (e.g. a generic marketing, data, consulting, or software firm);\n"
                f"  - it only mentions {theme} in passing.\n"
                "Return a Python list of the integers that genuinely qualify — e.g. [1, 3]. "
                "Return only the list, nothing else."
            )),
            HumanMessage(content=f"Theme: {theme}\n\nCandidates:\n{numbered}"),
        ])
        keep = ast.literal_eval(_strip_code_fence(response.content))
        if isinstance(keep, list):
            idxs = {int(n) for n in keep if isinstance(n, (int, float)) and not isinstance(n, bool)}
            relevant = [t for i, t in enumerate(topics, 1) if i in idxs]
            return relevant or topics  # fail open if the model rejected everything
    except Exception:
        pass
    return topics


def _select_topics(theme: str, today: str, search_results: str, history_context: str) -> list[str]:
    """Pick 3 on-brand, safe, on-theme topics from the search results, backfilling if needed."""
    response = llm.invoke([
        SystemMessage(content=(
            f"You are a newsletter editor for a {theme} publication. Today's date is {today}.\n"
            "From the search results below, select 5 distinct, interesting candidate topics "
            "that are underreported or novel — avoid the biggest mainstream headlines.\n"
            "Mix types: emerging technology/trend, a positive person or brand spotlight, "
            "and a community or grassroots story.\n"
            "STRICT RULES:\n"
            f"  - Every topic MUST be directly relevant to {theme}.\n"
            "  - Choose POSITIVE, constructive stories only.\n"
            "  - EXCLUDE anything involving allegations, controversy, lawsuits, crime, "
            "misconduct, scandal, or negative news about a named individual; politics; "
            "or subjects unrelated to the theme.\n"
            f"{history_context}\n"
            "Return a Python list of 5 strings only — no markdown, no code fences, no explanation."
        )),
        HumanMessage(content=f"Theme: {theme}\n\nSearch results:\n{search_results}"),
    ])
    content = _strip_code_fence(response.content)
    try:
        candidates = ast.literal_eval(content)
        if not (isinstance(candidates, list) and candidates):
            raise ValueError
    except Exception:
        candidates = [line.strip("- *1234567890.") for line in content.splitlines() if line.strip()]

    # Screen out unsafe/off-brand topics (keyword denylist), then off-theme ones (LLM relevance
    # gate), then backfill with safe generic topics to keep exactly 3.
    safe = _dedupe([t for t in candidates if isinstance(t, str) and _is_safe_topic(t)])
    if len(candidates) - len(safe):
        print(f"  Topic safety filter removed {len(candidates) - len(safe)} candidate(s)")

    relevant = _screen_relevance(theme, safe)
    if len(safe) - len(relevant):
        print(f"  Topic relevance filter removed {len(safe) - len(relevant)} off-theme candidate(s)")

    final = list(relevant)
    if len(final) < 3:
        final += [t for t in _fallback_topics(theme) if t not in final]
    return final[:3]

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

    summary, parsed_sources = _parse_research(response.content)
    # Merge model-reported sources with URLs already captured from tool output (Fix #3),
    # filtering out any placeholder/fake URLs (Guardrail #1).
    merged_sources = _dedupe((state.get("current_sources") or []) + _filter_urls(parsed_sources))
    print(f"    Sources found: {len(merged_sources)}")

    return {
        **state,
        "current_research":  summary,
        "current_sources":   merged_sources,
        "research_messages": prior_msgs + [response],
        "messages":          [response],
    }


def research_tool_node(state: NewsletterState) -> NewsletterState:
    result    = _tool_node.invoke(state)
    tool_msgs = result.get("messages", [])

    # Capture real source URLs and raw evidence directly from the tool output, rather
    # than trusting the model to reformat them into a SOURCES: block (Fix #3).
    tool_text   = "\n\n".join(str(getattr(m, "content", "")) for m in tool_msgs)
    new_urls    = _filter_urls(_extract_urls(tool_text))  # drop placeholder/fake URLs (Guardrail #1)
    all_sources = _dedupe((state.get("current_sources") or []) + new_urls)
    evidence    = ((state.get("current_evidence") or "") + "\n\n" + tool_text).strip()

    print(f"    Tool returned {len(tool_msgs)} result(s); {len(new_urls)} URL(s) extracted")
    return {
        **state,
        "current_sources":   all_sources,
        "current_evidence":  evidence,
        "messages":          tool_msgs,
        "research_messages": state.get("research_messages", []) + tool_msgs,
    }


def writer_node(state: NewsletterState) -> NewsletterState:
    topic         = state["topics"][state["current_index"]]
    today         = date.today().strftime("%B %d, %Y")
    draft_context = state.get("current_draft") or ""
    feedback = (
        f"\n\nReviewer feedback to address (fix this specifically):\n{draft_context}"
        if draft_context.startswith("REVISION:") else ""
    )
    sources       = state.get("current_sources") or []
    sources_block = "\n".join(f"- {s}" for s in sources) or "(none)"
    evidence      = (state.get("current_evidence") or "")[:EVIDENCE_CHAR_CAP]
    print(f"  Writing article for: {topic!r}  ({len(sources)} sources)")

    # Grounding rules shared by both modes (Fix #2).
    grounding = (
        f"Today's date is {today}.\n"
        "STRICT GROUNDING RULES:\n"
        "- Use ONLY facts, dates, names, numbers, and quotes that appear in the Research "
        "or Evidence below. Do NOT add details from your own prior knowledge.\n"
        "- Never state a specific date, deadline, statistic, or proper name unless it appears "
        "verbatim in the Research/Evidence.\n"
        "- If a specific detail is missing, write around it — do not guess or invent one.\n"
        f"- All dates must be consistent with today being {today}; do not present a past date "
        "as an upcoming or current deadline."
    )

    if sources:
        style = (
            "You are a newsletter writer. Write a short, engaging article (150-200 words) "
            "based strictly on the research. Clear headline, 2-3 paragraphs, punchy closing. "
            "No bullet points.\n"
            "If you cite a source URL inline, you MUST copy it EXACTLY from the Sources list "
            "below — never invent, guess, or alter a URL, and never use example.com or any "
            "placeholder. If no source fits, write without an inline link.\n\n"
            + grounding
        )
    else:
        # Fix #4: no sources captured — forbid specific claims entirely.
        style = (
            "You are a newsletter writer. No sources were found for this topic, so write a "
            "short, engaging GENERAL reflection (120-160 words) on the topic's broader theme. "
            "Clear headline, 2 short paragraphs, punchy closing. No bullet points.\n"
            "Because there are no sources, you MUST NOT include any specific dates, statistics, "
            "named people/companies, product launch claims, or quotes. Keep it general and "
            "clearly non-newsy.\n\n"
            + grounding
        )

    response = llm.invoke([
        SystemMessage(content=style),
        HumanMessage(content=(
            f"Topic: {topic}\n\n"
            f"Research:\n{state['current_research']}\n\n"
            f"Evidence (raw source excerpts):\n{evidence or '(none)'}\n\n"
            f"Sources:\n{sources_block}"
            f"{feedback}"
        ))
    ])
    return {**state, "current_draft": response.content, "messages": [response]}


def _advance_topic(state: NewsletterState, approved_draft: str | None) -> NewsletterState:
    """Move to the next topic, optionally appending an approved article. Resets per-topic state."""
    articles = state["articles"] + [approved_draft] if approved_draft is not None else state["articles"]
    return {
        **state,
        "articles":          articles,
        "current_draft":     None,
        "current_research":  None,
        "current_sources":   None,
        "current_evidence":  None,
        "current_index":     state["current_index"] + 1,
        "revision_count":    0,
        "research_messages": [],
        "messages":          [],
    }


def reviewer_node(state: NewsletterState) -> NewsletterState:
    topic = state["topics"][state["current_index"]]
    today = date.today().strftime("%B %d, %Y")
    print(f"  Reviewing article for: {topic!r}  (revision #{state['revision_count']})")

    # Nothing is force-approved: if an article can't pass review within the revision
    # limit, it is DROPPED (never shipped) rather than waved through.
    if state["revision_count"] >= MAX_REVISIONS:
        print("    Could not pass review within revision limit — DROPPING article (not shipped)")
        return _advance_topic(state, approved_draft=None)

    sources       = state.get("current_sources") or []
    sources_block = "\n".join(f"- {s}" for s in sources) or "(none)"
    full_evidence = state.get("current_evidence") or ""
    evidence      = full_evidence[:EVIDENCE_CHAR_CAP]
    draft         = state["current_draft"] or ""

    # Deterministic source-URL check (Guardrail #1): every URL the draft cites must be a
    # real captured source. Fabricated/placeholder URLs trigger a revision without an LLM call.
    allowed     = {u.rstrip("/").lower() for u in _extract_urls(" ".join(sources) + " " + full_evidence)}
    draft_urls  = _extract_urls(draft)
    bad_urls    = [u for u in draft_urls if not _is_valid_source_url(u) or u.rstrip("/").lower() not in allowed]
    if bad_urls:
        feedback = (
            f"REVISION: Remove or replace the unverified URL(s) {bad_urls[:2]} — cite only a URL "
            "that appears in the Sources list, or omit the link entirely."
        )
        print(f"    Verdict: {feedback[:60]}")
        return {**state, "current_draft": feedback, "revision_count": state["revision_count"] + 1, "messages": []}

    # Fact-checking reviewer (Fix #1): verify the draft against date, research, evidence, sources.
    response = llm.invoke([
        SystemMessage(content=(
            f"You are a fact-checking newsletter editor. Today's date is {today}.\n"
            "Verify the article against the Research, Evidence, and Sources provided. "
            "Reply REVISION if ANY of these are true:\n"
            "  - It states a specific date, deadline, statistic, name, or quote that is NOT "
            "supported by the Research/Evidence.\n"
            f"  - A date is impossible or stale given today is {today} (e.g. an 'upcoming' or "
            "'current' deadline that is actually in the past, or a future event described as done).\n"
            "  - It makes a factual claim with no supporting source when sources exist.\n"
            "  - It cites any URL not present in the Sources list.\n"
            "  - It is off-topic, or not roughly 120-200 words.\n"
            "If it passes all checks, reply with exactly the single word: APPROVED\n"
            "Otherwise reply: REVISION: <one sentence naming the specific problem to fix>\n"
            "Reply with nothing else."
        )),
        HumanMessage(content=(
            f"Topic: {topic}\n\n"
            f"Research:\n{state.get('current_research') or '(none)'}\n\n"
            f"Evidence (raw source excerpts):\n{evidence or '(none)'}\n\n"
            f"Sources:\n{sources_block}\n\n"
            f"Article to review:\n{draft}"
        ))
    ])
    verdict    = response.content.strip()
    is_approved = verdict.upper() == "APPROVED" or verdict.upper().startswith("APPROVED")
    print(f"    Verdict: {'APPROVED' if is_approved else verdict[:60]}")

    if is_approved:
        result = _advance_topic(state, approved_draft=draft)
        result["messages"] = [response]
        print(f"    Articles approved so far: {len(result['articles'])}")
        return result
    return {**state, "current_draft": verdict, "revision_count": state["revision_count"] + 1, "messages": [response]}


def newsletter_compiler_node(state: NewsletterState) -> NewsletterState:
    today    = date.today().strftime("%B %d, %Y")
    if not state["articles"]:
        # No article passed review — fail loudly rather than send an empty newsletter.
        raise RuntimeError("No articles passed review — nothing to publish.")
    print(f"  Compiling {len(state['articles'])} approved article(s)")
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
    # A pending revision keeps the same topic and goes back to the writer.
    if (state.get("current_draft") or "").startswith("REVISION:"):
        return "writer"
    # Otherwise the topic was approved or dropped — advance until all topics are processed.
    if state["current_index"] >= len(state["topics"]):
        return "newsletter_compiler"
    return "researcher"


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
