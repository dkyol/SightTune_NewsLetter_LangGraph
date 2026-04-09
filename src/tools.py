import concurrent.futures

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool


def run_topic_search(query: str) -> str:
    """Plain search for topic discovery — not an agent tool, called directly by the topic planner."""
    return SerpAPIWrapper().run(query)


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use this FIRST — default search tool."""
    return SerpAPIWrapper().run(query)


@tool
def tavily_search(query: str) -> str:
    """AI-optimized search with structured results. Use only when web_search is insufficient."""
    return str(TavilySearchResults(max_results=5).invoke(query))


@tool
def deep_research(query: str) -> str:
    """For complex topics requiring multi-step investigation. Use only when other tools are insufficient."""
    def _run():
        from deepagents import create_deep_agent  # lazy import
        agent = create_deep_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_run)
        try:
            return future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            return "Deep research timed out. Summarise from web_search results only."


RESEARCH_TOOLS = [web_search, tavily_search, deep_research]
