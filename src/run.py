"""
Entry point for the SightTune newsletter agent.
Run with:  python -m src.run
"""
import json
import os
import time
import traceback
import uuid
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback

from src.agent import build_graph
from src.mailer import send_newsletter

load_dotenv()

# Remap env var names to what LangChain expects
os.environ["SERPAPI_API_KEY"] = os.getenv("SERP_API", "")
os.environ["TAVILY_API_KEY"]  = os.getenv("TAVILY_API", "")

# LangSmith tracing (optional — set LANGCHAIN_API_KEY to enable)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = "sighttune-newsletter"

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


THEMES = [
    "piano music technology",
    "classical music",
]


def main():
    # Alternate themes by month: even months → THEMES[0], odd months → THEMES[1]
    # Override anytime by setting NEWSLETTER_THEME in the environment.
    default_theme = THEMES[date.today().month % 2]
    theme      = os.getenv("NEWSLETTER_THEME", default_theme)
    send_email = os.getenv("SEND_EMAIL", "true").lower() == "true"

    print(f"{'='*55}")
    print("  SightTune Newsletter Agent")
    print(f"  Theme : {theme}")
    print(f"  Date  : {date.today().isoformat()}")
    print(f"{'='*55}\n")

    app = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = {
        "theme":             theme,
        "topics":            [],
        "current_index":     0,
        "current_research":  None,
        "current_draft":     None,
        "revision_count":    0,
        "articles":          [],
        "research_messages": [],
        "messages":          [],
        "output":            None,
    }

    node_counts  = {}
    start_time   = time.time()
    final_output = None

    try:
        with get_openai_callback() as cb:
            for event in app.stream(initial_state, config=config):
                for node_name, node_output in event.items():
                    elapsed = time.time() - start_time
                    node_counts[node_name] = node_counts.get(node_name, 0) + 1
                    print(f"[{elapsed:6.1f}s] {node_name}")

                    if node_name == "newsletter_compiler":
                        final_output = node_output.get("output")

    except Exception:
        print("\nERROR — agent failed:")
        traceback.print_exc()
        raise SystemExit(1)

    total_time = time.time() - start_time

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = {
        "date":            date.today().isoformat(),
        "theme":           theme,
        "duration_s":      round(total_time, 1),
        "total_tokens":    cb.total_tokens,
        "total_cost":      round(cb.total_cost, 4),
        "articles_written": len(initial_state.get("articles", [])),
        "node_counts":     node_counts,
    }

    metrics_path = LOGS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    history_path = LOGS_DIR / "history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

    print(f"\n{'='*55}")
    print(f"  Done in {total_time:.1f}s")
    print(f"  Tokens : {cb.total_tokens:,}  |  Cost: ${cb.total_cost:.4f}")
    print(f"  Nodes  : {node_counts}")
    print(f"{'='*55}\n")

    # ── GitHub Actions summary ─────────────────────────────────────────────────
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("## SightTune Newsletter Run\n")
            f.write("| Metric | Value |\n|--------|-------|\n")
            f.write(f"| Date | {metrics['date']} |\n")
            f.write(f"| Duration | {metrics['duration_s']}s |\n")
            f.write(f"| Tokens | {metrics['total_tokens']:,} |\n")
            f.write(f"| Cost | ${metrics['total_cost']:.4f} |\n")
            f.write(f"| Node visits | {node_counts} |\n")

    # ── Save HTML output ───────────────────────────────────────────────────────
    if final_output:
        output_path = LOGS_DIR / f"newsletter_{date.today().isoformat()}.html"
        output_path.write_text(final_output, encoding="utf-8")
        print(f"HTML saved: {output_path}")

        if send_email:
            subject = f"SightTune Newsletter — {date.today().strftime('%B %Y')}"
            send_newsletter(final_output, subject)
    else:
        print("WARNING: no newsletter output captured")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
