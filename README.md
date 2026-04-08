# SightTune Newsletter Bot

An AI-powered newsletter agent for [SightTune Music Technology](https://sighttune.com) that autonomously researches, writes, reviews, and emails a monthly newsletter to subscribers.

![SightTune Logo](assets/logo.png)

## How it works

The agent runs as a [LangGraph](https://github.com/langchain-ai/langgraph) state machine with the following pipeline:

```
topic_planner → researcher ⇄ research_tools → writer → reviewer → newsletter_compiler
```

1. **Topic Planner** — Given a theme (e.g. "piano music technology"), generates 3 distinct, timely topics from the last 60 days.
2. **Researcher** — Searches the web for each topic using SerpAPI, Tavily, or a deep research agent (in that priority order).
3. **Writer** — Drafts a 150–200 word article from the research.
4. **Reviewer** — Edits and approves the draft or sends it back for revision (max 2 revision cycles).
5. **Newsletter Compiler** — Assembles all 3 articles into a branded HTML email.
6. **Mailer** — Sends the final email via Gmail SMTP in BCC batches of 490 to all subscribers loaded from Google Sheets.

## Setup

### Prerequisites

- Python 3.12+
- A Gmail account with an [App Password](https://myaccount.google.com/apppasswords) configured
- API keys for OpenAI, SerpAPI, and Tavily
- A Google Cloud service account with access to a Google Sheet containing your subscriber list

### Install

```bash
pip install -r requirements.txt
```

### Configure environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (uses `gpt-4o-mini`) |
| `SERP_API` | SerpAPI key for web search |
| `TAVILY_API` | Tavily API key for AI search |
| `GMAIL_USER` | Gmail address to send from |
| `GMAIL_APP_PASSWORD` | 16-character Gmail App Password |
| `GOOGLE_SHEET_ID` | ID from your Google Sheet URL |
| `GOOGLE_CREDENTIALS` | Full service account JSON (as a string) |
| `NEWSLETTER_THEME` | Override the default theme (optional) |
| `SEND_EMAIL` | Set to `false` to skip sending (default: `true`) |
| `LANGCHAIN_API_KEY` | LangSmith tracing key (optional) |

### Google Sheet structure

The subscriber sheet should have:
- **Column A**: Email addresses (header in row 1, data from row 2)
- **Column B**: Date subscribed

Share the sheet with your service account email.

### Subscriber signup automation

`data/signup_automation.gs` is a Google Apps Script that can be attached to a Google Form to automatically add new subscribers to the sheet.

## Running locally

```bash
python -m src.run
```

The agent alternates themes by month (even months → "piano music technology", odd months → "classical music") unless `NEWSLETTER_THEME` is set.

Output is saved to `logs/newsletter_YYYY-MM-DD.html`. Run metrics are appended to `logs/history.jsonl`.

## GitHub Actions

The newsletter sends automatically on the **3rd Wednesday of every month at 9am ET** via `.github/workflows/newsletter.yml`. It can also be triggered manually from the Actions tab.

Required repository secrets (mirror the `.env` variables above):
`OPENAI_API_KEY`, `SERP_API`, `TAVILY_API`, `GMAIL_USER`, `GMAIL_APP_PASSWORD`, `GOOGLE_CREDENTIALS`, `GOOGLE_SHEET_ID`, `LANGCHAIN_API_KEY`

After each run, metrics are committed back to `logs/history.jsonl` and the full HTML + logs are uploaded as a workflow artifact (retained 90 days).

## Development

```bash
# Lint
ruff check src tests

# Tests
pytest
```

## Project structure

```
src/
  agent.py          # LangGraph graph definition and all nodes
  run.py            # Entry point — orchestrates the run and handles metrics
  tools.py          # Research tools (web_search, tavily_search, deep_research)
  mailer.py         # Gmail SMTP sender with BCC batching
  subscribers.py    # Google Sheets subscriber loader
  email_template.py # HTML email builder
  state.py          # LangGraph state type definition
data/
  signup_automation.gs  # Google Apps Script for subscriber signup
logs/
  history.jsonl     # Append-only run metrics log
```
