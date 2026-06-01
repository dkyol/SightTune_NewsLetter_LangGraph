"""
Approval-check-and-send step.
Called by check_approval.yml every 3 hours.

  - Finds the most recent pending/newsletter_YYYY-MM-DD.html
  - If older than 3 days: removes it (expired, no approval received) and exits
  - Otherwise: checks Gmail IMAP once for an approval reply
  - If approved: sends the newsletter to subscribers and removes the pending file
  - If not yet approved: exits 0 — cron will retry in 3 hours
"""
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from src.approval import APPROVAL_SUBJECT_PREFIX, check_for_approval
from src.mailer import send_newsletter

load_dotenv(override=True)

os.environ["SERPAPI_API_KEY"] = os.getenv("SERP_API", "")
os.environ["TAVILY_API_KEY"]  = os.getenv("TAVILY_API", "")

PENDING_DIR  = Path(__file__).parent.parent / "pending"
EXPIRY_DAYS  = 3


def main():
    pending_files = sorted(PENDING_DIR.glob("newsletter_*.html"))
    if not pending_files:
        print("No pending newsletter — nothing to do.")
        return

    pending_path = pending_files[-1]
    date_str = pending_path.stem.replace("newsletter_", "")
    try:
        newsletter_date = date.fromisoformat(date_str)
    except ValueError:
        print(f"Unexpected pending filename: {pending_path.name} — skipping.")
        return

    age = date.today() - newsletter_date
    if age > timedelta(days=EXPIRY_DAYS):
        print(f"Pending newsletter {pending_path.name} expired after {EXPIRY_DAYS} days — removing.")
        pending_path.unlink()
        return

    print(f"Checking approval for {pending_path.name} (day {age.days + 1} of {EXPIRY_DAYS})...")
    approved = check_for_approval(since_date=newsletter_date)

    if not approved:
        print("No approval yet — will retry in 3 hours.")
        return

    html    = pending_path.read_text(encoding="utf-8")
    subject = f"SightTune Newsletter — {newsletter_date.strftime('%B %Y')}"
    send_newsletter(html, subject)

    pending_path.unlink()
    print(f"Newsletter sent and {pending_path.name} removed.")


if __name__ == "__main__":
    main()
