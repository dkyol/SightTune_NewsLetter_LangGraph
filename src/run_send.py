"""
Approval-check-and-send step.
Called by check_approval.yml about every 6 hours.

  - Finds the most recent pending/newsletter_YYYY-MM-DD.html
  - If older than 3 days: notifies the operator, removes it (expired) and exits
  - Otherwise: checks Gmail IMAP once for an approval reply
  - If approved: sends the newsletter to subscribers and removes the pending file
  - If not yet approved: exits 0 — cron will retry in about 6 hours
"""
import json
import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from src.approval import check_for_approval, send_expiry_notification
from src.mailer import send_newsletter

load_dotenv(override=True)

os.environ["SERPAPI_API_KEY"] = os.getenv("SERP_API", "")
os.environ["TAVILY_API_KEY"]  = os.getenv("TAVILY_API", "")

PENDING_DIR  = Path(__file__).parent.parent / "pending"
EXPIRY_DAYS  = 3


def _load_meta(pending_path: Path, newsletter_date: date) -> dict:
    """Read the sidecar metadata; fall back to defaults for older pending files."""
    default_subject = f"SightTune Newsletter — {newsletter_date.strftime('%B %Y')}"
    meta_path = pending_path.with_suffix(".json")
    if not meta_path.exists():
        return {"approval_email": None, "subject": default_subject}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Could not read sidecar {meta_path.name}: {e} — using defaults.")
        return {"approval_email": None, "subject": default_subject}
    meta.setdefault("subject", default_subject)
    meta.setdefault("approval_email", None)
    return meta


def main():
    pending_files = sorted(PENDING_DIR.glob("newsletter_*.html"))
    if not pending_files:
        print("No pending newsletter — nothing to do.")
        return

    pending_path = pending_files[-1]
    meta_path = pending_path.with_suffix(".json")
    date_str = pending_path.stem.replace("newsletter_", "")
    try:
        newsletter_date = date.fromisoformat(date_str)
    except ValueError:
        print(f"Unexpected pending filename: {pending_path.name} — skipping.")
        return

    meta    = _load_meta(pending_path, newsletter_date)
    subject = meta["subject"]

    # Use the exact operator address the request was sent to, so the reply-check
    # matches even if APPROVAL_EMAIL changed since generation.
    if meta["approval_email"]:
        os.environ["APPROVAL_EMAIL"] = meta["approval_email"]

    age = date.today() - newsletter_date
    if age > timedelta(days=EXPIRY_DAYS):
        print(f"Pending newsletter {pending_path.name} expired after {EXPIRY_DAYS} days — removing.")
        send_expiry_notification(subject)
        pending_path.unlink()
        meta_path.unlink(missing_ok=True)
        return

    print(f"Checking approval for {pending_path.name} (day {age.days + 1} of {EXPIRY_DAYS})...")
    approved = check_for_approval(since_date=newsletter_date)

    if not approved:
        print("No approval yet — will retry in about 6 hours.")
        return

    html = pending_path.read_text(encoding="utf-8")
    send_newsletter(html, subject)

    pending_path.unlink()
    meta_path.unlink(missing_ok=True)
    print(f"Newsletter sent and {pending_path.name} removed.")


if __name__ == "__main__":
    main()
