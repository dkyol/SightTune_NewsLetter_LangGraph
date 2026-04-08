"""
Loads the subscriber list from a private Google Sheet.

Sheet structure (set up once):
    Row 1  : headers  →  Email | Date Subscribed
    Row 2+ : data     →  subscriber@email.com | 2026-04-01

Authentication: Google Service Account credentials stored as the
GOOGLE_CREDENTIALS GitHub Secret (the full JSON key as a string).
"""
import json
import os

import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def load_subscribers() -> list[str]:
    """Return a deduplicated list of subscriber emails from Google Sheets."""
    creds_json = os.environ["GOOGLE_CREDENTIALS"]   # full service account JSON string
    sheet_id   = os.environ["GOOGLE_SHEET_ID"]      # the long ID from the Sheet URL

    try:
        creds_data = json.loads(creds_json)
    except json.JSONDecodeError as e:
        raise RuntimeError("GOOGLE_CREDENTIALS is not valid JSON") from e

    try:
        creds  = Credentials.from_service_account_info(creds_data, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet  = client.open_by_key(sheet_id).sheet1
    except Exception as e:
        raise RuntimeError(f"Failed to access Google Sheet — check credentials and sheet permissions: {e}") from e

    try:
        raw = sheet.col_values(1)[1:]
    except Exception as e:
        raise RuntimeError(f"Failed to read subscriber column from sheet: {e}") from e

    emails = list({e.strip().lower() for e in raw if e.strip() and "@" in e})
    print(f"Loaded {len(emails)} subscribers from Google Sheet")
    return emails
