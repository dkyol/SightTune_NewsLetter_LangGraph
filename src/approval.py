"""
Human-in-the-loop approval gate.

Flow:
  1. send_approval_request()  — called by run.py after generation; emails the newsletter
                                preview to the operator and exits. The HTML is saved to
                                pending/ by run.py so the send step can find it later.
  2. check_for_approval()     — called by run_send.py every 3 hours via cron; makes one
                                non-blocking IMAP check and returns True/False immediately.

The operator email defaults to GMAIL_USER (send-to-self) but can be overridden
via the APPROVAL_EMAIL env var.
"""
import email
import email.utils
import imaplib
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

APPROVAL_SUBJECT_PREFIX = "[APPROVAL REQUIRED]"
_IMAP_HOST = "imap.gmail.com"
_IMAP_PORT = 993


def send_approval_request(html_content: str, newsletter_subject: str) -> None:
    """Email the newsletter preview for human review."""
    gmail_user     = os.environ["GMAIL_USER"]
    gmail_password = os.environ["GMAIL_APP_PASSWORD"]
    approval_email = os.getenv("APPROVAL_EMAIL", gmail_user)

    subject = f"{APPROVAL_SUBJECT_PREFIX} {newsletter_subject}"

    msg            = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = gmail_user
    msg["To"]      = approval_email

    plain = (
        "A new SightTune newsletter is ready for review.\n\n"
        "Reply to this email with the word 'approved' anywhere in the body to send it out.\n"
        "The newsletter preview is shown below in the HTML version of this email.\n\n"
        "The bot will check for your reply every 3 hours for up to 3 days."
    )
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
        smtp.login(gmail_user, gmail_password)
        smtp.sendmail(gmail_user, [approval_email], msg.as_string())

    print(f"Approval request sent to {approval_email}")
    print(f"  Subject : {subject}")
    print("  Waiting : reply with 'approved' in the body within 3 days.")


def check_for_approval(since_date: date) -> bool:
    """Single non-blocking IMAP check. Returns True if an approval reply was found.

    Looks for any UNSEEN message from the approval address received on or after
    `since_date` that contains 'approved' anywhere in the plain-text body.
    Marks matched messages as read to prevent double-matching on the next check.
    """
    gmail_user     = os.environ["GMAIL_USER"]
    gmail_password = os.environ["GMAIL_APP_PASSWORD"]
    approval_email = os.getenv("APPROVAL_EMAIL", gmail_user)

    # IMAP SINCE format: "01-Jun-2026"
    since_str = since_date.strftime("%d-%b-%Y")

    try:
        with imaplib.IMAP4_SSL(_IMAP_HOST, _IMAP_PORT) as imap:
            imap.login(gmail_user, gmail_password)
            imap.select("INBOX")

            _, msg_ids = imap.search(
                None, f'FROM "{approval_email}" UNSEEN SINCE {since_str}'
            )
            if not msg_ids or not msg_ids[0]:
                print("  No approval email found.")
                return False

            for msg_id in msg_ids[0].split():
                _, msg_data = imap.fetch(msg_id, "(RFC822)")
                raw_bytes = msg_data[0][1]
                msg = email.message_from_bytes(raw_bytes)

                body = _extract_plain_body(msg)
                if "approved" in body.lower():
                    imap.store(msg_id, "+FLAGS", "\\Seen")
                    print(f"  Approval received! (IMAP id {msg_id.decode()})")
                    return True

        print("  No approval found in matching messages.")
        return False

    except Exception as e:
        print(f"  IMAP check failed: {e}")
        return False


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_plain_body(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.get("Content-Disposition"):
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="ignore")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode("utf-8", errors="ignore")
    return ""
