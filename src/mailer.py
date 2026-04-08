import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src.subscribers import load_subscribers

BATCH_SIZE = 490  # safely under Gmail's 500 recipients-per-message limit


def send_newsletter(html_content: str, subject: str) -> None:
    """Send the newsletter HTML via Gmail SMTP in BCC batches of 490."""
    gmail_user     = os.environ["GMAIL_USER"]
    gmail_password = os.environ["GMAIL_APP_PASSWORD"]
    to_emails      = load_subscribers()

    if not to_emails:
        print("No subscribers found — skipping send.")
        return

    total_batches = -(-len(to_emails) // BATCH_SIZE)  # ceiling division

    try:
        smtp = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Gmail SMTP: {e}") from e

    try:
        smtp.login(gmail_user, gmail_password)
    except smtplib.SMTPAuthenticationError as e:
        smtp.quit()
        raise RuntimeError("Gmail authentication failed — check GMAIL_USER and GMAIL_APP_PASSWORD") from e

    failed_batches = []
    with smtp:
        for i in range(0, len(to_emails), BATCH_SIZE):
            batch     = to_emails[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = gmail_user
            msg["To"]      = gmail_user        # only sender visible
            msg["Bcc"]     = ", ".join(batch)  # subscribers hidden from each other
            msg.attach(MIMEText(html_content, "html"))

            try:
                smtp.sendmail(gmail_user, batch, msg.as_string())
                print(f"  Batch {batch_num}/{total_batches} sent → {len(batch)} recipients")
            except Exception as e:
                print(f"  Batch {batch_num}/{total_batches} FAILED: {e}")
                failed_batches.append(batch_num)

    if failed_batches:
        raise RuntimeError(f"Send incomplete — failed batches: {failed_batches}")
    print(f"Done — {len(to_emails)} total recipients across {total_batches} batch(es)")
