"""Unit tests for the HITL approval check — no network, no real IMAP."""
from datetime import date
from email.message import EmailMessage
from unittest import mock

from src.approval import APPROVAL_SUBJECT_PREFIX, check_for_approval


def _raw_email(subject: str, body: str) -> bytes:
    msg = EmailMessage()
    msg["From"] = "operator@example.com"
    msg["To"] = "bot@example.com"
    msg["Subject"] = subject
    msg.set_content(body)
    return msg.as_bytes()


class _FakeIMAP:
    """Minimal stand-in for imaplib.IMAP4_SSL used as a context manager."""

    def __init__(self, raw_messages):
        # raw_messages: list[bytes], one per "message" the search returns
        self._raw = raw_messages
        self.stored = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        return ("OK", [b"LOGIN OK"])

    def select(self, mailbox):
        return ("OK", [b"1"])

    def search(self, charset, *criteria):
        ids = " ".join(str(i + 1) for i in range(len(self._raw)))
        return ("OK", [ids.encode()])

    def fetch(self, msg_id, parts):
        idx = int(msg_id) - 1
        return ("OK", [(b"header", self._raw[idx])])

    def store(self, msg_id, flags, value):
        self.stored.append(msg_id)
        return ("OK", [b"STORE OK"])


def _run_check(raw_messages):
    fake = _FakeIMAP(raw_messages)
    env = {"GMAIL_USER": "bot@example.com", "GMAIL_APP_PASSWORD": "pw",
           "APPROVAL_EMAIL": "operator@example.com"}
    with mock.patch.dict("os.environ", env), \
         mock.patch("src.approval.imaplib.IMAP4_SSL", return_value=fake):
        result = check_for_approval(since_date=date(2026, 6, 1))
    return result, fake


def test_approves_on_matching_subject_and_body():
    raw = _raw_email(
        subject=f"Re: {APPROVAL_SUBJECT_PREFIX} SightTune Newsletter — June 2026",
        body="Approved, looks great!",
    )
    result, fake = _run_check([raw])
    assert result is True
    assert fake.stored, "matched message should be flagged \\Seen"


def test_ignores_approved_word_with_unrelated_subject():
    # A real send must NOT trigger just because some unrelated email says "approved".
    raw = _raw_email(subject="Lunch tomorrow?", body="Sounds approved by me, see you then")
    result, _ = _run_check([raw])
    assert result is False


def test_matching_subject_without_approved_word_is_not_approval():
    raw = _raw_email(
        subject=f"Re: {APPROVAL_SUBJECT_PREFIX} SightTune Newsletter — June 2026",
        body="Hold off, I want to change the headline first.",
    )
    result, _ = _run_check([raw])
    assert result is False


def test_no_messages_returns_false():
    result, _ = _run_check([])
    assert result is False
