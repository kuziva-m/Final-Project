"""
web/mail.py -- send the app's emails (built in web/emails.py).

Hugging Face Spaces block outgoing SMTP, so mail goes over HTTPS to Resend
(resend.com). It needs two settings:

    RESEND_API_KEY    an API key from resend.com
    LEDGER_MAIL_FROM  the sender, e.g. "MSME Ledger <noreply@mail.digisme.site>",
                      on a domain verified in Resend

Without them email is off, and the forgot-password page says so.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

_API = "https://api.resend.com/emails"


def enabled() -> bool:
    return bool(os.environ.get("RESEND_API_KEY") and os.environ.get("LEDGER_MAIL_FROM"))


def send(to: str, subject: str, text: str, html: str | None = None) -> bool:
    """Send an email (plain text, and HTML when given); True when Resend accepted it."""
    if not enabled():
        return False
    message = {"from": os.environ["LEDGER_MAIL_FROM"], "to": [to], "subject": subject, "text": text}
    if html:
        message["html"] = html
    request = urllib.request.Request(
        _API,
        data=json.dumps(message).encode("utf-8"),
        headers={"Authorization": f"Bearer {os.environ['RESEND_API_KEY']}",
                 "Content-Type": "application/json", "User-Agent": "msme-ledger"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError) as exc:
        # The address itself stays out of the log.
        log.warning("email to someone at %s was not sent: %s", to.rpartition("@")[2], exc)
        return False
