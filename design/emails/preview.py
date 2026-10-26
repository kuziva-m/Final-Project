"""design/emails/preview.py -- every email as an HTML file.

    python design/emails/preview.py      # writes design/emails/previews/*.html

Nothing is sent: this is how the design gets checked in a browser (and
pasted into a real mail client) without putting test mail in anyone's inbox.
The images point at the local blocks in web/static/email, so run
render_blocks.py first after changing a block.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from web import emails  # noqa: E402

OUT = Path(__file__).resolve().parent / "previews"
IMAGES = (ROOT / "web" / "static" / "email").as_uri()
SITE = "https://www.digisme.site"

SAMPLES = {
    "scan-ready": lambda: emails.scan_ready(IMAGES, link=f"{SITE}/review/20261026T101500_seed_book.pdf",
                                            name="Seed book, December", pages=3, rows=41, unsure=4),
    "password-reset": lambda: emails.password_reset(IMAGES, email="owner@example.com",
                                                    link=f"{SITE}/reset/example-token"),
    "no-account": lambda: emails.no_account(IMAGES, email="owner@example.com", link=f"{SITE}/signup",
                                            code_required=True),
    "password-changed": lambda: emails.password_changed(IMAGES, email="owner@example.com",
                                                        when="26 Oct 2026, 10:52 UTC", link=f"{SITE}/forgot"),
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, build in SAMPLES.items():
        subject, html, text = build()
        (OUT / f"{name}.html").write_text(html, encoding="utf-8")
        (OUT / f"{name}.txt").write_text(f"Subject: {subject}\n\n{text}\n", encoding="utf-8")
        print(f"{name}: {subject}")
    print(f"written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
