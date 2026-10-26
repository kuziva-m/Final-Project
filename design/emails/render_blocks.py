"""design/emails/render_blocks.py -- render the fixed parts of the emails to PNGs.

    python design/emails/render_blocks.py

Built the way megatanUK's emails are: the fixed parts of the design (header,
hero band, footer) ship as images, so no mail client can reflow the type,
and everything that changes per email stays live HTML (see web/emails.py).
Each block in design/emails/blocks is rendered with Microsoft Edge (or
Chrome) in headless mode at 2x, into web/static/email/, which the app serves
and the emails link to. The emails are in English only.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BLOCKS_DIR = Path(__file__).resolve().parent / "blocks"
OUT_DIR = ROOT / "web" / "static" / "email"
FONTS = (ROOT / "web" / "static" / "fonts").as_uri()

# name: (template, values), rendered to name.png.
BLOCKS = {
    "header": ("header.html", {}),
    "hero-scan": ("hero.html", {"eyebrow": "Scan complete", "heading": "Your table is ready.", "drawing": "scan"}),
    "hero-reset": ("hero.html", {"eyebrow": "Password reset", "heading": "Choose a new password.", "drawing": "key"}),
    "hero-changed": ("hero.html", {"eyebrow": "Account security", "heading": "Your password was changed.",
                                   "drawing": "lock"}),
    "hero-no-account": ("hero.html", {"eyebrow": "Password reset", "heading": "No account for this address.",
                                      "drawing": "envelope"}),
    "footer": ("footer.html", {}),
}


def browser() -> str:
    candidates = [os.environ.get("LEDGER_BROWSER", ""),
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  "/usr/bin/google-chrome", "/usr/bin/chromium"]
    for path in candidates:
        if path and Path(path).is_file():
            return path
    sys.exit("No Edge or Chrome found; set LEDGER_BROWSER to one.")


def run(exe: str, *args: str) -> str:
    return subprocess.run([exe, "--headless", "--disable-gpu", "--hide-scrollbars", *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def main() -> None:
    exe = browser()
    env = Environment(loader=FileSystemLoader([str(BLOCKS_DIR), str(ROOT / "web" / "templates")]), autoescape=True)
    emblem = Markup((ROOT / "web" / "templates" / "partials" / "emblem.svg").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, (template, values) in BLOCKS.items():
            html = env.get_template(template).render(lang="en", _=lambda text, **v: text.format(**v),
                                                     fonts=FONTS, emblem=emblem, **values)
            page = Path(tmp) / f"{name}.html"
            page.write_text(html, encoding="utf-8")
            found = re.search(r'data-h="(\d+)"', run(exe, "--virtual-time-budget=4000", "--dump-dom", page.as_uri()))
            if not found:
                sys.exit(f"{name}: the page did not report its height")
            height = int(found.group(1))
            # 2x for sharp text on phones, by zooming the page rather than the
            # screen: headless Edge crops the shot when told to scale itself.
            # Its window frame also takes some of the window, so shoot a
            # larger window and cut the block out of the top-left corner.
            page.write_text(html.replace("</head>", "<style>html { zoom: 2; }</style></head>"), encoding="utf-8")
            png = OUT_DIR / f"{name}.png"
            run(exe, "--virtual-time-budget=4000", f"--window-size=1400,{height * 2 + 300}",
                f"--screenshot={png}", page.as_uri())
            with Image.open(png) as shot:
                shot.crop((0, 0, 1200, height * 2)).save(png, optimize=True)
            kb = png.stat().st_size // 1024
            total += kb
            print(f"{png.name:22} 600x{height}  {kb} KB")
    print(f"written to {OUT_DIR.relative_to(ROOT)}, {total} KB in all")


if __name__ == "__main__":
    main()
