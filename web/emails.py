"""
web/emails.py -- the app's emails, built the way megatanUK's are.

Each email is a 600px table. The fixed parts of the design -- header, hero
band, footer -- are PNGs rendered from design/emails (run
python design/emails/render_blocks.py after changing them), so no mail
client can reflow the type. Everything that changes per email -- the
address, the document, the counts, the link -- stays live HTML under them.

Buttons are real links in a table, never images, so a client that blocks
images (Outlook does by default) still shows the one thing to press, and
the images carry alt text for the same reason. Every email also comes as
plain text. Emails are in English whatever language the app is shown in.

    subject, html, text = emails.scan_ready(images_url, link=..., ...)
"""

from __future__ import annotations

from html import escape

INK = "#0B1040"
TEXT = "#2B3150"
MUTED = "#5A6080"
PAGE = "#EEF0FA"
PAPER = "#FBFAF6"
RULE = "#D6DCF4"
AMBER_BG = "#FBE8B8"
AMBER = "#7A4E00"
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def _block(images: str, name: str, alt: str) -> str:
    """A rendered design block: 1200px wide, shown at 600."""
    return f"""
  <tr><td style="padding: 0; font-size: 0; line-height: 0;">
    <img src="{escape(images)}/{name}.png" alt="{escape(alt)}" width="600"
         style="display: block; width: 100%; max-width: 600px; height: auto; border: 0;" />
  </td></tr>"""


def _row(inner: str, padding: str = "0 40px") -> str:
    return f"""
  <tr><td style="padding: {padding}; background: #FFFFFF;">{inner}</td></tr>"""


def _para(text: str, size: int = 16, color: str = TEXT, top: int = 0) -> str:
    """A paragraph of live text; *text* is escaped here."""
    return (f'<p style="margin: {top}px 0 0; font-family: {FONT}; font-size: {size}px; line-height: 1.6; '
            f'color: {color};">{escape(text)}</p>')


def _button(label: str, href: str) -> str:
    """Square and navy, like the app's Save button."""
    return _row(f"""
    <table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>
      <td bgcolor="{INK}" style="background: {INK};">
        <a href="{escape(href)}" style="display: inline-block; padding: 17px 30px; font-family: {FONT}; font-size: 14px;
           font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #FFFFFF; text-decoration: none;">{escape(label)}</a>
      </td>
    </tr></table>""", "26px 40px 6px")


def _details(rows: list[tuple[str, str, bool]]) -> str:
    """Label and value pairs on paper; a flagged value is in the app's amber."""
    cells = "".join(f"""
        <tr>
          <td style="padding: 10px 0; border-bottom: 1px solid {RULE}; font-family: {FONT}; font-size: 12px; font-weight: 600;
              letter-spacing: 0.1em; text-transform: uppercase; color: {MUTED}; width: 38%; vertical-align: top;">{escape(label)}</td>
          <td style="padding: 10px 0; border-bottom: 1px solid {RULE}; font-family: {FONT}; font-size: 15px; font-weight: 600; color: {INK};">
            {f'<span style="display: inline-block; padding: 2px 8px; background: {AMBER_BG}; color: {AMBER};">{escape(value)}</span>' if flag else escape(value)}
          </td>
        </tr>""" for label, value, flag in rows)
    return _row(f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background: {PAPER};">
      <tr><td style="padding: 6px 22px 8px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">{cells}</table>
      </td></tr>
    </table>""", "22px 40px 0")


def _small(text: str, padding: str = "14px 40px 0") -> str:
    return _row(_para(text, 13, MUTED), padding)


def _link_fallback(link: str) -> str:
    return _row(f'<p style="margin: 0; font-family: {FONT}; font-size: 13px; line-height: 1.6; color: {MUTED};">'
                f'Button not working? Copy this link into your browser:<br>'
                f'<a href="{escape(link)}" style="color: #3150D9; word-break: break-all;">{escape(link)}</a></p>',
                "16px 40px 30px")


def _email(images: str, hero: str, hero_alt: str, body: str, preheader: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<meta name="x-apple-disable-message-reformatting" />
<title>MSME Ledger</title>
</head>
<body style="margin: 0; padding: 0; background: {PAGE};">
  <div style="display: none; max-height: 0; overflow: hidden; opacity: 0;">{escape(preheader)}</div>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background: {PAGE};">
    <tr><td align="center" style="padding: 28px 10px 34px;">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0"
             style="width: 600px; max-width: 600px; background: #FFFFFF;">
        {_block(images, "header", "MSME Ledger")}
        {_block(images, hero, hero_alt)}
        {body}
        {_block(images, "footer", "MSME Ledger, Chinhoyi University of Technology")}
      </table>
    </td></tr>
  </table>
</body></html>"""


def _plural(n: int, word: str) -> str:
    return f"1 {word}" if n == 1 else f"{n} {word}s"


def scan_ready(images: str, *, link: str, name: str, pages: int, rows: int, unsure: int) -> tuple[str, str, str]:
    """A scan has been read: what came out of it, and the button to check it."""
    read = f"{_plural(pages, 'page')} · {_plural(rows, 'row')}"
    check = f"{_plural(unsure, 'cell')} to check"
    lines = [f"We've finished reading {name}.",
             "Check the table against your photo, fix anything that was read wrong, then save it to your account.",
             "Nothing is saved to your account until you press Save."]
    body = (_row(_para(lines[0]), "30px 40px 0") + _row(_para(lines[1], top=12))
            + _details([("Document", name, False), ("Read", read, False), ("To check", check, unsure > 0)])
            + _button("Check your table", link) + _small(lines[2]) + _link_fallback(link))
    text = "\n\n".join([lines[0], f"{read}\n{check}", f"Check your table:\n{link}", lines[1], lines[2]])
    return f"Your table is ready: {name}", _email(images, "hero-scan", "Your table is ready.", body,
                                                    f"{read} · {check}"), text


def password_reset(images: str, *, email: str, link: str) -> tuple[str, str, str]:
    lines = [f"Someone asked to reset the password for {email}.",
             "To choose a new password, press the button. The link works once, within an hour.",
             "If it wasn't you, ignore this email. Your password stays the same."]
    body = (_row(_para(lines[0]), "30px 40px 0") + _row(_para(lines[1], top=12))
            + _button("Choose a new password", link) + _small(lines[2]) + _link_fallback(link))
    text = "\n\n".join([lines[0], f"To choose a new password, open this link within an hour:\n{link}", lines[2]])
    return ("Reset your MSME Ledger password",
            _email(images, "hero-reset", "Choose a new password.", body, "The link works for an hour."), text)


def no_account(images: str, *, email: str, link: str, code_required: bool) -> tuple[str, str, str]:
    """Someone asked to reset the password of an address with no account."""
    code = " You'll need the sign-up code from the app's owner." if code_required else ""
    lines = [f"Someone asked to reset the password for {email}, but there is no MSME Ledger account for this address.",
             "If it was you, you can create one." + code,
             "If it wasn't you, ignore this email."]
    body = (_row(_para(lines[0]), "30px 40px 0") + _row(_para(lines[1], top=12))
            + _button("Create an account", link) + _small(lines[2]) + _link_fallback(link))
    text = "\n\n".join([lines[0], f"{lines[1]}\n{link}", lines[2]])
    return ("No MSME Ledger account for this address",
            _email(images, "hero-no-account", "No account for this address.", body, "If it was you, you can create one."),
            text)


def password_changed(images: str, *, email: str, when: str, link: str) -> tuple[str, str, str]:
    lines = [f"The password for {email} was changed on {when}.",
             "Any other devices that were signed in have been signed out.",
             "If this wasn't you, choose a new password straight away and tell the app's owner."]
    body = (_row(_para(lines[0]), "30px 40px 0") + _row(_para(lines[1], top=12)) + _row(_para(lines[2], top=12))
            + _button("Reset your password", link) + _row("", "0 40px 30px"))
    text = "\n\n".join(lines + [f"Reset your password:\n{link}"])
    return ("Your MSME Ledger password was changed",
            _email(images, "hero-changed", "Your password was changed.", body, lines[0]), text)
