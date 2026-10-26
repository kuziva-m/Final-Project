"""
web/app.py -- Flask front end for the OCR ledger pipeline, serving both the
HTML review UI and a JSON API for the companion mobile app.

No new OCR/preprocessing/extraction logic lives here; this module only wires
together the existing pipeline:

    preprocessing.clean.preprocess -- deskew/denoise/CLAHE/binarise
    ocr.fusion.run_engines         -- Tesseract-on-cleaned + EasyOCR-on-raw
    ocr.fusion.fuse_from_texts     -- fused into structured rows + provenance

See ocr/fusion.py for the fusion rule: Tesseract's rows are the structural
skeleton (date, item, column layout); EasyOCR supplies clean numeric values
where the two disagree, and every decision is logged (provenance) so the
result is auditable -- no model, only rules.

When ANTHROPIC_API_KEY is set, ocr/claude_reader.py reads the photo instead
and returns rows in the same shape; if that call fails, the fusion above is
used. The engine that produced the rows is always reported as used_engine.

Documents read with their own headings are saved in the scans table and get
their own page (/documents/<id>) to download, share, edit or add pages to.

Scanning needs an account: every upload, document and record belongs to the
user who made it, and no one else can see it. On a Hugging Face Space the
database is backed up to a private dataset (web/backup.py), since the
Space's own disk is wiped whenever it restarts.

Run:
    python -m web.app
    (or) python web/app.py
"""

from __future__ import annotations

import csv
import hmac
import io
import json
import logging
import os
import re
import secrets
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow `python web/app.py` to find the top-level packages (preprocessing,
# ocr, extraction) regardless of the process's working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
from flask import (
    Flask,
    flash,
    g,
    has_request_context,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from itsdangerous import BadSignature, URLSafeTimedSerializer
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from extraction.fields import FIELDNAMES
from ocr import claude_reader
from ocr.fusion import fuse_from_texts, run_engines
from preprocessing.clean import preprocess
from preprocessing.pdf_pages import PdfError, render_pages
from web.backup import Backup
from web.exports import build_workbook, document_label
from web import emails, mail, statement
from web.i18n import JS_STRINGS, LANGUAGES, translate

# ---------------------------------------------------------------------------
# Paths -- everything reads from / writes to folders, nothing is hardcoded.
# ---------------------------------------------------------------------------

DATA_DIR = _REPO_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "records.db"

for _dir in (UPLOADS_DIR, PROCESSED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}

app = Flask(__name__)
# Scanned PDFs can be large; reject anything over the limit with a clear message.
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("LEDGER_MAX_UPLOAD_MB", "100")) * 1024 * 1024
# Flask's 500 KB default for submitted form fields is too small for a review
# page holding many pages of tables.
app.config["MAX_FORM_MEMORY_SIZE"] = 16 * 1024 * 1024


def _secret_key() -> str:
    """FLASK_SECRET_KEY, or a random key kept in data/secret_key.

    The session cookie says who is signed in, so its signing key must be
    secret. Keeping the made-up key in a file means a laptop running the app
    offline doesn't sign everyone out each time it restarts. (On a Space the
    disk is wiped on restart, so set FLASK_SECRET_KEY there.)"""
    if os.environ.get("FLASK_SECRET_KEY"):
        return os.environ["FLASK_SECRET_KEY"]
    path = DATA_DIR / "secret_key"
    try:
        key = path.read_text(encoding="ascii").strip()
        if len(key) >= 32:
            return key
    except OSError:
        pass
    key = secrets.token_hex(32)
    try:
        path.write_text(key, encoding="ascii")
    except OSError:
        pass  # read-only disk: the key lasts until the server restarts
    return key


app.secret_key = _secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    # Not sent with requests from other sites, so they can't post forms as you.
    SESSION_COOKIE_SAMESITE="Lax",
    # A Space is served over HTTPS (SPACE_ID is set by Hugging Face).
    SESSION_COOKIE_SECURE=bool(os.environ.get("SPACE_ID")),
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)
logging.basicConfig(level=logging.INFO)
backup = Backup.from_environment()


# ---------------------------------------------------------------------------
# Language -- English, chiShona or isiNdebele, chosen with a cookie. Strings
# are keyed by their English text (see web/i18n.py); anything without a
# translation shows in English.
# ---------------------------------------------------------------------------


def current_lang() -> str:
    lang = request.cookies.get("lang", "en") if has_request_context() else "en"
    return lang if lang in LANGUAGES else "en"


def tr(text: str, **values) -> str:
    """*text* in the visitor's language, with ``{name}`` placeholders filled."""
    return translate(text, current_lang(), **values)


app.jinja_env.globals.update(_=tr, LANGUAGES=LANGUAGES)


@app.context_processor
def inject_language() -> dict:
    return {"lang": current_lang(), "js_strings": {text: tr(text) for text in JS_STRINGS}}


# ---------------------------------------------------------------------------
# SQLite -- stdlib sqlite3, table auto-created on startup.
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create the tables if they do not already exist.

    users   -- accounts: email, password hash, and a session key that signs
               out every device when changed; name and position (shown to
               colleagues), and the organisation they belong to, if any.
    organisations -- businesses whose staff share their documents; code is
               what staff enter to join.
    uploads -- which user each upload belongs to, by the time stamp that
               starts all of its file names, and its organisation then.
    records -- fixed date/item/qty/price/total rows (OCR fusion, mobile app).
    scans   -- documents read with their own headings; tables_json holds
               [{"title", "columns", "rows", "page", "image"}], since columns
               vary per business. name is the owner's name for the document.
               user_id scanned it; org_id, when set, shares it with everyone
               in that organisation; updated_by changed it last.
    """
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT,
                item        TEXT,
                qty         TEXT,
                price       TEXT,
                total       TEXT,
                source_file TEXT,
                created_at  TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                engine      TEXT,
                tables_json TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                session_key   TEXT NOT NULL,
                created_at    TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                stamp       TEXT PRIMARY KEY,
                user_id     INTEGER NOT NULL,
                created_at  TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS organisations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                code        TEXT NOT NULL UNIQUE,
                created_at  TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE TABLE IF NOT EXISTS app_state (key TEXT PRIMARY KEY, value TEXT)")
        # Columns added after the first release; older databases get them here.
        for table, columns in (("scans", ("name TEXT", "updated_at TEXT", "user_id INTEGER", "org_id INTEGER",
                                          "updated_by INTEGER")),
                               ("records", ("user_id INTEGER",)),
                               ("users", ("name TEXT", "position TEXT", "org_id INTEGER",
                                          "org_admin INTEGER NOT NULL DEFAULT 0")),
                               ("uploads", ("org_id INTEGER",))):
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for column in columns:
                if column.split()[0] not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
        seeded = _seed_organisations(conn)
        conn.commit()
    if seeded:
        backup.mark_dirty()


def _seed_organisations(conn: sqlite3.Connection) -> bool:
    """Create the organisations named in LEDGER_ORGANISATIONS, e.g.
    "Delta=DELTA2026, Acme=ACME7731". Each is created once only, so renaming
    it or changing its code in the app later sticks. The first person to join
    one becomes its admin."""
    seeded = False
    now = datetime.now(timezone.utc).isoformat()
    for entry in os.environ.get("LEDGER_ORGANISATIONS", "").split(","):
        name, _, code = entry.partition("=")
        name, code = name.strip()[:80], _org_code(code)
        if not name or len(code) < 4:
            continue
        key = f"seeded-organisation:{code}"
        if conn.execute("SELECT 1 FROM app_state WHERE key = ?", (key,)).fetchone():
            continue
        if not conn.execute("SELECT 1 FROM organisations WHERE code = ?", (code,)).fetchone():
            conn.execute("INSERT INTO organisations (name, code, created_at) VALUES (?, ?, ?)", (name, code, now))
            app.logger.info("Created the organisation %s from LEDGER_ORGANISATIONS.", name)
        conn.execute("INSERT INTO app_state (key, value) VALUES (?, ?)", (key, now))
        seeded = True
    return seeded


def get_db() -> sqlite3.Connection:
    """Return the request-scoped SQLite connection, creating it if needed."""
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc: BaseException | None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ---------------------------------------------------------------------------
# Accounts -- scanning needs a signed-in user, and everyone sees only their own
# uploads, documents and records -- except that the staff of an organisation
# (joined with its code) all see its documents. The web pages use the session
# cookie; the phone app gets a token from /api/login and sends it as
# "Authorization: Bearer <token>".
# ---------------------------------------------------------------------------

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_MIN_PASSWORD = 8
_api_tokens = URLSafeTimedSerializer(app.secret_key, salt="ledger-api")
_API_TOKEN_DAYS = 30

# Pages anyone can open; everything else needs a signed-in user.
_PUBLIC_ENDPOINTS = {"index", "login", "signup", "logout", "set_language", "static", "api_health", "api_login",
                     "forgot_password", "reset_password"}
# Pages that open before someone who has joined an organisation has said
# their name and position (every other page asks for them first).
_BEFORE_WELCOME = {"welcome", "logout", "set_language", "static", "account", "leave_organisation",
                   "processed_file", "uploaded_file"}

# Repeated wrong passwords for one email are refused for a while.
_failed_logins: dict[str, list[float]] = {}
_failed_lock = threading.Lock()
_MAX_FAILURES, _FAILURE_WINDOW = 10, 15 * 60


def _normal_code(text: str) -> str:
    """A sign-up code as compared: case, spaces and quotes around it don't matter."""
    return text.strip().strip("'\"").strip().casefold()


def _signup_code() -> str:
    """When LEDGER_SIGNUP_CODE is set, creating an account needs it."""
    return _normal_code(os.environ.get("LEDGER_SIGNUP_CODE", ""))


def _too_many_failures(email: str) -> bool:
    now = time.time()
    with _failed_lock:
        recent = [t for t in _failed_logins.get(email, []) if now - t < _FAILURE_WINDOW]
        _failed_logins[email] = recent
        return len(recent) >= _MAX_FAILURES


def _record_failure(email: str) -> None:
    with _failed_lock:
        _failed_logins.setdefault(email, []).append(time.time())


# Password resets: a signed link, emailed (web/mail.py), that works for an hour
# and only until the password is changed. A few per email per hour at most.
_reset_tokens = URLSafeTimedSerializer(app.secret_key, salt="ledger-password-reset")
_RESET_MINUTES = 60
_reset_requests: dict[str, list[float]] = {}
_MAX_RESETS, _RESET_WINDOW = 3, 60 * 60


def _password_mark(user: sqlite3.Row) -> str:
    """Changes whenever the password does, without giving anything of it away."""
    return hmac.new(app.secret_key.encode(), user["password_hash"].encode(), "sha256").hexdigest()[:16]


def _reset_token(user: sqlite3.Row) -> str:
    return _reset_tokens.dumps({"uid": user["id"], "mark": _password_mark(user)})


def _user_for_reset(token: str) -> sqlite3.Row | None:
    """The user a reset link is for, while it is fresh and unused; else None."""
    try:
        data = _reset_tokens.loads(token, max_age=_RESET_MINUTES * 60)
        uid, mark = data["uid"], str(data["mark"])
    except (BadSignature, KeyError, TypeError):
        return None
    user = get_db().execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    if user is None or not hmac.compare_digest(_password_mark(user), mark):
        return None
    return user


def _too_many_resets(email: str) -> bool:
    now = time.time()
    with _failed_lock:
        recent = [t for t in _reset_requests.get(email, []) if now - t < _RESET_WINDOW] + [now]
        _reset_requests[email] = recent
        return len(recent) > _MAX_RESETS


def _public_url(endpoint: str, **values) -> str:
    """An absolute link for an email: LEDGER_PUBLIC_URL (a custom domain) if set,
    else the Space's own address -- never the Host header a request claims."""
    base = os.environ.get("LEDGER_PUBLIC_URL", "").strip().rstrip("/")
    if not base and os.environ.get("SPACE_HOST"):
        base = "https://" + os.environ["SPACE_HOST"].split(",")[0].strip()
    return base + url_for(endpoint, **values) if base else url_for(endpoint, _external=True, **values)


def _send_email(to: str, built: tuple[str, str, str]) -> None:
    """Send an email from web/emails.py in the background, so no page waits on it."""
    subject, html, text = built
    threading.Thread(target=mail.send, args=(to, subject, text, html), daemon=True).start()


def _email_images() -> str:
    """Where the emails' design blocks are served from (web/static/email)."""
    return _public_url("static", filename="email")


def _check_login(email: str, password: str) -> tuple[sqlite3.Row | None, str | None]:
    """The user for *email* and *password*, or ``(None, message)``."""
    email = email.strip().lower()
    if _too_many_failures(email):
        return None, tr("Too many wrong passwords. Wait 15 minutes and try again.")
    user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if user is None or not check_password_hash(user["password_hash"], password):
        _record_failure(email)
        return None, tr("That email and password don't match an account.")
    with _failed_lock:
        _failed_logins.pop(email, None)
    return user, None


def _user_from_request() -> dict | None:
    """The signed-in user from the session cookie or an API token."""
    uid, key = session.get("uid"), session.get("key")
    auth = request.headers.get("Authorization", "")
    if uid is None and auth.startswith("Bearer "):
        try:
            data = _api_tokens.loads(auth[len("Bearer "):], max_age=_API_TOKEN_DAYS * 24 * 3600)
            uid, key = data.get("uid"), data.get("key")
        except (BadSignature, AttributeError):
            return None
    if uid is None or not key:
        return None
    # The session key also guards against a stale cookie meeting a reused id.
    row = get_db().execute(
        "SELECT u.id, u.email, u.session_key, u.name, u.position, u.org_id, u.org_admin, o.name AS org_name "
        "FROM users u LEFT JOIN organisations o ON o.id = u.org_id WHERE u.id = ?", (uid,)).fetchone()
    if row is None or not hmac.compare_digest(row["session_key"], str(key)):
        return None
    in_org = row["org_name"] is not None
    return {"id": row["id"], "email": row["email"], "name": row["name"] or "", "position": row["position"] or "",
            "org_id": row["org_id"] if in_org else None, "org_name": row["org_name"],
            "org_admin": in_org and bool(row["org_admin"])}


def _wants_json() -> bool:
    return (request.path.startswith("/api/") or request.endpoint in ("upload", "progress")
            or request.headers.get("X-Requested-With") == "XMLHttpRequest")


@app.before_request
def require_sign_in():
    g.user = _user_from_request()
    if (g.user is not None and g.user["org_id"] and not g.user["position"] and request.method == "GET"
            and not _wants_json() and request.endpoint not in _BEFORE_WELCOME):
        return redirect(url_for("welcome", next=request.full_path.rstrip("?")))
    if g.user is not None or request.endpoint is None or request.endpoint in _PUBLIC_ENDPOINTS:
        return None
    if _wants_json():
        return jsonify({"error": tr("Please sign in first.")}), 401
    flash(tr("Please sign in to continue."))
    target = request.full_path.rstrip("?") if request.method == "GET" else url_for("index")
    return redirect(url_for("login", next=target))


@app.context_processor
def inject_user() -> dict:
    return {"user": g.get("user")}


def _safe_target(target: str | None) -> str:
    """*target* if it is a path on this site, else the home page."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return url_for("index")
    return target


def _sign_in(user: sqlite3.Row):
    session.clear()
    session.permanent = True
    session["uid"], session["key"] = user["id"], user["session_key"]


# ---------------------------------------------------------------------------
# Organisations -- a business's staff join with its code (when creating their
# account, or on the Account page), say their name and position, and then all
# see and edit its documents. Admins see the code and can change it, rename
# the organisation, and make others admin or remove them. Documents saved in
# an organisation belong to it: someone who leaves stops seeing them, and
# they stay for everyone else.
# ---------------------------------------------------------------------------

# Suggested for "your position"; any position can be typed.
POSITIONS = ("Owner", "Manager", "Accountant", "Bookkeeper", "Sales clerk", "Cashier", "Stock controller",
             "Supervisor")

# Wrong codes are counted per visitor and across everyone, so a code can't be
# found by trying many.
_code_failures: dict[str, list[float]] = {}
_MAX_CODE_FAILURES, _MAX_CODE_FAILURES_ALL = 10, 100


def _org_code(text: str) -> str:
    """A code as stored: capitals and digits only, so "delta 2026" is DELTA2026."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())[:24]


def _visitor() -> str:
    return (request.access_route[0] if request.access_route else request.remote_addr) or "?"


def _too_many_code_failures() -> bool:
    now = time.time()
    with _failed_lock:
        for key in list(_code_failures):
            _code_failures[key] = [t for t in _code_failures[key] if now - t < _FAILURE_WINDOW]
            if not _code_failures[key]:
                del _code_failures[key]
        mine = len(_code_failures.get(_visitor(), []))
        everyone = sum(len(times) for times in _code_failures.values())
    return mine >= _MAX_CODE_FAILURES or everyone >= _MAX_CODE_FAILURES_ALL


def _record_code_failure() -> None:
    with _failed_lock:
        _code_failures.setdefault(_visitor(), []).append(time.time())


def _organisation_for_code(text: str) -> sqlite3.Row | None:
    code = _org_code(text)
    if len(code) < 4:
        return None
    return get_db().execute("SELECT * FROM organisations WHERE code = ?", (code,)).fetchone()


def _join_organisation(user_id: int, org: sqlite3.Row) -> None:
    """Put *user_id* in *org*. The first to join one with no admin becomes its admin."""
    db = get_db()
    has_admin = db.execute("SELECT 1 FROM users WHERE org_id = ? AND org_admin = 1", (org["id"],)).fetchone()
    db.execute("UPDATE users SET org_id = ?, org_admin = ? WHERE id = ?",
               (org["id"], 0 if has_admin else 1, user_id))


def _new_org_code(name: str) -> str:
    """A code for a new organisation: the first word of its name and four digits, e.g. MBARE4821."""
    words = name.split()
    word = _org_code(words[0] if words else "")[:10] or "TEAM"
    db = get_db()
    while True:
        code = f"{word}{secrets.randbelow(9000) + 1000}"
        if not db.execute("SELECT 1 FROM organisations WHERE code = ?", (code,)).fetchone():
            return code


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Create an account. One code box: an organisation's code joins it (and
    is enough to sign up); otherwise the app's sign-up code, when one is set."""
    if g.user:
        return redirect(url_for("index"))
    email = request.form.get("email", "").strip().lower()
    entered = request.form.get("code", "").strip()
    if request.method == "POST":
        password = request.form.get("password", "")
        code = _signup_code()
        org = None
        error = None
        if not _EMAIL.fullmatch(email) or len(email) > 254:
            error = tr("Enter a valid email address.")
        elif len(password) < _MIN_PASSWORD:
            error = tr("Use a password of at least {n} characters.", n=_MIN_PASSWORD)
        elif entered and _too_many_code_failures():
            error = tr("Too many wrong codes. Wait 15 minutes and try again.")
        elif entered:
            # Neither kind of code minds case, spaces or quotes around it.
            org = _organisation_for_code(entered)
            if org is None and not (code and hmac.compare_digest(_normal_code(entered), code)):
                _record_code_failure()
                error = tr("That code isn't right. Check it with whoever gave it to you.")
        elif code:
            error = tr("Enter your organisation's code, or the sign-up code from the app's owner.")
        if not error:
            db = get_db()
            try:
                db.execute(
                    "INSERT INTO users (email, password_hash, session_key, created_at) VALUES (?, ?, ?, ?)",
                    (email, generate_password_hash(password), secrets.token_hex(16),
                     datetime.now(timezone.utc).isoformat()),
                )
            except sqlite3.IntegrityError:
                error = tr("There is already an account with that email. Sign in, then join from your Account page.")
            else:
                user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
                if org is not None:
                    _join_organisation(user["id"], org)
                db.commit()
        if error:
            flash(error)
        else:
            backup.mark_dirty()
            _sign_in(user)
            if org is not None:
                flash(tr("Welcome! You've joined {org}.", org=org["name"]))
                return redirect(url_for("welcome", next=_safe_target(request.args.get("next"))))
            flash(tr("Welcome! Your account is ready, and everything you save is kept in it."))
            return redirect(_safe_target(request.args.get("next")))
    return render_template("auth.html", mode="signup", email=email, entered_code=entered,
                           code_required=bool(_signup_code()))


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(_safe_target(request.args.get("next")))
    email = request.form.get("email", "").strip().lower()
    if request.method == "POST":
        user, error = _check_login(email, request.form.get("password", ""))
        if user is not None:
            _sign_in(user)
            return redirect(_safe_target(request.args.get("next")))
        flash(error)
    return render_template("auth.html", mode="login", email=email, code_required=False)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash(tr("You're signed out."))
    return redirect(url_for("index"))


@app.route("/api/login", methods=["POST"])
def api_login():
    """For the phone app: ``{email, password}`` -> ``{"token": ...}`` for API calls."""
    body = request.get_json(silent=True) or {}
    user, error = _check_login(str(body.get("email", "")), str(body.get("password", "")))
    if user is None:
        return jsonify({"error": error}), 401
    token = _api_tokens.dumps({"uid": user["id"], "key": user["session_key"]})
    return jsonify({"token": token, "email": user["email"]})


@app.route("/forgot", methods=["GET", "POST"])
def forgot_password():
    """Email a link for choosing a new password. The reply is the same whether
    or not there is an account, so the page can't be used to find accounts.

    An address with no account gets an email saying so, with a link to sign
    up: only the owner of the address learns there is no account."""
    if g.user:
        return redirect(url_for("index"))
    email = request.form.get("email", "").strip().lower()
    sent_to = None
    if request.method == "POST" and mail.enabled():
        if not _EMAIL.fullmatch(email) or len(email) > 254:
            flash(tr("Enter a valid email address."))
        else:
            user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if not _too_many_resets(email):
                # Sent in the background, so the reply takes as long either way.
                if user is not None:
                    _send_email(email, emails.password_reset(
                        _email_images(), email=email,
                        link=_public_url("reset_password", token=_reset_token(user))))
                else:
                    _send_email(email, emails.no_account(
                        _email_images(), email=email, link=_public_url("signup"),
                        code_required=bool(_signup_code())))
            sent_to = email
    return render_template("auth.html", mode="forgot", email=email, sent_to=sent_to,
                           mail_on=mail.enabled(), code_required=False)


@app.route("/reset/<token>", methods=["GET", "POST"])
def reset_password(token: str):
    """Choose a new password from an emailed link; signs out every other device."""
    user = _user_for_reset(token)
    if user is None:
        flash(tr("That link has expired or has already been used. Ask for a new one."))
        return redirect(url_for("forgot_password"))
    if request.method == "POST":
        password = request.form.get("password", "")
        if len(password) < _MIN_PASSWORD:
            flash(tr("Use a password of at least {n} characters.", n=_MIN_PASSWORD))
        elif password != request.form.get("confirm", ""):
            flash(tr("The two passwords don't match."))
        else:
            db = get_db()
            db.execute("UPDATE users SET password_hash = ?, session_key = ? WHERE id = ?",
                       (generate_password_hash(password), secrets.token_hex(16), user["id"]))
            db.commit()
            backup.mark_dirty()
            with _failed_lock:
                _failed_logins.pop(user["email"], None)
            _sign_in(db.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone())
            if mail.enabled():
                _send_email(user["email"], emails.password_changed(
                    _email_images(), email=user["email"],
                    when=datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC").lstrip("0"),
                    link=_public_url("forgot_password")))
            flash(tr("Your password is changed and you're signed in. Any other devices were signed out."))
            return redirect(url_for("index"))
    return render_template("auth.html", mode="reset", email=user["email"], code_required=False)


def _members(org_id: int) -> list[sqlite3.Row]:
    """Everyone in an organisation, admins first, then by name."""
    return get_db().execute(
        "SELECT id, email, name, position, org_admin FROM users WHERE org_id = ? "
        "ORDER BY org_admin DESC, COALESCE(NULLIF(name, ''), email) COLLATE NOCASE", (org_id,)).fetchall()


@app.route("/account")
def account():
    """Your name and position, your organisation -- or joining or starting one -- and signing out."""
    org, members, personal = None, [], 0
    if g.user["org_id"]:
        db = get_db()
        org = db.execute("SELECT * FROM organisations WHERE id = ?", (g.user["org_id"],)).fetchone()
        members = _members(g.user["org_id"])
        personal = db.execute("SELECT COUNT(*) FROM scans WHERE user_id = ? AND org_id IS NULL",
                              (g.user["id"],)).fetchone()[0]
    return render_template("account.html", org=org, members=members, personal=personal, positions=POSITIONS)


def _save_details() -> bool:
    """Save the name and position from the form; False, with a message, if one is missing."""
    name = " ".join(request.form.get("name", "").split())[:80]
    position = " ".join(request.form.get("position", "").split())[:60]
    if g.user["org_id"] and (not name or not position):
        flash(tr("Add your name and your position, so colleagues know who scanned what."))
        return False
    db = get_db()
    db.execute("UPDATE users SET name = ?, position = ? WHERE id = ?",
               (name or None, position or g.user["position"] or None, g.user["id"]))
    db.commit()
    backup.mark_dirty()
    return True


@app.route("/welcome", methods=["GET", "POST"])
def welcome():
    """Straight after joining an organisation: your name and position, which colleagues see."""
    target = _safe_target(request.args.get("next"))
    if not g.user["org_id"]:
        return redirect(target)
    if request.method == "POST" and _save_details():
        return redirect(target)
    return render_template("welcome.html", positions=POSITIONS, next=target)


@app.route("/account/details", methods=["POST"])
def save_details():
    if _save_details():
        flash(tr("Saved your details."))
    return redirect(url_for("account"))


def _not_in_organisation() -> bool:
    """True if the signed-in user isn't in one; else says to leave first."""
    if g.user["org_id"]:
        flash(tr("Leave {org} first to join another organisation.", org=g.user["org_name"]))
        return False
    return True


@app.route("/account/join", methods=["POST"])
def join_organisation():
    """Join an organisation with its code, from the Account page."""
    if not _not_in_organisation():
        return redirect(url_for("account"))
    if _too_many_code_failures():
        flash(tr("Too many wrong codes. Wait 15 minutes and try again."))
        return redirect(url_for("account"))
    org = _organisation_for_code(request.form.get("code", ""))
    if org is None:
        _record_code_failure()
        flash(tr("That code isn't right. Check it with whoever gave it to you."))
        return redirect(url_for("account"))
    _join_organisation(g.user["id"], org)
    get_db().commit()
    backup.mark_dirty()
    flash(tr("Welcome! You've joined {org}.", org=org["name"]))
    return redirect(url_for("welcome", next=url_for("documents")))


@app.route("/account/organisation", methods=["POST"])
def create_organisation():
    """Start an organisation; its creator is its admin and gets a code to give staff."""
    if not _not_in_organisation():
        return redirect(url_for("account"))
    name = " ".join(request.form.get("name", "").split())[:80]
    if not name:
        flash(tr("Enter the organisation's name."))
        return redirect(url_for("account"))
    db = get_db()
    code = _new_org_code(name)
    cursor = db.execute("INSERT INTO organisations (name, code, created_at) VALUES (?, ?, ?)",
                        (name, code, datetime.now(timezone.utc).isoformat()))
    db.execute("UPDATE users SET org_id = ?, org_admin = 1 WHERE id = ?", (cursor.lastrowid, g.user["id"]))
    db.commit()
    backup.mark_dirty()
    flash(tr("{org} is ready. Give your staff the code {code} to join.", org=name, code=code))
    return redirect(url_for("welcome", next=url_for("account")))


@app.route("/account/leave", methods=["POST"])
def leave_organisation():
    if not g.user["org_id"]:
        return redirect(url_for("account"))
    db = get_db()
    if g.user["org_admin"]:
        others = db.execute("SELECT COUNT(*), COALESCE(SUM(org_admin), 0) FROM users WHERE org_id = ? AND id != ?",
                            (g.user["org_id"], g.user["id"])).fetchone()
        if others[0] and not others[1]:
            flash(tr("Make someone else an admin first, so {org} still has one.", org=g.user["org_name"]))
            return redirect(url_for("account"))
    db.execute("UPDATE users SET org_id = NULL, org_admin = 0 WHERE id = ?", (g.user["id"],))
    db.commit()
    backup.mark_dirty()
    flash(tr("You've left {org}. Its documents stay with it.", org=g.user["org_name"]))
    return redirect(url_for("account"))


def _colleague(user_id: int) -> sqlite3.Row | None:
    """Someone else in the signed-in admin's organisation; None if not, or not an admin."""
    if not g.user["org_admin"] or user_id == g.user["id"]:
        return None
    return get_db().execute("SELECT * FROM users WHERE id = ? AND org_id = ?",
                            (user_id, g.user["org_id"])).fetchone()


@app.route("/account/members/<int:user_id>/remove", methods=["POST"])
def remove_member(user_id: int):
    member = _colleague(user_id)
    if member is not None:
        db = get_db()
        db.execute("UPDATE users SET org_id = NULL, org_admin = 0 WHERE id = ?", (user_id,))
        db.commit()
        backup.mark_dirty()
        flash(tr("Removed {name} from {org}. If they know the code, change it so they can't join again.",
                 name=member["name"] or member["email"], org=g.user["org_name"]))
    return redirect(url_for("account"))


@app.route("/account/members/<int:user_id>/admin", methods=["POST"])
def make_admin(user_id: int):
    member = _colleague(user_id)
    if member is not None:
        db = get_db()
        db.execute("UPDATE users SET org_admin = 1 WHERE id = ?", (user_id,))
        db.commit()
        backup.mark_dirty()
        flash(tr("{name} is now an admin of {org}.", name=member["name"] or member["email"], org=g.user["org_name"]))
    return redirect(url_for("account"))


@app.route("/account/organisation/code", methods=["POST"])
def change_org_code():
    """A new code to join with (typed, or made up); the old one stops working."""
    if not g.user["org_admin"]:
        return redirect(url_for("account"))
    db = get_db()
    wanted = _org_code(request.form.get("code", ""))
    if not wanted:
        wanted = _new_org_code(g.user["org_name"])
    elif len(wanted) < 6:
        flash(tr("Use at least 6 letters and numbers for the code."))
        return redirect(url_for("account"))
    elif (db.execute("SELECT 1 FROM organisations WHERE code = ? AND id != ?", (wanted, g.user["org_id"])).fetchone()
          or wanted == _org_code(_signup_code())):
        flash(tr("That code is taken. Try another."))
        return redirect(url_for("account"))
    db.execute("UPDATE organisations SET code = ? WHERE id = ?", (wanted, g.user["org_id"]))
    db.commit()
    backup.mark_dirty()
    flash(tr("The code is now {code}. The old one no longer works.", code=wanted))
    return redirect(url_for("account"))


@app.route("/account/organisation/name", methods=["POST"])
def rename_organisation():
    name = " ".join(request.form.get("name", "").split())[:80]
    if g.user["org_admin"] and name:
        db = get_db()
        db.execute("UPDATE organisations SET name = ? WHERE id = ?", (name, g.user["org_id"]))
        db.commit()
        backup.mark_dirty()
        flash(tr("The organisation is now called {org}.", org=name))
    return redirect(url_for("account"))


def _share_with_organisation(scans: list[dict]) -> None:
    """Give the signed-in user's own documents *scans*, and their page photos, to their organisation."""
    db = get_db()
    for scan in scans:
        db.execute("UPDATE scans SET org_id = ? WHERE id = ? AND user_id = ? AND org_id IS NULL",
                   (g.user["org_id"], scan["id"], g.user["id"]))
        for name in _photo_names(scan["tables"]):
            match = _STAMP.match(name)
            if match:
                db.execute("UPDATE uploads SET org_id = ? WHERE stamp = ? AND user_id = ?",
                           (g.user["org_id"], match.group(1), g.user["id"]))
    db.commit()
    backup.mark_dirty()


@app.route("/account/share-all", methods=["POST"])
def share_all_documents():
    """Share every document saved before joining with the organisation."""
    if g.user["org_id"]:
        mine = [scan for scan in _user_scans() if scan["org_id"] is None and scan["user_id"] == g.user["id"]]
        _share_with_organisation(mine)
        flash(tr("Shared {n} document(s) with {org}.", n=len(mine), org=g.user["org_name"]))
    return redirect(url_for("account"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _unique_stem(original_filename: str) -> str:
    """Stamp-prefixed, path-safe stem so repeat uploads never collide.

    The stamp (time plus random digits) starts every file made from the
    upload, and records who owns them (see _owns_file)."""
    safe_name = secure_filename(original_filename)
    stem = Path(safe_name).stem or "upload"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{timestamp}{secrets.randbelow(10**6):06d}_{stem}"


_STAMP = re.compile(r"^(\d{8}T\d+)_")


def _owns_file(name: str) -> bool:
    """Whether the signed-in user made the upload that file *name* came from."""
    match = _STAMP.match(name or "")
    if not match or g.get("user") is None:
        return False
    row = get_db().execute("SELECT user_id FROM uploads WHERE stamp = ?", (match.group(1),)).fetchone()
    return row is not None and row["user_id"] == g.user["id"]


def _can_see_file(name: str) -> bool:
    """Whether the signed-in user may see the page photos from file *name*'s upload:
    they made it, or it belongs to their organisation (its shared documents)."""
    match = _STAMP.match(name or "")
    if not match or g.get("user") is None:
        return False
    row = get_db().execute("SELECT user_id, org_id FROM uploads WHERE stamp = ?", (match.group(1),)).fetchone()
    return row is not None and (row["user_id"] == g.user["id"]
                                or (row["org_id"] is not None and row["org_id"] == g.user["org_id"]))


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


# A scanned record book can be dozens of pages; each page read by Claude
# costs a few cents, so cap the pages read per PDF.
MAX_PDF_PAGES = _env_int("LEDGER_MAX_PDF_PAGES", 40)
# Pages the free reader (Tesseract + EasyOCR) reads per PDF. It is far slower,
# so when it only stands in for a failed Claude call it reads the first few;
# when it is the only reader (offline, no API key) it reads as many as Claude
# would. LEDGER_MAX_FALLBACK_PDF_PAGES sets it explicitly.
MAX_FALLBACK_PDF_PAGES = (_env_int("LEDGER_MAX_FALLBACK_PDF_PAGES", 5)
                          if os.environ.get("LEDGER_MAX_FALLBACK_PDF_PAGES") else None)


def _fallback_page_cap() -> int:
    if MAX_FALLBACK_PDF_PAGES:
        return MAX_FALLBACK_PDF_PAGES
    return 5 if claude_reader.is_enabled() else MAX_PDF_PAGES


def _pages_note(read: int, total: int, is_pdf: bool) -> str:
    if not is_pdf:
        return ""
    if read < total:
        return tr("first {read} of {total} pages", read=read, total=total)
    return tr("1 page") if total == 1 else tr("{n} pages", n=total)


def _save_upload(upload) -> tuple[Path | None, str | None]:
    """Validate and save *upload*; return ``(saved_path, None)`` or ``(None, error)``."""
    if upload is None or upload.filename == "":
        return None, tr("No file selected. Choose a photo or PDF and try again.")

    if not _allowed_file(upload.filename):
        return None, tr("Unsupported file type. Allowed: {types}", types=", ".join(sorted(ALLOWED_EXTENSIONS)))

    stem = _unique_stem(upload.filename)
    ext = Path(secure_filename(upload.filename)).suffix.lower()
    source_path = UPLOADS_DIR / f"{stem}{ext}"
    upload.save(str(source_path))
    db = get_db()
    db.execute("INSERT INTO uploads (stamp, user_id, created_at, org_id) VALUES (?, ?, ?, ?)",
               (_STAMP.match(stem).group(1), g.user["id"], datetime.now(timezone.utc).isoformat(), g.user["org_id"]))
    db.commit()
    return source_path, None


def _uploaded_path(token: str) -> Path | None:
    """The signed-in user's saved upload named by *token* (from /upload), or None."""
    name = secure_filename(token or "")
    if not name or name != token or not _allowed_file(name) or not _owns_file(name):
        return None
    path = UPLOADS_DIR / name
    return path if path.is_file() else None


# Light JPEGs of each page for the browser: the full renders are multi-MB
# PNGs, too heavy for phones on mobile data.
_VIEW_EDGE, _THUMB_EDGE = 1400, 240


def _page_images(page_path: Path) -> dict:
    """Write a page's viewing JPEG and thumbnail; return their file names."""
    img = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if img is None:
        return {}
    names = {}
    for kind, edge, quality in (("view", _VIEW_EDGE, 80), ("thumb", _THUMB_EDGE, 70)):
        h, w = img.shape[:2]
        scale = min(1.0, edge / max(h, w))
        small = img if scale >= 1.0 else cv2.resize(
            img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        name = f"{page_path.stem}_{kind}.jpg"
        cv2.imwrite(str(PROCESSED_DIR / name), small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        names[kind] = name
    return names


def _thumb_for(view_name: str | None) -> str | None:
    if view_name and view_name.endswith("_view.jpg"):
        thumb = view_name[: -len("_view.jpg")] + "_thumb.jpg"
        if (PROCESSED_DIR / thumb).is_file() or backup.has_image(thumb):
            return thumb
    return None


# Scans this short show each page's straightened, sharpened copy (what
# Tesseract reads) beside its table; longer ones skip the extra work.
STRAIGHTENED_MAX_PAGES = 4


def _write_straight(cleaned, stem: str) -> None:
    """A viewing-size copy of a page's cleaned image, named after its viewing JPEG."""
    h, w = cleaned.shape[:2]
    scale = min(1.0, _VIEW_EDGE / max(h, w))
    small = cleaned if scale >= 1.0 else cv2.resize(
        cleaned, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(PROCESSED_DIR / f"{stem}_straight.png"), small)


def _straight_for(view_name: str | None) -> str | None:
    """The straightened copy that goes with a page's viewing JPEG, if there is one."""
    if view_name and view_name.endswith("_view.jpg"):
        straight = view_name[: -len("_view.jpg")] + "_straight.png"
        if (PROCESSED_DIR / straight).is_file() or backup.has_image(straight):
            return straight
    return None


# ---------------------------------------------------------------------------
# Reading progress -- which pages of an upload are done, for the waiting
# screen. Kept in memory: the Space runs a single server process.
# ---------------------------------------------------------------------------

_progress: dict[str, dict] = {}
_progress_lock = threading.Lock()
_PROGRESS_TTL_SECONDS = 3600


def _progress_update(token: str | None, **fields) -> None:
    if not token:
        return
    now = time.time()
    with _progress_lock:
        for key in [k for k, v in _progress.items() if now - v["updated"] > _PROGRESS_TTL_SECONDS]:
            del _progress[key]
        entry = _progress.setdefault(token, {"stage": "preparing", "total": 0, "done": set(), "thumbs": []})
        entry.update(fields)
        entry["updated"] = now


def _progress_page_done(token: str | None, index: int) -> None:
    if not token:
        return
    with _progress_lock:
        entry = _progress.get(token)
        if entry is not None:
            entry["done"].add(index)
            entry["updated"] = time.time()


def _run_pipeline_on_upload(upload, detect_tables: bool = False) -> tuple[dict | None, str | None]:
    """Save *upload*, then run the pipeline on it (see _run_pipeline_on_file)."""
    source_path, error = _save_upload(upload)
    if error:
        return None, error
    return _run_pipeline_on_file(source_path, detect_tables)


def _run_pipeline_on_file(source_path: Path, detect_tables: bool = False,
                          token: str | None = None) -> tuple[dict | None, str | None]:
    """Run preprocess + recognition on a saved upload and return the result.

    Shared by the HTML /process route and the JSON /api/scan route so both
    surfaces run the exact same pipeline. Returns ``(artifacts, None)`` on
    success or ``(None, error_message)`` on a validation/decode failure.

    A PDF is rendered to one image per page first; every page then goes
    through the same per-image pipeline, and the first page is the preview.
    With a *token*, progress is reported page by page (see /progress).

    ``artifacts`` keys: used_engine, fallback_reason, source_file,
    raw_filename, processed_filename, pages_note, pages (each page's
    viewing JPEG and thumbnail), plus either ``tables`` (when *detect_tables*
    and Claude read the pages with their own headings) or ``rows`` +
    ``provenance`` (fixed FIELDNAMES rows).
    """
    source_file = source_path.name
    stem = source_path.stem
    ext = source_path.suffix.lower()
    _progress_update(token, stage="preparing")

    is_pdf = ext == ".pdf"
    if is_pdf:
        try:
            page_paths, total_pages = render_pages(source_path, UPLOADS_DIR, stem, MAX_PDF_PAGES)
        except PdfError as exc:
            source_path.unlink(missing_ok=True)
            return None, tr(str(exc))
    else:
        page_paths, total_pages = [source_path], 1

    short_scan = len(page_paths) <= STRAIGHTENED_MAX_PAGES

    def clean_page(page_path: Path) -> Path | None:
        img = cv2.imread(str(page_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        clean_path = PROCESSED_DIR / f"{page_path.stem}_clean.png"
        cleaned = preprocess(img, save_path=clean_path)
        if short_scan:
            _write_straight(cleaned, page_path.stem)
        return clean_path

    first_clean = clean_page(page_paths[0])
    if first_clean is None:
        source_path.unlink(missing_ok=True)
        return None, tr("Could not read that image. It may be corrupt or an unsupported format.")
    cleaned = {0: first_clean}
    if short_scan:
        cleaned.update({i: clean_page(path) for i, path in enumerate(page_paths[1:], start=1)})

    pages = [{"page": n, **_page_images(path)} for n, path in enumerate(page_paths, start=1)]
    files = {
        "source_file": source_file,
        "raw_filename": page_paths[0].name,
        "processed_filename": first_clean.name,
        "pages": pages,
    }
    _progress_update(token, stage="reading", total=len(page_paths), done=set(),
                     thumbs=[p.get("thumb") for p in pages])

    def page_done(index: int) -> None:
        _progress_page_done(token, index)

    rows = None
    fallback_reason = ""
    if claude_reader.is_enabled():
        claude_engine = f"Claude vision ({claude_reader.model_name()})"
        try:
            if detect_tables:
                tables = claude_reader.read_tables_from_pages(page_paths, on_page_done=page_done)
                if not tables:
                    tables = [{"title": "", "columns": ["Column 1", "Column 2", "Column 3"],
                               "rows": [["", "", ""]], "page": 1}]
                for table in tables:
                    table["page"] = table.get("page") or 1
                    table["image"] = pages[table["page"] - 1].get("view")
                return {**files, "tables": tables, "used_engine": claude_engine, "fallback_reason": "",
                        "pages_note": _pages_note(len(page_paths), total_pages, is_pdf)}, None
            rows = claude_reader.read_rows_from_pages(page_paths, on_page_done=page_done)
            provenance = [{} for _ in rows]
            used_engine = claude_engine
            pages_read = len(page_paths)
        except claude_reader.ClaudeReadError as exc:
            app.logger.warning("Claude read failed, falling back to OCR fusion: %s", exc)
            fallback_reason = str(exc)
    else:
        fallback_reason = "no API key is set"

    if rows is None:
        # Tesseract-on-cleaned (row structure) + EasyOCR-on-raw (clean values),
        # each engine called once per page, then fused with provenance -- which
        # engine won each field -- so the review UI can show what fusion
        # auto-corrected.
        rows, provenance = [], []
        fallback_pages = page_paths[:_fallback_page_cap()]
        _progress_update(token, total=len(fallback_pages), done=set(), fallback=True)
        for i, page_path in enumerate(fallback_pages):
            clean_path = cleaned[i] if i in cleaned else clean_page(page_path)
            if clean_path is not None:
                tesseract_text, easyocr_text = run_engines(page_path, processed_path=clean_path)
                page_rows, page_prov = fuse_from_texts(tesseract_text, easyocr_text, return_provenance=True)
                rows.extend(page_rows)
                provenance.extend(page_prov)
            page_done(i)
        used_engine = "fused (tesseract + easyocr)"
        if fallback_reason and claude_reader.is_enabled():
            used_engine += f" -- Claude unavailable: {fallback_reason}"
        pages_read = len(fallback_pages)

    if not rows:
        # Never dead-end the owner with nothing to correct -- give one blank row.
        rows = [{k: "" for k in FIELDNAMES}]
        provenance = [{}]
        used_engine = "none (manual entry)"

    return {**files, "rows": rows, "provenance": provenance, "used_engine": used_engine,
            "fallback_reason": fallback_reason,
            "pages_note": _pages_note(pages_read, total_pages, is_pdf)}, None


def _save_rows(source_file: str, rows: list[dict]) -> int:
    """Insert the signed-in user's non-blank *rows* under *source_file*; return count saved."""
    created_at = datetime.now(timezone.utc).isoformat()
    db = get_db()
    saved = 0
    for row in rows:
        clean = {field: str(row.get(field, "") or "").strip() for field in FIELDNAMES}
        if not any(clean.values()):
            continue  # skip fully-blank rows (e.g. an unused manual-entry row)
        db.execute(
            """
            INSERT INTO records (date, item, qty, price, total, source_file, created_at, user_id)
            VALUES (:date, :item, :qty, :price, :total, :source_file, :created_at, :user_id)
            """,
            {**clean, "source_file": source_file, "created_at": created_at, "user_id": g.user["id"]},
        )
        saved += 1
    db.commit()
    backup.mark_dirty()
    return saved


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    message = tr("That file is too large. The limit is {mb} MB.", mb=limit_mb)
    if request.path.startswith("/api/") or request.path == "/upload":
        return jsonify({"error": message}), 413
    flash(message)
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Saved documents (the scans table)
# ---------------------------------------------------------------------------


def _scan_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "source_file": row["source_file"],
        "engine": row["engine"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "tables": json.loads(row["tables_json"]),
        "user_id": row["user_id"],
        "org_id": row["org_id"],
        "updated_by": row["updated_by"],
    }


_SCAN_COLUMNS = "id, name, source_file, engine, tables_json, created_at, updated_at, user_id, org_id, updated_by"


def _visible() -> tuple[str, tuple]:
    """SQL for the documents the signed-in user may see: their own personal
    ones, and every one belonging to their organisation."""
    return ("((org_id IS NULL AND user_id = ?) OR (org_id IS NOT NULL AND org_id = ?))",
            (g.user["id"], g.user["org_id"] or -1))


def _load_scan(scan_id: int) -> dict | None:
    """Document *scan_id* if the signed-in user may see it, else None."""
    where, params = _visible()
    row = get_db().execute(f"SELECT {_SCAN_COLUMNS} FROM scans WHERE id = ? AND {where}",
                           (scan_id, *params)).fetchone()
    return _scan_dict(row) if row else None


def _user_scans(limit: int | None = None, mine: bool = False) -> list[dict]:
    """The documents the signed-in user may see (*mine*: only ones they scanned), most recently changed first."""
    where, params = _visible()
    sql = f"SELECT {_SCAN_COLUMNS} FROM scans WHERE {where}"
    if mine:
        sql += " AND user_id = ?"
        params += (g.user["id"],)
    sql += " ORDER BY COALESCE(updated_at, created_at) DESC, id DESC"
    if limit:
        sql += " LIMIT ?"
        params += (limit,)
    return [_scan_dict(row) for row in get_db().execute(sql, params)]


def _can_delete(scan: dict) -> bool:
    """Whoever scanned a document may delete it; in an organisation, its admins may too."""
    return scan["user_id"] == g.user["id"] or (scan["org_id"] is not None and g.user["org_admin"])


def _people(scans: list[dict]) -> dict[int, sqlite3.Row]:
    """The users who scanned or last changed *scans*, by id."""
    ids = {scan["user_id"] for scan in scans} | {scan["updated_by"] for scan in scans}
    ids = [i for i in ids if i is not None]
    if not ids:
        return {}
    rows = get_db().execute(f"SELECT id, email, name, position FROM users WHERE id IN ({','.join('?' * len(ids))})",
                            ids)
    return {row["id"]: row for row in rows}


def _person(user_id: int, people: dict[int, sqlite3.Row]) -> str:
    """A colleague's name and position, for showing who scanned or changed a document."""
    row = people.get(user_id)
    if row is None:
        return tr("a former member")
    name = row["name"] or row["email"]
    return f"{name}, {tr(row['position'])}" if row["position"] else name


def _nice_date(iso: str | None) -> str:
    """'2 Oct 2026' from a stored ISO timestamp (or today)."""
    try:
        when = datetime.fromisoformat(iso) if iso else datetime.now(timezone.utc)
    except ValueError:
        when = datetime.now(timezone.utc)
    return f"{when.day} {when:%b %Y}"


def _scan_summary(scan: dict, people: dict[int, sqlite3.Row] | None = None) -> dict:
    pages = sorted({t.get("page") or 1 for t in scan["tables"]})
    people = _people([scan]) if people is None else people
    return {
        "id": scan["id"],
        "label": document_label(scan),
        "date": _nice_date(scan["updated_at"] or scan["created_at"]),
        "pages": len(pages),
        "rows": sum(len(t["rows"]) for t in scan["tables"]),
        # Who scanned it and who changed it since, shown when it is shared.
        # "you" has sentences of its own, as it changes the grammar.
        "shared": scan["org_id"] is not None,
        "mine": scan["user_id"] == g.user["id"],
        "by": None if scan["user_id"] == g.user["id"] else _person(scan["user_id"], people),
        "changed": scan["updated_by"] not in (None, scan["user_id"]),
        "changed_by": None if scan["updated_by"] in (None, g.user["id"]) else _person(scan["updated_by"], people),
        "can_delete": _can_delete(scan),
    }


def _summaries(scans: list[dict]) -> list[dict]:
    people = _people(scans)
    return [_scan_summary(scan, people) for scan in scans]


def _known_names() -> list[str]:
    """Document names used before, newest first (suggested when naming a new one)."""
    where, params = _visible()
    rows = get_db().execute(
        f"SELECT name, MAX(id) AS newest FROM scans WHERE {where} AND name IS NOT NULL AND name != '' "
        "GROUP BY name ORDER BY newest DESC LIMIT 30", params
    )
    return [row["name"] for row in rows]


# ---------------------------------------------------------------------------
# HTML routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Upload form and recent documents; for visitors, a way to sign in."""
    if g.user is None:
        return render_template("index.html", recent=[], names=[], total_documents=0, max_pdf_pages=MAX_PDF_PAGES)
    where, params = _visible()
    total = get_db().execute(f"SELECT COUNT(*) FROM scans WHERE {where}", params).fetchone()[0]
    recent = _summaries(_user_scans(limit=5))
    return render_template("index.html", recent=recent, names=_known_names(), total_documents=total,
                           max_pdf_pages=MAX_PDF_PAGES)


@app.route("/documents")
def documents():
    """Every document the signed-in user can see; in an organisation, ``?mine=1`` for only their own."""
    mine = bool(g.user["org_id"]) and request.args.get("mine") == "1"
    return render_template("documents.html", documents=_summaries(_user_scans(mine=mine)), mine=mine)


@app.route("/lang/<code>")
def set_language(code: str):
    """Switch the interface language, then go back to the page it was chosen on."""
    response = redirect(_safe_target(request.args.get("next")))
    if code in LANGUAGES:
        response.set_cookie("lang", code, max_age=365 * 24 * 3600, samesite="Lax")
    return response


def _image_url(name: str | None) -> str | None:
    """Address of a page photo the signed-in user may see, if it exists (here or in the backup)."""
    if (name and secure_filename(name) == name and _can_see_file(name)
            and ((PROCESSED_DIR / name).is_file() or backup.has_image(name))):
        return url_for("processed_file", filename=name)
    return None


def _unsure_count(artifacts: dict) -> int:
    """Cells to check: ones with "?" for unreadable characters, plus words the
    reader marked as unsure."""
    tables = artifacts.get("tables")
    if tables:
        flagged = {(t, r, c) for t, table in enumerate(tables) for r, c in table.get("unsure") or []}
        question = {(t, r, c) for t, table in enumerate(tables) for r, row in enumerate(table["rows"])
                    for c, cell in enumerate(row) if "?" in cell}
        return len(flagged | question)
    return sum(1 for row in artifacts["rows"] for field in FIELDNAMES
               if "?" in str(row.get(field, "")) or field in (row.get("unsure") or []))


def _cells_and_rows(artifacts: dict) -> tuple[list[str], int]:
    """Every cell of a read result, and how many rows it has."""
    tables = artifacts.get("tables")
    if tables:
        return ([cell for table in tables for row in table["rows"] for cell in row],
                sum(len(table["rows"]) for table in tables))
    return [str(row.get(field, "")) for row in artifacts["rows"] for field in FIELDNAMES], len(artifacts["rows"])


def _render_review(artifacts: dict, scan: dict | None = None):
    """The editable review page, for a fresh read or a saved document (*scan*)."""
    tables = artifacts.get("tables")
    page_urls: dict[int, str] = {}
    straight_urls: dict[int, str] = {}
    views = [(page["page"], page.get("view")) for page in artifacts.get("pages") or []]
    views += [(table.get("page") or 1, table.get("image")) for table in tables or []]
    for page, view in views:
        url = _image_url(view)
        if url:
            page_urls.setdefault(page, url)
            straight = _image_url(_straight_for(view))
            if straight:
                straight_urls.setdefault(page, straight)
    if 1 not in page_urls and artifacts.get("raw_filename"):
        page_urls[1] = url_for("uploaded_file", filename=artifacts["raw_filename"])
    if len(page_urls) > STRAIGHTENED_MAX_PAGES:
        straight_urls = {}

    groups = []
    if tables:
        by_page: dict[int, list] = {}
        for t, table in enumerate(tables):
            by_page.setdefault(table.get("page") or 1, []).append((t, table))
        groups = [{"page": page, "image": page_urls.get(page), "straight": straight_urls.get(page),
                   "tables": by_page[page]} for page in sorted(by_page)]
    cells, n_rows = _cells_and_rows(artifacts)

    append_to = None
    if artifacts.get("append_to"):
        target = _load_scan(artifacts["append_to"])
        if target:
            append_to = {"id": target["id"], "label": document_label(target)}

    used_engine = artifacts["used_engine"]
    return render_template(
        "process.html",
        tables=tables,
        groups=groups,
        rows=artifacts.get("rows"),
        page_images=[(page, url, straight_urls.get(page)) for page, url in sorted(page_urls.items())],
        source_file=artifacts["source_file"],
        used_engine=used_engine,
        free_reader=used_engine.startswith(("fused", "none")),
        fallback_reason=artifacts.get("fallback_reason", ""),
        pages_note=artifacts.get("pages_note", ""),
        doc_name=(scan["name"] if scan else artifacts.get("doc_name")) or "",
        scan=scan,
        append_to=append_to,
        names=_known_names(),
        summary={"pages": max(len(groups), len(page_urls), 1), "rows": n_rows,
                 "unsure": _unsure_count(artifacts)},
    )


def _review_path(token: str) -> Path | None:
    """Where the read result for the signed-in user's upload *token* is kept, or None."""
    name = secure_filename(token or "")
    if not name or name != token or not _owns_file(name):
        return None
    return PROCESSED_DIR / f"{name}.review.json"


def _document_options() -> dict:
    """The optional name and target document sent along with an upload."""
    try:
        append_to = int(request.form.get("append_to") or 0)
    except ValueError:
        append_to = 0
    return {"doc_name": request.form.get("name", "").strip()[:120], "append_to": append_to or None}


@app.route("/process", methods=["POST"])
def process():
    """Run the pipeline and show the editable table(s).

    Plain form submit (no JavaScript): takes the file and renders the review.

    With a ``token`` from /upload (the page's script uploads first so it can
    show real upload progress): reads that file, keeps the result, and
    answers ``{"review_url": ...}`` -- or ``{"error": ...}`` -- so the script
    can open the review page as a normal page load. While it reads, the
    script polls /progress/<token> to show each page finishing.
    """
    token = request.form.get("token")
    if token:
        source_path = _uploaded_path(token)
        if source_path is None:
            return jsonify({"error": tr("That upload could not be found. Please choose the file again.")}), 404
        _progress_update(token, stage="preparing")
        try:
            artifacts, error = _run_pipeline_on_file(source_path, detect_tables=True, token=token)
        except Exception:
            _progress_update(token, stage="error", error=tr("The server hit an error while reading this file."))
            raise
        if error:
            _progress_update(token, stage="error", error=error)
            return jsonify({"error": error}), 400
        artifacts.update(_document_options())
        review_file = _review_path(token)
        review_file.write_text(json.dumps(artifacts, ensure_ascii=False), encoding="utf-8")
        # Kept in the backup with its page photos, so the emailed link still
        # opens after the Space restarts.
        views = [page.get("view") for page in artifacts.get("pages") or []]
        backup.mark_dirty([review_file] + [PROCESSED_DIR / name for view in views
                                            for name in (view, _straight_for(view)) if name])
        if mail.enabled():
            cells, n_rows = _cells_and_rows(artifacts)
            _send_email(g.user["email"], emails.scan_ready(
                _email_images(), link=_public_url("review", token=token),
                name=document_label({"name": artifacts.get("doc_name"), "source_file": artifacts["source_file"],
                                     "id": 0}),
                pages=max(len(views), 1), rows=n_rows, unsure=_unsure_count(artifacts)))
        review_url = url_for("review", token=token)
        _progress_update(token, stage="done", review_url=review_url)
        return jsonify({"review_url": review_url})

    artifacts, error = _run_pipeline_on_upload(request.files.get("image"), detect_tables=True)
    if error:
        flash(error)
        return redirect(url_for("index"))
    artifacts.update(_document_options())
    return _render_review(artifacts)


@app.route("/progress/<token>")
def progress(token: str):
    """How far the read of upload *token* has got: stage, pages done, thumbnails."""
    if not _owns_file(token):
        return jsonify({"stage": "unknown"}), 404
    with _progress_lock:
        entry = _progress.get(token)
        entry = {**entry, "done": set(entry["done"])} if entry else None
    if entry is None:
        path = _review_path(token)
        if path is not None and path.is_file():
            return jsonify({"stage": "done", "review_url": url_for("review", token=token)})
        return jsonify({"stage": "unknown"}), 404
    return jsonify({
        "stage": entry["stage"],
        "total": entry["total"],
        "done": len(entry["done"]),
        "fallback": entry.get("fallback", False),
        "pages": [
            {"thumb": url_for("processed_file", filename=thumb) if thumb else None, "done": i in entry["done"]}
            for i, thumb in enumerate(entry["thumbs"])
        ],
        "review_url": entry.get("review_url"),
        "error": entry.get("error"),
    })


@app.route("/review/<token>")
def review(token: str):
    """Show a result read earlier via /process (refresh-safe; after a restart
    it comes back from the backup, so the emailed link keeps working)."""
    path = _review_path(token)
    if path is not None and not path.is_file():
        backup.fetch_image(path.name, PROCESSED_DIR)
    if path is None or not path.is_file():
        flash(tr("That review could not be found. Please upload the file again."))
        return redirect(url_for("index"))
    return _render_review(json.loads(path.read_text(encoding="utf-8")))


@app.route("/upload", methods=["POST"])
def upload():
    """Save an upload without reading it; returns ``{"token": ...}`` for /process."""
    source_path, error = _save_upload(request.files.get("image"))
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"token": source_path.name})


@app.route("/save", methods=["POST"])
def save():
    """Persist the (possibly corrected) rows from /process into SQLite."""
    source_file = request.form.get("source_file", "").strip()
    try:
        num_rows = int(request.form.get("num_rows", "0"))
    except ValueError:
        num_rows = 0

    rows = [
        {field: request.form.get(f"{field}_{i}", "") for field in FIELDNAMES}
        for i in range(min(max(num_rows, 0), _MAX_ROWS))
    ]
    saved = _save_rows(source_file, rows)

    flash(tr("Saved {n} record(s) from {source}.", n=saved, source=source_file or tr("upload")))
    return redirect(url_for("index"))


# Bounds on the dynamic review form, so a crafted POST can't make the server
# loop over millions of fields.
_MAX_TABLES, _MAX_COLS, _MAX_ROWS, _MAX_PAGE = 200, 50, 2000, 10000
_MAX_ID = 2**62


def _form_int(name: str, upper: int) -> int:
    try:
        return max(0, min(int(request.form.get(name, "0")), upper))
    except ValueError:
        return 0


@app.route("/save-tables", methods=["POST"])
def save_tables():
    """Persist tables read with the page's own headings into the scans table.

    Saves a new document, or -- with ``scan_id`` -- the edits to a saved one,
    or -- with ``append_to`` -- adds these pages after a saved one's pages.
    """
    source_file = request.form.get("source_file", "").strip()
    engine = request.form.get("engine", "").strip()
    name = request.form.get("name", "").strip()[:120] or None
    editing = _load_scan(_form_int("scan_id", _MAX_ID)) if request.form.get("scan_id") else None
    adding_to = _load_scan(_form_int("append_to", _MAX_ID)) if request.form.get("append_to") else None

    # Pages ticked "Delete this page" are left out.
    deleted_pages = {int(key[len("delete_page_"):]) for key in request.form
                     if key.startswith("delete_page_") and key[len("delete_page_"):].isdigit()}
    tables = []
    for t in range(_form_int("num_tables", _MAX_TABLES)):
        if (_form_int(f"page_{t}", _MAX_PAGE) or 1) in deleted_pages:
            continue
        n_cols = _form_int(f"num_cols_{t}", _MAX_COLS)
        n_rows = _form_int(f"num_rows_{t}", _MAX_ROWS)
        columns = [request.form.get(f"col_{t}_{c}", "").strip() for c in range(n_cols)]
        # Cells still marked as unsure ("row:col;row:col"), kept so the
        # amber shading survives saving until someone corrects them.
        flagged = set()
        for pair in request.form.get(f"unsure_{t}", "").split(";")[: _MAX_ROWS * _MAX_COLS]:
            r_text, _, c_text = pair.partition(":")
            if r_text.isdigit() and c_text.isdigit():
                flagged.add((int(r_text), int(c_text)))
        rows, unsure = [], []
        for r in range(n_rows):
            row = [request.form.get(f"cell_{t}_{r}_{c}", "").strip() for c in range(n_cols)]
            if any(row):
                unsure += [[len(rows), c] for c in range(n_cols) if (r, c) in flagged]
                rows.append(row)
        if rows:
            image = request.form.get(f"image_{t}", "")
            tables.append({
                "title": request.form.get(f"title_{t}", "").strip(),
                "columns": [col or f"Column {i + 1}" for i, col in enumerate(columns)],
                "rows": rows,
                "page": _form_int(f"page_{t}", _MAX_PAGE) or 1,
                "image": image if _image_url(image) else None,
                "unsure": unsure,
            })

    target = editing or adding_to
    if not tables:
        if deleted_pages and editing:
            flash(tr("To remove every page, delete the document instead."))
            return redirect(url_for("delete_document", scan_id=editing["id"]))
        flash(tr("Nothing to save: every row was empty."))
        return redirect(url_for("document", scan_id=target["id"]) if target else url_for("index"))

    # Number the remaining pages 1, 2, 3... again, in their order.
    renumbered = {page: n for n, page in enumerate(sorted({t["page"] for t in tables}), start=1)}
    for table in tables:
        table["page"] = renumbered[table["page"]]

    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    n_rows = sum(len(t["rows"]) for t in tables)
    if editing:
        db.execute("UPDATE scans SET name = ?, tables_json = ?, updated_at = ?, updated_by = ? WHERE id = ?",
                   (name, json.dumps(tables, ensure_ascii=False), now, g.user["id"], editing["id"]))
        _remove_photos(_photo_names(editing["tables"]) - _photo_names(tables))
        scan_id = editing["id"]
        message = tr("Saved your changes: {rows} row(s) in {tables} table(s).", rows=n_rows, tables=len(tables))
    elif adding_to:
        offset = max((t.get("page") or 1 for t in adding_to["tables"]), default=0)
        for table in tables:
            table["page"] += offset
        new_pages = len({t["page"] for t in tables})
        db.execute("UPDATE scans SET name = ?, tables_json = ?, updated_at = ?, updated_by = ? WHERE id = ?",
                   (name or adding_to["name"], json.dumps(adding_to["tables"] + tables, ensure_ascii=False),
                    now, g.user["id"], adding_to["id"]))
        scan_id = adding_to["id"]
        message = tr("Added {pages} page(s) to {name}.", pages=new_pages,
                     name=name or document_label(adding_to))
    else:
        # Saved in an organisation, a document is shared with everyone in it.
        cursor = db.execute(
            "INSERT INTO scans (name, source_file, engine, tables_json, created_at, user_id, org_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, source_file, engine, json.dumps(tables, ensure_ascii=False), now, g.user["id"], g.user["org_id"]),
        )
        scan_id = cursor.lastrowid
        message = tr("Saved {rows} row(s) in {tables} table(s) from {source}.",
                     rows=n_rows, tables=len(tables), source=name or source_file or tr("upload"))
    db.commit()
    # Keep the database and these pages' photos in the backup.
    photos = [PROCESSED_DIR / t["image"] for t in tables if t["image"]]
    photos += [PROCESSED_DIR / thumb for t in tables if (thumb := _thumb_for(t["image"]))]
    photos += [PROCESSED_DIR / straight for t in tables if (straight := _straight_for(t["image"]))]
    backup.mark_dirty(photos)

    flash(message)
    return redirect(url_for("document", scan_id=scan_id, saved=1))


@app.route("/documents/<int:scan_id>")
def document(scan_id: int):
    """A saved document: download, share, copy, edit, or add pages to it."""
    scan = _load_scan(scan_id)
    if scan is None:
        flash(tr("That document could not be found."))
        return redirect(url_for("index"))

    pages: dict[int, dict] = {}
    for table in scan["tables"]:
        page = pages.setdefault(table.get("page") or 1, {"tables": [], "image": None, "thumb": None})
        page["tables"].append({"title": table["title"], "rows": len(table["rows"]), "cols": len(table["columns"])})
        if table.get("image") and not page["image"]:
            page["image"] = _image_url(table["image"])
            page["thumb"] = _image_url(_thumb_for(table["image"]))

    summary = _scan_summary(scan)
    return render_template(
        "document.html",
        scan=scan,
        summary=summary,
        pages=sorted(pages.items()),
        saved=request.args.get("saved") == "1",
        download_stem=_download_stem(summary["label"], scan["created_at"]),
        names=_known_names(),
        max_pdf_pages=MAX_PDF_PAGES,
    )


def _photo_names(tables: list[dict]) -> set[str]:
    """Page photos, thumbnails and straightened copies that *tables* use."""
    names = {t["image"] for t in tables if t.get("image")}
    stems = [name[: -len("_view.jpg")] for name in names if name.endswith("_view.jpg")]
    return names | {stem + "_thumb.jpg" for stem in stems} | {stem + "_straight.png" for stem in stems}


def _remove_photos(names: set[str]) -> None:
    """Delete page photos no longer used by any document, here and in the backup."""
    # Every document, not only the ones this user sees: a colleague's may use them.
    every = get_db().execute("SELECT tables_json FROM scans") if names else []
    still_used = set().union(*(_photo_names(json.loads(row[0])) for row in every))
    unused = {name for name in names - still_used if secure_filename(name) == name}
    for name in unused:
        (PROCESSED_DIR / name).unlink(missing_ok=True)
    backup.forget_images(unused)


@app.route("/documents/<int:scan_id>/share", methods=["POST"])
def share_document(scan_id: int):
    """Share one of your documents saved before joining with your organisation."""
    scan = _load_scan(scan_id)
    if scan is None:
        flash(tr("That document could not be found."))
        return redirect(url_for("index"))
    if g.user["org_id"] and scan["org_id"] is None and scan["user_id"] == g.user["id"]:
        _share_with_organisation([scan])
        flash(tr("Everyone in {org} can now see {name}.", org=g.user["org_name"], name=document_label(scan)))
    return redirect(url_for("document", scan_id=scan_id))


@app.route("/documents/<int:scan_id>/delete", methods=["GET", "POST"])
def delete_document(scan_id: int):
    """Ask, then delete a saved document and its page photos."""
    scan = _load_scan(scan_id)
    if scan is None:
        flash(tr("That document could not be found."))
        return redirect(url_for("index"))
    if not _can_delete(scan):
        flash(tr("Only the person who scanned this document, or an admin of {org}, can delete it.",
                 org=g.user["org_name"]))
        return redirect(url_for("document", scan_id=scan_id))
    label = document_label(scan)
    if request.method == "GET":
        return render_template("delete.html", scan=scan, summary=_scan_summary(scan))
    db = get_db()
    db.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    db.commit()
    _remove_photos(_photo_names(scan["tables"]))
    backup.mark_dirty()
    flash(tr("Deleted {name}.", name=label))
    return redirect(url_for("documents"))


@app.route("/documents/<int:scan_id>/edit")
def edit_document(scan_id: int):
    """Reopen a saved document in the review page to fix something."""
    scan = _load_scan(scan_id)
    if scan is None:
        flash(tr("That document could not be found."))
        return redirect(url_for("index"))
    artifacts = {"tables": scan["tables"], "source_file": scan["source_file"] or "",
                 "used_engine": scan["engine"] or ""}
    return _render_review(artifacts, scan=scan)


@app.route("/documents/<int:scan_id>/export/<fmt>")
def export_document(scan_id: int, fmt: str):
    """Download one saved document. ``?tidy=0`` keeps cells exactly as written."""
    scan = _load_scan(scan_id)
    if scan is None or fmt not in ("xlsx", "csv", "json"):
        flash(tr("That document could not be found."))
        return redirect(url_for("index"))
    stem = _download_stem(document_label(scan), scan["created_at"])
    return _download(fmt, [], _RECORD_COLUMNS, [scan], stem, tidy=request.args.get("tidy") != "0")


@app.route("/documents/<int:scan_id>/statement")
def document_statement(scan_id: int):
    """The document laid out like a bank statement (see web/statement.py).

    The page's form sets what each column holds (c0, c1, ... by heading), the
    business name, currency and opening balance; ``download=xlsx`` or ``csv``
    with the same settings gives the file.
    """
    scan = _load_scan(scan_id)
    if scan is None:
        flash(tr("That document could not be found."))
        return redirect(url_for("index"))
    columns = statement.document_columns(scan["tables"])
    for i, column in enumerate(columns):
        chosen = request.args.get(f"c{i}")
        column["role"] = chosen if chosen in statement.ROLES else column["guess"]
    label = document_label(scan)
    # A shared document is the organisation's: its name is the business's.
    default_business = g.user["org_name"] if scan["org_id"] is not None and g.user["org_name"] else label
    business = request.args.get("business", default_business).strip()[:120] or default_business
    currency = request.args.get("currency") if request.args.get("currency") in statement.CURRENCIES else "USD"
    opening_text = request.args.get("opening", "").strip()[:30]
    result = statement.build(scan["tables"], {c["key"]: c["role"] for c in columns},
                             statement.parse_amount(opening_text) or 0.0 if opening_text else 0.0)
    prepared = datetime.now()

    stem = _download_stem(f"{label} statement", scan["created_at"])
    download = request.args.get("download")
    if download == "xlsx":
        data = statement.workbook(result, business, currency, prepared)
    elif download == "csv":
        data = statement.csv_text(result).encode("utf-8-sig")
    else:
        settings = {key: value for key, value in request.args.items() if key != "download"}
        return render_template("statement.html", scan=scan, label=label, columns=columns, st=result,
                               business=business, currency=currency, opening_text=opening_text,
                               period=statement.period(result), prepared=prepared.strftime("%d/%m/%Y"),
                               settings=settings, roles=statement.ROLES, currencies=statement.CURRENCIES,
                               title_en=statement.TITLE, not_a_bank=statement.NOT_A_BANK,
                               columns_en=statement.COLUMNS)
    return send_file(io.BytesIO(data), mimetype=_MIMETYPES[download], as_attachment=True,
                     download_name=f"{stem}.{download}")


# ---------------------------------------------------------------------------
# JSON API -- used by the companion mobile app (app/). Same pipeline and
# SQLite store as the HTML routes above, just JSON in/out instead of
# server-rendered forms.
# ---------------------------------------------------------------------------


@app.route("/api/health")
def api_health():
    """Liveness probe -- the mobile app pings this to detect a cold Space waking up."""
    return jsonify({"status": "ok"})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    """Upload a ledger photo, run the pipeline, return rows + provenance as JSON."""
    artifacts, error = _run_pipeline_on_upload(request.files.get("image"))
    if error:
        return jsonify({"error": error}), 400

    return jsonify({
        "rows": artifacts["rows"],
        "provenance": artifacts["provenance"],
        "used_engine": artifacts["used_engine"],
        "pages_note": artifacts["pages_note"],
        "source_file": artifacts["source_file"],
        "raw_url": url_for("uploaded_file", filename=artifacts["raw_filename"]),
        "processed_url": url_for("processed_file", filename=artifacts["processed_filename"]),
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    """Persist (possibly corrected) rows into SQLite. Body: {source_file, rows}."""
    body = request.get_json(silent=True) or {}
    source_file = str(body.get("source_file", "")).strip()
    rows = body.get("rows", [])
    if not isinstance(rows, list):
        return jsonify({"error": "'rows' must be a list."}), 400

    saved = _save_rows(source_file, rows)
    return jsonify({"saved": saved})


_RECORD_COLUMNS = ["id", "date", "item", "qty", "price", "total", "source_file", "created_at"]
_FILENAME_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_MIMETYPES = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
    "json": "application/json",
}


def _download_stem(label: str, iso_date: str | None = None) -> str:
    """'Seed book – 2 Oct 2026': the document's name and date, safe as a file name."""
    label = _FILENAME_BAD_CHARS.sub("-", label).strip(" .") or "MSME Ledger"
    return f"{label[:80]} – {_nice_date(iso_date)}"


def _csv_text(records: list[dict], columns: list[str], scans: list[dict]) -> str:
    """Records first, then each scanned table as its own block: a label line
    (document, title and page), its header row, its rows."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    if records or not scans:
        writer.writerow(columns)
        writer.writerows([r[c] for c in columns] for r in records)
    for scan in scans:
        label = document_label(scan)
        for table in scan["tables"]:
            if buf.tell():
                writer.writerow([])
            parts = [label] + ([table["title"]] if table["title"] else [])
            writer.writerow([" - ".join(parts) + f" (page {table.get('page') or 1})"])
            writer.writerow(table["columns"])
            writer.writerows(table["rows"])
    return buf.getvalue()


def _download(fmt: str, records: list[dict], columns: list[str], scans: list[dict], stem: str, tidy: bool = True):
    if fmt == "xlsx":
        data = build_workbook(records, columns, scans, tidy=tidy)
    elif fmt == "csv":
        # With a byte-order mark, Excel opens the CSV as UTF-8.
        data = _csv_text(records, columns, scans).encode("utf-8-sig")
    else:
        data = json.dumps({"records": records, "scans": scans}, indent=2, ensure_ascii=False).encode("utf-8")
    return send_file(io.BytesIO(data), mimetype=_MIMETYPES[fmt], as_attachment=True,
                     download_name=f"{stem}.{fmt}")


@app.route("/export/<fmt>")
def export(fmt: str):
    """Download everything the signed-in user can see, as Excel, CSV or JSON
    (in an organisation, that includes its shared documents).

    Covers both the fixed-field records and the scans saved with their own
    headings. Excel puts each page of each scan on its own sheet, with an
    "All pages" sheet per multi-page document (see web/exports.py). JSON is
    {"records": [...], "scans": [...]}. A single document can be downloaded
    from its own page instead (/documents/<id>/export/<fmt>).
    """
    if fmt not in _MIMETYPES:
        flash(tr("Unknown export format '{fmt}'. Use 'xlsx', 'csv' or 'json'.", fmt=fmt))
        return redirect(url_for("index"))

    db = get_db()
    records = [dict(row) for row in db.execute(
        f"SELECT {', '.join(_RECORD_COLUMNS)} FROM records WHERE user_id = ? ORDER BY id", (g.user["id"],))]
    where, params = _visible()
    scans = [_scan_dict(row) for row in db.execute(
        f"SELECT {_SCAN_COLUMNS} FROM scans WHERE {where} ORDER BY id", params)]

    stem = _download_stem(document_label(scans[0])) if len(scans) == 1 and not records else _download_stem("MSME Ledger")
    n_rows = len(records) + sum(len(t["rows"]) for s in scans for t in s["tables"])
    flash(tr("Exported {n} row(s) as {file}.", n=n_rows, file=f"{stem}.{fmt}"))
    return _download(fmt, records, _RECORD_COLUMNS, scans, stem, tidy=request.args.get("tidy") != "0")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    """Serve one of the signed-in user's raw uploads (for previews)."""
    if not _owns_file(filename):
        return jsonify({"error": "Not found."}), 404
    return send_from_directory(str(UPLOADS_DIR), filename)


@app.route("/processed/<path:filename>")
def processed_file(filename: str):
    """Serve a page photo or straightened image the signed-in user may see.

    Photos of saved documents come back from the backup after a restart.
    Colleagues get images only, never someone's unsaved read result (.json)."""
    if not (_owns_file(filename) or (_can_see_file(filename) and filename.lower().endswith((".jpg", ".png")))):
        return jsonify({"error": "Not found."}), 404
    if not (PROCESSED_DIR / filename).is_file() and secure_filename(filename) == filename:
        backup.fetch_image(filename, PROCESSED_DIR)
    return send_from_directory(str(PROCESSED_DIR), filename)


# Bring saved data back after a restart, then keep backing it up.
backup.restore(DB_PATH)
init_db()
backup.start()
if backup.enabled and threading.current_thread() is threading.main_thread():
    def _push_and_exit(_signum, _frame):
        # One last push when the Space is stopped or replaced by a new version.
        backup.push()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _push_and_exit)


def _local_addresses() -> list[str]:
    """This computer's addresses on the local network, for phones to connect to."""
    import socket

    found: list[str] = []
    try:
        # No packet is sent; this only asks which network card would be used.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.254.254.254", 1))
            found.append(probe.getsockname()[0])
    except OSError:
        pass
    try:
        found += socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass
    return [ip for i, ip in enumerate(found) if not ip.startswith("127.") and ip not in found[:i]]


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    # 0.0.0.0:7860 matches the Hugging Face Spaces container convention, and
    # lets phones on the same Wi-Fi reach a laptop running the app offline.
    port = _env_int("PORT", 7860)
    print("\nMSME Ledger is running.")
    print(f"  On this computer:            http://localhost:{port}")
    for ip in _local_addresses():
        print(f"  On phones on the same Wi-Fi: http://{ip}:{port}")
    reader = "Claude, with the free reader as backup" if claude_reader.is_enabled() else "the free reader (offline)"
    print(f"  Reading pages with {reader}.  Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
