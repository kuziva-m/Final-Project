"""
ocr/online_reader.py -- read pages with Claude through the online app, on a
laptop that has no API key of its own.

The online app (the Hugging Face Space) holds the Claude key. A laptop is
connected to it once, on its Account page, with the online app's address and
an online account's email and password. The online app answers with a
sign-in token -- the same kind the phone app uses -- which is kept in
data/online_reader.json; the password is not kept. Then, for each page:

    POST <online app>/api/read?mode=tables&prepared=1
    Authorization: Bearer <token>
    body: the page as a JPEG, cropped and sized for Claude here first, so
          only a few hundred KB go over the connection

and the answer is Claude's tables (or fixed rows, with mode=rows). Each answer
carries a fresh token, so a laptop in regular use stays signed in; the token
stops working if the account's password is changed.

No internet, an expired sign-in or the account's daily page limit raise
OnlineReadError; the app then reads those pages with the free reader.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2

from preprocessing.vision_prep import prepare_for_vision

CONFIG_PATH = Path(os.environ.get("LEDGER_ONLINE_READER_CONFIG")
                   or Path(__file__).resolve().parent.parent / "data" / "online_reader.json")
# Claude takes up to a minute or so on a dense page; signing in is quick.
READ_TIMEOUT, CONNECT_TIMEOUT, STATUS_TIMEOUT = 180, 30, 5
PAGE_WORKERS = 3
NO_CONNECTION = "the online app could not be reached (no internet?)"

_lock = threading.Lock()


class OnlineReadError(RuntimeError):
    """The online app did not read the page; the message says why."""


# ---------------------------------------------------------------------------
# The connection (data/online_reader.json)
# ---------------------------------------------------------------------------


def config() -> dict | None:
    """``{"url", "email", "token", "connected_at"}``, or None when not connected."""
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("url") and data.get("token") else None


def is_enabled() -> bool:
    return config() is not None


def _save(data: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_PATH)


def host() -> str:
    cfg = config()
    return urllib.parse.urlsplit(cfg["url"]).netloc if cfg else ""


def normal_url(text: str) -> str:
    """The online app's address as typed, made usable: https:// added, and a
    Hugging Face Space's page (huggingface.co/spaces/owner/name) turned into
    the address the app itself answers on (owner-name.hf.space)."""
    url = (text or "").strip().rstrip("/")
    if not url:
        raise OnlineReadError("Enter the online app's address.")
    if "://" not in url:
        url = "https://" + url
    parts = urllib.parse.urlsplit(url)
    space = re.fullmatch(r"/spaces/([^/]+)/([^/]+)", parts.path or "")
    if parts.hostname in ("huggingface.co", "www.huggingface.co") and space:
        sub = re.sub(r"[^a-z0-9-]", "-", f"{space.group(1)}-{space.group(2)}".lower())
        return f"https://{sub}.hf.space"
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise OnlineReadError("That isn't a web address.")
    if parts.scheme == "http" and parts.hostname not in ("localhost", "127.0.0.1"):
        raise OnlineReadError("Use the https:// address, so the password and pages are sent securely.")
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


def _call(method: str, url: str, *, body: bytes | None = None, headers: dict | None = None,
          timeout: float) -> tuple[int, dict]:
    """``(status, JSON answer)``; OnlineReadError if the online app can't be reached."""
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    except (urllib.error.URLError, OSError) as exc:  # no internet, DNS, refused, timed out
        raise OnlineReadError(NO_CONNECTION) from exc
    try:
        answer = json.loads(raw or b"{}")
    except ValueError:
        answer = {}
    return status, answer if isinstance(answer, dict) else {}


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def connect(url: str, email: str, password: str) -> dict:
    """Sign in to the online app at *url* and keep its token. Returns its
    reading status (``{"claude", "daily_pages", "read_today"}``)."""
    url = normal_url(url)
    status, answer = _call("POST", url + "/api/login", body=json.dumps({"email": email, "password": password}).encode(),
                           headers={"Content-Type": "application/json"}, timeout=CONNECT_TIMEOUT)
    if status == 401:
        raise OnlineReadError(answer.get("error") or "That email and password don't match an account there.")
    if status != 200 or not answer.get("token"):
        raise OnlineReadError(f"The online app answered with an error ({status}). Check the address.")
    token = answer["token"]
    status, info = _call("GET", url + "/api/read/status", headers=_auth(token), timeout=CONNECT_TIMEOUT)
    if status == 404:
        raise OnlineReadError("That online app can't read pages for laptops yet. Update it first.")
    if status != 200:
        raise OnlineReadError(f"The online app answered with an error ({status}).")
    if not info.get("claude"):
        raise OnlineReadError("The online app has no Claude key set, so it can't read pages.")
    _save({"url": url, "email": answer.get("email") or email.strip().lower(), "token": info.get("token") or token,
           "connected_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return info


def disconnect() -> None:
    with _lock:
        CONFIG_PATH.unlink(missing_ok=True)


def _keep_token(cfg: dict, token: str | None) -> None:
    """Store the fresh token an answer carried, unless the laptop was
    disconnected or connected elsewhere meanwhile."""
    if not token or token == cfg["token"]:
        return
    with _lock:
        current = config()
        if current and current["url"] == cfg["url"] and current["email"] == cfg["email"]:
            _save({**current, "token": token})


def status() -> dict | None:
    """The online app's reading status for this laptop's account, or None if
    it can't be reached right now. Raises OnlineReadError if the sign-in has
    expired."""
    cfg = config()
    if cfg is None:
        return None
    try:
        code, info = _call("GET", cfg["url"] + "/api/read/status", headers=_auth(cfg["token"]), timeout=STATUS_TIMEOUT)
    except OnlineReadError:
        return None
    if code == 401:
        raise OnlineReadError("The online sign-in has expired. Connect again.")
    if code != 200:
        return None
    _keep_token(cfg, info.get("token"))
    return info


# ---------------------------------------------------------------------------
# Reading pages
# ---------------------------------------------------------------------------


def _encode(page_path: Path) -> bytes:
    img = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if img is None:
        raise OnlineReadError("the page image could not be opened")
    ok, buf = cv2.imencode(".jpg", prepare_for_vision(img), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise OnlineReadError("the page image could not be prepared")
    return buf.tobytes()


def read_page(page_path: Path, tables: bool = True) -> tuple[list, str]:
    """Claude's reading of one page through the online app: ``(tables or
    rows, engine)``. Raises OnlineReadError with the reason when it fails."""
    cfg = config()
    if cfg is None:
        raise OnlineReadError("this computer isn't connected to the online app")
    mode = "tables" if tables else "rows"
    status, answer = _call("POST", f"{cfg['url']}/api/read?mode={mode}&prepared=1", body=_encode(page_path),
                           headers={**_auth(cfg["token"]), "Content-Type": "image/jpeg"}, timeout=READ_TIMEOUT)
    if status == 200 and isinstance(answer.get(mode), list):
        _keep_token(cfg, answer.get("token"))
        return answer[mode], f"{answer.get('engine') or 'Claude vision'} via {urllib.parse.urlsplit(cfg['url']).netloc}"
    if status == 401:
        raise OnlineReadError("the online sign-in has expired; connect again on the Account page")
    if status == 404:
        raise OnlineReadError("the online app can't read pages for laptops yet (it needs updating)")
    raise OnlineReadError(answer.get("error") or f"the online app answered with error {status}")


def read_pages(page_paths: list[Path], tables: bool = True, on_page_done=None) -> tuple[list, str]:
    """Every page through the online app, a few at a time: ``(results, engine)``
    where each result is that page's tables (or rows) or the OnlineReadError
    it failed with. Once the online app can't be reached, the pages not yet
    sent fail straight away instead of each waiting to time out."""
    unreachable = threading.Event()
    engines: list[str] = []

    def run(index: int, path: Path):
        try:
            if unreachable.is_set():
                raise OnlineReadError(NO_CONNECTION)
            result, engine = read_page(path, tables)
            engines.append(engine)
        except OnlineReadError as exc:
            if str(exc) == NO_CONNECTION:
                unreachable.set()
            result = exc
        if on_page_done:
            on_page_done(index)
        return result

    with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as pool:
        results = list(pool.map(run, range(len(page_paths)), page_paths))
    return results, (engines[0] if engines else "")
