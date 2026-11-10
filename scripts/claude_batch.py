"""
scripts/claude_batch.py -- read a batch of ledger photos with Claude, so the
correct contents of each page only need checking, not typing.

    python scripts/claude_batch.py photos/ --split test
    python scripts/claude_batch.py more_photos/ book.pdf --split train --email you@example.com

Each page (a PDF is split into pages) is read by Claude -- with this
computer's ANTHROPIC_API_KEY if it has one, otherwise through the online app
it is connected to on the Account page -- and saved in the app as a document
called "Benchmark test: <file> p<n>" (or "Benchmark train: ...").

Then open each one in My documents, fix anything Claude got wrong against the
photo, and Save. Saving marks the page as checked. scripts/score_free_reader.py
measures the free reader against the checked pages.

  --split test   pages kept aside for measuring. Never tune the free reader on these.
  --split train  pages for tuning and, later, training the free reader.

Pages are kept at full size in data/benchmark (never committed: they are real
business records). Running it again skips pages it has already read, so an
interrupted batch can simply be run again. Each page is one Claude request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_store import CLAUDE_DIR, PAGES_DIR, connect, load_index, save_index  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def collect(paths: list[str]) -> list[Path]:
    files = []
    for name in paths:
        path = Path(name)
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS | {".pdf"})
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTS | {".pdf"}:
            files.append(path)
        else:
            print(f"  skipped {name}: not a photo, PDF or folder")
    return files


def pick_claude():
    """``(read(page_path) -> tables, label)``: this computer's key, or the online app."""
    from ocr import claude_reader, online_reader
    from web.app import _valid_tables

    if claude_reader.is_enabled():
        return claude_reader.read_tables, f"Claude vision ({claude_reader.model_name()})"
    if online_reader.is_enabled():
        def read(page_path: Path) -> list[dict]:
            tables, _engine = online_reader.read_page(page_path, tables=True)
            return _valid_tables(tables)
        return read, f"Claude vision via {online_reader.host()}"
    sys.exit("Can't reach Claude: set ANTHROPIC_API_KEY on this computer, or connect it to the online app "
             "on the app's Account page.")


def pick_user(conn, email: str | None):
    if email:
        user = conn.execute("SELECT id, email FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
        if user is None:
            sys.exit(f"No account with the email {email} in this app.")
        return user
    users = conn.execute("SELECT id, email FROM users").fetchall()
    if len(users) != 1:
        sys.exit("Say which account the documents go in, with --email.")
    return users[0]


def pages_of(path: Path, workdir: Path, max_pages: int) -> list[tuple[str, Path]]:
    """``(label, page image)`` for each page of a photo or PDF."""
    if path.suffix.lower() != ".pdf":
        return [(path.name, path)]
    from preprocessing.pdf_pages import PdfError, render_pages

    try:
        pages, total = render_pages(path, workdir, "page", max_pages)
    except PdfError as exc:
        print(f"  skipped {path.name}: {exc}")
        return []
    if total > len(pages):
        print(f"  {path.name}: reading the first {len(pages)} of {total} pages (--max-pages)")
    return [(f"{path.name} p{n}", page) for n, page in enumerate(pages, start=1)]


def save_document(conn, user, page_id: str, page_path: Path, tables: list[dict], name: str, engine: str) -> int:
    """The page as a document in the app, owned by *user*, with its photo."""
    from web.app import _STAMP, _page_images

    view = _page_images(page_path).get("view")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("INSERT OR IGNORE INTO uploads (stamp, user_id, created_at, org_id) VALUES (?, ?, ?, NULL)",
                 (_STAMP.match(page_id).group(1), user["id"], now))
    if not tables:
        tables = [{"title": "", "columns": ["Column 1", "Column 2", "Column 3"], "rows": [["", "", ""]], "unsure": []}]
    doc_tables = [{"title": t.get("title", ""), "columns": t["columns"], "rows": t["rows"], "page": 1,
                   "image": view, "unsure": t.get("unsure") or []} for t in tables]
    cursor = conn.execute(
        "INSERT INTO scans (name, source_file, engine, tables_json, created_at, user_id, org_id) "
        "VALUES (?, ?, ?, ?, ?, ?, NULL)",
        (name, page_path.name, engine, json.dumps(doc_tables, ensure_ascii=False), now, user["id"]))
    conn.commit()
    return cursor.lastrowid


def main() -> int:
    parser = argparse.ArgumentParser(description="Read ledger photos with Claude and save them as documents to check.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("paths", nargs="+", help="photos, PDFs or folders of them")
    parser.add_argument("--split", choices=("test", "train"), required=True,
                        help="test: kept aside for measuring; train: for tuning and training")
    parser.add_argument("--email", help="the app account the documents go in (needed if there are several)")
    parser.add_argument("--name", default="Benchmark", help='start of each document\'s name (default "Benchmark")')
    parser.add_argument("--workers", type=int, default=3, help="pages read at once (default 3)")
    parser.add_argument("--max-pages", type=int, default=40, help="pages read per PDF (default 40)")
    args = parser.parse_args()

    from ocr.claude_reader import ClaudeReadError
    from ocr.online_reader import OnlineReadError
    from web.app import DB_PATH, _unique_stem

    read, engine = pick_claude()
    conn = connect(DB_PATH)
    user = pick_user(conn, args.email)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    CLAUDE_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index()
    seen = {entry["sha1"] for entry in index["pages"].values()}

    todo = []
    with tempfile.TemporaryDirectory() as tmp:
        for n_file, path in enumerate(collect(args.paths)):
            workdir = Path(tmp) / str(n_file)
            workdir.mkdir()
            for n, (label, page) in enumerate(pages_of(path, workdir, args.max_pages), start=1):
                sha1 = hashlib.sha1(page.read_bytes()).hexdigest()
                if sha1 in seen:
                    print(f"  already read: {label}")
                    continue
                seen.add(sha1)
                page_id = _unique_stem(f"{path.stem}_p{n}")
                dest = PAGES_DIR / f"{page_id}{page.suffix.lower()}"
                shutil.copyfile(page, dest)
                todo.append({"page_id": page_id, "path": dest, "label": label, "sha1": sha1, "source": str(path)})

    if not todo:
        print("Nothing new to read.")
        return 0
    print(f"Reading {len(todo)} page(s) with {engine} ...")
    done, failed = 0, []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(read, item["path"]): item for item in todo}
        for future in as_completed(futures):
            item = futures[future]
            try:
                tables = future.result()
            except (ClaudeReadError, OnlineReadError) as exc:
                failed.append((item["label"], str(exc)))
                item["path"].unlink(missing_ok=True)  # read again on the next run
                print(f"  FAILED {item['label']}: {exc}")
                continue
            (CLAUDE_DIR / f"{item['page_id']}.json").write_text(json.dumps(tables, indent=2, ensure_ascii=False),
                                                                   encoding="utf-8")
            name = f"{args.name} {args.split}: {item['label']}"
            scan_id = save_document(conn, user, item["page_id"], item["path"], tables, name, engine)
            unsure = sum(len(t.get("unsure") or []) for t in tables)
            index["pages"][item["page_id"]] = {
                "label": item["label"], "source": item["source"], "file": item["path"].name, "sha1": item["sha1"],
                "split": args.split, "scan_id": scan_id, "read_by": engine,
                "read_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            save_index(index)
            done += 1
            rows = sum(len(t["rows"]) for t in tables)
            print(f"  read {item['label']}: {rows} rows, {unsure} cell(s) Claude wasn't sure of")

    print(f"\nRead {done} page(s); {len(failed)} failed{' (run again to retry them)' if failed else ''}.")
    if done:
        print(f'In the app, open My documents: each page is a document named "{args.name} {args.split}: ...".\n'
              "Fix anything wrong against the photo (amber cells first) and Save -- saving marks it checked.\n"
              "Then measure the free reader:  python scripts/score_free_reader.py"
              + ("" if args.split == "test" else " --split train"))
    return 1 if failed and not done else 0


if __name__ == "__main__":
    sys.exit(main())
