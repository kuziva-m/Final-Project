"""
scripts/score_free_reader.py -- measure the free reader against pages whose
correct contents have been checked (see scripts/claude_batch.py).

    python scripts/score_free_reader.py                   # the test pages
    python scripts/score_free_reader.py --split train
    python scripts/score_free_reader.py --reader claude   # how Claude's first reading did
    python scripts/score_free_reader.py --note "digits-only number columns"

For each checked page it runs the free reader as the app does (on the
full-size page), lines its table up with the correct one, and counts:

  cell accuracy   cells with writing on the page that came out exactly right
  numbers / words the same, for number and date columns and for the rest
  to fix          cells someone has to change, plus made-up rows and columns
  CER             character error rate, which also credits near misses
  rows / columns  how many the reader found

Spacing and capitals don't count as mistakes. Results go to
data/benchmark/runs/<time>/ -- summary.json, pages.csv and each page's free
reader table -- and the change since the last run is shown.

Pages nobody has saved since Claude read them are left out: scoring against
Claude's unchecked reading measures agreement with Claude, not accuracy.
--include-unchecked scores them anyway (marked *).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.table_score import score_page, summary  # noqa: E402
from scripts.benchmark_store import CLAUDE_DIR, PAGES_DIR, ROOT, RUNS_DIR, connect, correct_tables, load_index  # noqa: E402

FIXED_COLUMNS = ["Date", "Item", "Qty", "Price", "Total"]


def free_reader_tables(page_path: Path, work_dir: Path) -> list[dict]:
    """The free reader's tables for a page, as the app makes them."""
    import cv2

    from extraction.fields import FIELDNAMES
    from ocr import offline
    from preprocessing.clean import preprocess

    def clean() -> Path | None:
        img = cv2.imread(str(page_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        out = work_dir / f"{page_path.stem}_clean.png"
        preprocess(offline.shrink(img), save_path=out)
        return out

    read = offline.read_page(page_path, clean, want_table=True)
    if read.table:
        return [read.table]
    if read.rows:
        return [{"columns": FIXED_COLUMNS,
                 "rows": [[str(row.get(f, "") or "") for f in FIELDNAMES] for row in read.rows]}]
    if read.lines:
        return [{"columns": ["Text"], "rows": [[line] for line in read.lines]}]
    return []


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _previous(reader: str, split: str, before: str) -> dict | None:
    """The last run's summary for the same reader and pages, for comparison."""
    if not RUNS_DIR.is_dir():
        return None
    for run in sorted(RUNS_DIR.iterdir(), reverse=True):
        try:
            data = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if run.name < before and data.get("reader") == reader and data.get("split") == split:
            return data
    return None


def _fmt(value, unit="%") -> str:
    return "-" if value is None else f"{value}{unit}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the free reader against checked pages.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--split", choices=("test", "train", "all"), default="test", help="which pages (default test)")
    parser.add_argument("--reader", choices=("free", "claude"), default="free",
                        help="free: run the free reader now; claude: score Claude's first reading")
    parser.add_argument("--include-unchecked", action="store_true", help="also score pages nobody has checked")
    parser.add_argument("--pages", nargs="*", help="only pages whose id or name contains one of these")
    parser.add_argument("--note", default="", help="what changed, kept with the results")
    args = parser.parse_args()

    from web.app import DB_PATH

    index = load_index()
    conn = connect(DB_PATH)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}-{args.reader}-{args.split}"
    tables_dir = run_dir / "tables"

    entries = [(pid, e) for pid, e in sorted(index["pages"].items(), key=lambda kv: kv[1]["label"])
               if args.split == "all" or e["split"] == args.split]
    if args.pages:
        entries = [(pid, e) for pid, e in entries if any(p in pid or p in e["label"] for p in args.pages)]

    if args.reader == "free":
        from ocr import offline
        offline.warm_up()  # load the model first, so the first page's time is fair

    total: Counter = Counter()
    rows_out, unchecked_left_out, seconds = [], 0, []
    for page_id, entry in entries:
        truth, checked = correct_tables(conn, entry)
        if truth is None:
            print(f"  {entry['label']}: its document was deleted -- left out")
            continue
        if not checked and not args.include_unchecked:
            unchecked_left_out += 1
            continue
        tables_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        if args.reader == "free":
            read = free_reader_tables(PAGES_DIR / entry["file"], tables_dir)
        else:
            path = CLAUDE_DIR / f"{page_id}.json"
            read = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        took = time.time() - started
        seconds.append(took)
        (tables_dir / f"{page_id}.json").write_text(json.dumps(read, indent=2, ensure_ascii=False), encoding="utf-8")

        stats = score_page(truth, read)
        total.update(stats)
        page = summary(stats)
        rows_out.append({"page": entry["label"] + ("" if checked else " *"), "page_id": page_id,
                         "split": entry["split"], "checked": checked, "seconds": round(took, 1), **page})
        print(f"  {rows_out[-1]['page'][:38]:38} {_fmt(page['cell_accuracy']):>7} right, "
              f"{page['cells_to_fix_per_page']:>5} to fix, {took:5.1f}s")

    if not rows_out:
        print("No pages to score."
              + (f" {unchecked_left_out} page(s) haven't been checked yet: open each Benchmark document in the app, "
                 "fix it and Save." if unchecked_left_out else
                 " Read some with scripts/claude_batch.py first."))
        return 1

    overall = summary(total, pages=len(rows_out))
    overall["seconds_per_page"] = round(sum(seconds) / len(seconds), 1)
    result = {"reader": args.reader, "split": args.split, "note": args.note, "commit": _commit(),
              "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "unchecked_included": sum(1 for r in rows_out if not r["checked"]),
              "summary": overall, "counts": dict(total)}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (run_dir / "pages.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    previous = _previous(args.reader, args.split, run_dir.name)
    before = previous["summary"] if previous else {}
    label = "Free reader" if args.reader == "free" else "Claude's first reading"
    print(f"\n{label}: {len(rows_out)} {args.split} page(s)" + (f" -- {args.note}" if args.note else ""))
    for key, name, unit, better in (
            ("cell_accuracy", "Cells right", "%", 1), ("number_accuracy", "  numbers and dates", "%", 1),
            ("word_accuracy", "  words", "%", 1), ("cells_to_fix_per_page", "Cells to fix per page", "", -1),
            ("cer", "Character errors (CER)", "%", -1), ("rows_found", "Rows found", "%", 1),
            ("columns_found", "Columns found", "%", 1), ("headings_right", "Headings right", "%", 1),
            ("seconds_per_page", "Seconds per page", "s", -1)):
        now, then = overall.get(key), before.get(key)
        change = ""
        if now is not None and then is not None and now != then:
            delta = round(now - then, 1)
            change = (f"   {'+' if delta > 0 else ''}{delta}{' pts' if unit == '%' else unit} "
                      f"({'better' if delta * better > 0 else 'worse'})")
        print(f"  {name:26} {_fmt(now, unit):>8}{change}")
    if previous:
        print(f"  (compared with the run of {previous['when'][:16].replace('T', ' ')}"
              + (f": {previous['note']}" if previous.get("note") else "") + ")")
    if result["unchecked_included"]:
        print(f"  * {result['unchecked_included']} unchecked page(s) included: those measure agreement with Claude.")
    elif unchecked_left_out:
        print(f"  {unchecked_left_out} unchecked page(s) left out.")
    print(f"Details: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
