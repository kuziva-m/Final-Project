"""
extraction/table_score.py -- how close a read table is to the correct one.

Used by scripts/score_free_reader.py. Tables are ``{"columns": [...],
"rows": [[...], ...]}``. A reader can miss, add, merge or split rows and
columns, so the two tables are lined up first -- columns, then rows, each
kept in page order -- and every cell of the correct table is compared with
the cell it lines up with (an empty one when its row or column wasn't found).

Everything is counted from the correct table's side ("of the cells on the
page, how many came out right"), plus the rows and columns the reader made
up, each one deletion for whoever checks the table.
"""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher


def normal(text) -> str:
    """Cell text as compared: spacing and capitals don't matter."""
    return " ".join(str(text or "").split()).casefold()


def similarity(a, b) -> float:
    a, b = normal(a), normal(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def edit_distance(a, b) -> int:
    """Characters to add, remove or change to turn *a* into *b* (normalised)."""
    a, b = normal(a), normal(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def align(n: int, m: int, sim, threshold: float) -> dict[int, int]:
    """Pair items 0..n-1 with 0..m-1 in order, maximising total similarity;
    pairs less similar than *threshold* are left unpaired. ``{i: j}``."""
    sims = [[sim(i, j) for j in range(m)] for i in range(n)]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = sims[i - 1][j - 1]
            pair = best[i - 1][j - 1] + s if s >= threshold else -1.0
            best[i][j] = max(best[i - 1][j], best[i][j - 1], pair)
    pairs: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        s = sims[i - 1][j - 1]
        if s >= threshold and best[i][j] == best[i - 1][j - 1] + s:
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return pairs


def is_number_column(cells: list[str]) -> bool:
    """Mostly numbers (amounts, quantities, dates) rather than words."""
    filled = [normal(c).replace(" ", "") for c in cells if normal(c)]
    numeric = sum(1 for c in filled if 2 * sum(ch.isdigit() for ch in c) >= len(c))
    return len(filled) >= 2 and numeric >= 0.6 * len(filled)


def _shape(table: dict | None) -> tuple[list[str], list[list[str]]]:
    if not table:
        return [], []
    columns = [str(c or "") for c in table.get("columns") or []]
    rows = [([str(c or "") for c in row] + [""] * len(columns))[:len(columns)] for row in table.get("rows") or []]
    return columns, rows


def score_table(truth: dict, read: dict | None) -> Counter:
    """Counts for one correct table against one read table (or None)."""
    t_cols, t_rows = _shape(truth)
    r_cols, r_rows = _shape(read)

    def column_text(cols, rows, c):
        return cols[c] + "\n" + "\n".join(row[c] for row in rows)

    cmap = align(len(t_cols), len(r_cols),
                 lambda i, j: similarity(column_text(t_cols, t_rows, i), column_text(r_cols, r_rows, j)), 0.2)

    def row_sim(i, j):
        if not cmap:
            return 0.0
        return sum(similarity(t_rows[i][c], r_rows[j][rc]) for c, rc in cmap.items()) / len(cmap)

    rmap = align(len(t_rows), len(r_rows), row_sim, 0.3)
    kinds = ["number" if is_number_column([row[c] for row in t_rows]) else "text" for c in range(len(t_cols))]

    stats: Counter = Counter()
    for i, row in enumerate(t_rows):
        for c, want in enumerate(row):
            got = r_rows[rmap[i]][cmap[c]] if i in rmap and c in cmap else ""
            right = normal(want) == normal(got)
            stats["cells"] += 1
            stats["cells_to_fix"] += not right
            if normal(want):
                stats["filled"] += 1
                stats[f"filled_{kinds[c]}"] += 1
                stats["filled_right"] += right
                stats[f"right_{kinds[c]}"] += right
            elif normal(got):
                stats["made_up_cells"] += 1  # writing where the page has none
            stats["char_errors"] += edit_distance(want, got)
            stats["chars"] += len(normal(want))
    for c, heading in enumerate(t_cols):
        stats["headings"] += 1
        stats["headings_right"] += normal(heading) == normal(r_cols[cmap[c]] if c in cmap else "")
    stats["rows"] += len(t_rows)
    stats["rows_found"] += len(rmap)
    stats["extra_rows"] += len(r_rows) - len(set(rmap.values()))
    stats["columns"] += len(t_cols)
    stats["columns_found"] += len(cmap)
    stats["extra_columns"] += len(r_cols) - len(set(cmap.values()))
    return stats


def score_page(truth_tables: list[dict], read_tables: list[dict]) -> Counter:
    """Counts for a page: each correct table against the read table that fits
    it best; rows of read tables that fit none count as made up."""
    stats: Counter = Counter()
    used: set[int] = set()
    for truth in truth_tables:
        options = [(score_table(truth, read), k) for k, read in enumerate(read_tables)] or [(score_table(truth, None), -1)]
        best, k = max(options, key=lambda o: (o[0]["filled_right"], -o[0]["cells_to_fix"]))
        stats.update(best)
        used.add(k)
    for k, read in enumerate(read_tables):
        if k not in used:
            stats["extra_rows"] += len(_shape(read)[1])
    return stats


def summary(stats: Counter, pages: int = 1) -> dict:
    """Readable figures from summed counts (percentages 0-100)."""

    def pct(a, b):
        return round(100.0 * stats[a] / stats[b], 1) if stats[b] else None

    return {
        "pages": pages,
        "cell_accuracy": pct("filled_right", "filled"),
        "number_accuracy": pct("right_number", "filled_number"),
        "word_accuracy": pct("right_text", "filled_text"),
        "cells_to_fix_per_page": (round((stats["cells_to_fix"] + stats["extra_rows"] + stats["extra_columns"]) / pages, 1)
                                  if pages else None),
        "cells_to_fix_pct": pct("cells_to_fix", "cells"),
        "cer": round(100.0 * stats["char_errors"] / stats["chars"], 1) if stats["chars"] else None,
        "rows_found": pct("rows_found", "rows"),
        "columns_found": pct("columns_found", "columns"),
        "headings_right": pct("headings_right", "headings"),
        "made_up_rows": stats["extra_rows"],
        "made_up_columns": stats["extra_columns"],
    }
