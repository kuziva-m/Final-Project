"""
api/database.py — storage for digitised ledger scans.

Each scan is stored as an independent record for its business, keeping
whatever columns that business's ledger actually has — no shared schema is
enforced across businesses, since each one formats its records differently
and cross-business analysis is out of scope for now.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from supabase import Client, create_client

_client = None


def _get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set.")
        _client = create_client(url, key)
    return _client


def save_scan(
    business_id: str,
    tables: List[Dict[str, Any]],
    overall_confidence: float,
) -> Dict[str, Any]:
    """
    Store one digitised scan as an independent record.

    `tables` is the list of {columns, rows, notes} dicts as extracted,
    stored as-is in a JSONB column so each business's own table layout is
    preserved without forcing a shared schema.

    Returns the inserted row (includes the generated `id`).
    """
    client = _get_client()
    record = {
        "business_id": business_id,
        "tables": tables,
        "overall_confidence": overall_confidence,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }
    result = client.table("scans").insert(record).execute()
    return result.data[0]
