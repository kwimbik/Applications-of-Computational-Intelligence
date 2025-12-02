#!/usr/bin/env python3
"""
Compute per-user lifespans from a Pushshift-style Reddit comment dump.

For each user, the script records the first and last comment timestamps and
their resulting lifespan (in days) within the subreddit.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract user lifespans from a Reddit comment dump."
    )
    parser.add_argument(
        "comment_file",
        type=Path,
        help="Path to a *_comments file inside the data directory.",
    )
    parser.add_argument(
        "--output-csv",
        default="lifespan_data.csv",
        type=Path,
        help="Path to write the per-user lifespan summary (default: lifespan_data.csv).",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include authors marked as [deleted] (default: skipped).",
    )
    return parser.parse_args()


def iter_records(file_path: Path) -> Iterable[dict]:
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def to_timestamp(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def accumulate_lifespans(
    comment_file: Path, include_deleted: bool
) -> pd.DataFrame:
    stats: Dict[str, Dict[str, float | int]] = defaultdict(
        lambda: {"first": None, "last": None, "count": 0}
    )

    for record in iter_records(comment_file):
        author = record.get("author")
        if not author:
            continue
        if not include_deleted and author == "[deleted]":
            continue

        created_ts = to_timestamp(record.get("created_utc"))
        if created_ts is None:
            continue

        data = stats[author]
        if data["first"] is None or created_ts < data["first"]:
            data["first"] = created_ts
        if data["last"] is None or created_ts > data["last"]:
            data["last"] = created_ts
        data["count"] = int(data["count"]) + 1

    rows = []
    for user, data in stats.items():
        first_ts = data["first"]
        last_ts = data["last"]
        if first_ts is None or last_ts is None:
            continue
        lifespan_days = max(0.0, (last_ts - first_ts) / 86400.0)
        rows.append(
            {
                "user": user,
                "first_comment_utc": format_iso(first_ts),
                "last_comment_utc": format_iso(last_ts),
                "lifespan_days": lifespan_days,
                "comment_count": data["count"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.comment_file.exists():
        raise SystemExit(f"Comment file not found: {args.comment_file}")

    df = accumulate_lifespans(args.comment_file, include_deleted=args.include_deleted)
    if df.empty:
        raise SystemExit("No user lifespans calculated. Check input filters.")

    df.sort_values("lifespan_days", ascending=False).to_csv(args.output_csv, index=False)
    print(f"Wrote lifespan data to {args.output_csv}")


if __name__ == "__main__":
    main()
