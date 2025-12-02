#!/usr/bin/env python3
"""
Aggregate per-user activity and scores to feed the scatter plot visualization.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-user activity and average score from a Reddit comment dump."
    )
    parser.add_argument(
        "comment_file",
        type=Path,
        help="Path to the *_comments file to aggregate.",
    )
    parser.add_argument(
        "--output-csv",
        default=Path("user_scatter_data.csv"),
        type=Path,
        help="Where to write the aggregated data (default: user_scatter_data.csv).",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include authors marked as [deleted] (default: skipped).",
    )
    return parser.parse_args()


def to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def accumulate(comment_file: Path, include_deleted: bool) -> pd.DataFrame:
    stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"comment_count": 0, "total_score": 0}
    )

    for record in iter_records(comment_file):
        author = record.get("author")
        if not author:
            continue
        if not include_deleted and author == "[deleted]":
            continue

        score_val = to_int(record.get("score"))
        if score_val == 0:
            ups = to_int(record.get("ups"))
            downs = to_int(record.get("downs"))
            score_val = ups - downs

        stats[author]["comment_count"] += 1
        stats[author]["total_score"] += score_val

    rows = []
    for user, data in stats.items():
        count = data["comment_count"]
        if count == 0:
            continue
        avg = data["total_score"] / count
        rows.append(
            {
                "user": user,
                "comment_count": count,
                "total_score": data["total_score"],
                "average_score": avg,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.comment_file.exists():
        raise SystemExit(f"Comment file not found: {args.comment_file}")

    df = accumulate(args.comment_file, include_deleted=args.include_deleted)
    if df.empty:
        raise SystemExit("No user data was aggregated.")

    df.sort_values("comment_count", ascending=False).to_csv(args.output_csv, index=False)
    print(f"Wrote scatter data to {args.output_csv}")


if __name__ == "__main__":
    main()
