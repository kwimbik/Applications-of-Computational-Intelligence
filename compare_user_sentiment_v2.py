#!/usr/bin/env python3
"""
Compute cross-board sentiment contrasts with activity counts (v2).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find users with opposite sentiment and track their activity (v2)."
    )
    parser.add_argument(
        "positive_file",
        type=Path,
        help="Comment dump where positive comments are searched.",
    )
    parser.add_argument(
        "negative_file",
        type=Path,
        help="Comment dump where negative comments are searched.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("user_sentiment_contrast_v2.csv"),
        help="CSV destination (default: user_sentiment_contrast_v2.csv).",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include authors marked as [deleted] (default: skipped).",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=15,
        help="Users to collect per direction (default: 15).",
    )
    parser.add_argument(
        "--board-a-label",
        type=str,
        default=None,
        help="Human-readable label for the first board (default: stem of positive_file).",
    )
    parser.add_argument(
        "--board-b-label",
        type=str,
        default=None,
        help="Human-readable label for the second board (default: stem of negative_file).",
    )
    return parser.parse_args()


def to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def iter_records(file_path: Path) -> Iterable[dict]:
    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def process_file(
    file_path: Path, include_deleted: bool
) -> Tuple[Dict[str, int], Dict[str, dict], Dict[str, dict]]:
    counts: Dict[str, int] = defaultdict(int)
    best_pos: Dict[str, dict] = {}
    worst_neg: Dict[str, dict] = {}

    for record in iter_records(file_path):
        author = record.get("author")
        if not author:
            continue
        if not include_deleted and author == "[deleted]":
            continue

        counts[author] += 1
        score_val = to_int(record.get("score"))
        if score_val == 0:
            ups = to_int(record.get("ups"))
            downs = to_int(record.get("downs"))
            score_val = ups - downs

        if score_val > 0:
            existing = best_pos.get(author)
            if existing is None or score_val > existing["score"]:
                best_pos[author] = package_comment(record, score_val)
        elif score_val < 0:
            existing = worst_neg.get(author)
            if existing is None or score_val < existing["score"]:
                worst_neg[author] = package_comment(record, score_val)

    return counts, best_pos, worst_neg


def package_comment(record: dict, score_val: int) -> dict:
    return {
        "comment_id": record.get("id"),
        "score": score_val,
        "body": record.get("body"),
        "created_utc": record.get("created_utc"),
        "link_id": record.get("link_id"),
        "parent_id": record.get("parent_id"),
        "permalink": record.get("permalink"),
    }


def collect_rows(
    source_best: Dict[str, dict],
    target_worst: Dict[str, dict],
    source_counts: Dict[str, int],
    target_counts: Dict[str, int],
    direction: str,
    positive_board: str,
    negative_board: str,
    board_a_label: str,
    board_b_label: str,
    limit: int,
) -> list:
    entries = []
    users = [
        user
        for user in source_best
        if user in target_worst and source_counts.get(user) and target_counts.get(user)
    ]
    users.sort(
        key=lambda u: abs(source_best[u]["score"] - target_worst[u]["score"]),
        reverse=True,
    )

    for user in users[:limit]:
        pos = source_best[user]
        neg = target_worst[user]
        entries.append(
            {
                "user": user,
                "direction": direction,
                "positive_board": positive_board,
                "negative_board": negative_board,
                "board_a_label": board_a_label,
                "board_b_label": board_b_label,
                "contrast": pos["score"] - neg["score"],
                "positive_comment_id": pos["comment_id"],
                "positive_score": pos["score"],
                "positive_body": pos["body"],
                "positive_created_utc": pos["created_utc"],
                "positive_link_id": pos["link_id"],
                "positive_parent_id": pos["parent_id"],
                "positive_permalink": pos.get("permalink"),
                "positive_comment_count": source_counts.get(user, 0),
                "negative_comment_id": neg["comment_id"],
                "negative_score": neg["score"],
                "negative_body": neg["body"],
                "negative_created_utc": neg["created_utc"],
                "negative_link_id": neg["link_id"],
                "negative_parent_id": neg["parent_id"],
                "negative_permalink": neg.get("permalink"),
                "negative_comment_count": target_counts.get(user, 0),
            }
        )
    return entries


def main() -> None:
    args = parse_args()
    if not args.positive_file.exists():
        raise SystemExit(f"Positive file not found: {args.positive_file}")
    if not args.negative_file.exists():
        raise SystemExit(f"Negative file not found: {args.negative_file}")

    board_a_label = args.board_a_label or args.positive_file.stem
    board_b_label = args.board_b_label or args.negative_file.stem

    counts_a, best_pos_a, worst_neg_a = process_file(
        args.positive_file, args.include_deleted
    )
    counts_b, best_pos_b, worst_neg_b = process_file(
        args.negative_file, args.include_deleted
    )

    rows = []
    rows.extend(
        collect_rows(
            best_pos_a,
            worst_neg_b,
            counts_a,
            counts_b,
            "A->B",
            board_a_label,
            board_b_label,
            board_a_label,
            board_b_label,
            args.target_count,
        )
    )
    rows.extend(
        collect_rows(
            best_pos_b,
            worst_neg_a,
            counts_b,
            counts_a,
            "B->A",
            board_b_label,
            board_a_label,
            board_a_label,
            board_b_label,
            args.target_count,
        )
    )

    if not rows:
        raise SystemExit("No users satisfied the sentiment criteria.")

    output_fields = [
        "user",
        "direction",
        "positive_board",
        "negative_board",
        "board_a_label",
        "board_b_label",
        "contrast",
        "positive_comment_id",
        "positive_score",
        "positive_body",
        "positive_created_utc",
        "positive_link_id",
        "positive_parent_id",
        "positive_permalink",
        "positive_comment_count",
        "negative_comment_id",
        "negative_score",
        "negative_body",
        "negative_created_utc",
        "negative_link_id",
        "negative_parent_id",
        "negative_permalink",
        "negative_comment_count",
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
