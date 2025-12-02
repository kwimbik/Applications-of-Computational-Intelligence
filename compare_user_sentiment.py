#!/usr/bin/env python3
"""
Identify users who were positive in one subreddit and negative in another.

Given two Pushshift-style Reddit comment dumps, the script searches for users
who have at least one positively scored comment in the first dump and at least
one negatively scored comment in the second dump. For each such user it records
the highest-scoring positive comment and the lowest-scoring negative comment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find users with opposite sentiment across two comment files."
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
        default=Path("user_sentiment_contrast.csv"),
        help="CSV destination for matched users (default: user_sentiment_contrast.csv).",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include authors marked as [deleted] (default: skipped).",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=50,
        help="Number of users to collect per direction before stopping (default: 50).",
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


def best_positive_comments(
    file_path: Path, include_deleted: bool
) -> Dict[str, dict]:
    best: Dict[str, dict] = {}
    for record in iter_records(file_path):
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

        if score_val <= 0:
            continue

        existing = best.get(author)
        if existing is None or score_val > existing["score"]:
            best[author] = package_comment(record, score_val)
    return best


def worst_negative_comments(
    file_path: Path, include_deleted: bool
) -> Dict[str, dict]:
    worst: Dict[str, dict] = {}
    for record in iter_records(file_path):
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

        if score_val >= 0:
            continue

        existing = worst.get(author)
        if existing is None or score_val < existing["score"]:
            worst[author] = package_comment(record, score_val)
    return worst


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


def main() -> None:
    args = parse_args()
    if not args.positive_file.exists():
        raise SystemExit(f"Positive file not found: {args.positive_file}")
    if not args.negative_file.exists():
        raise SystemExit(f"Negative file not found: {args.negative_file}")
    board_a_label = args.board_a_label or args.positive_file.stem
    board_b_label = args.board_b_label or args.negative_file.stem

    positives = best_positive_comments(args.positive_file, args.include_deleted)
    negatives = worst_negative_comments(args.negative_file, args.include_deleted)
    reverse_positives = best_positive_comments(args.negative_file, args.include_deleted)
    reverse_negatives = worst_negative_comments(args.positive_file, args.include_deleted)

    def collect_pairs(pos_dict, neg_dict):
        users = []
        for user in sorted(pos_dict.keys()):
            if user in neg_dict:
                users.append(user)
                if len(users) >= args.target_count:
                    break
        return users

    forward_users = collect_pairs(positives, negatives)
    reverse_users = collect_pairs(reverse_positives, reverse_negatives)

    all_users = forward_users + reverse_users
    if not all_users:
        raise SystemExit("No users satisfied the sentiment criteria.")

    output_fields = [
        "user",
        "direction",
        "positive_board",
        "negative_board",
        "board_a_label",
        "board_b_label",
        "positive_comment_id",
        "positive_score",
        "positive_body",
        "positive_created_utc",
        "positive_link_id",
        "positive_parent_id",
        "positive_permalink",
        "negative_comment_id",
        "negative_score",
        "negative_body",
        "negative_created_utc",
        "negative_link_id",
        "negative_parent_id",
        "negative_permalink",
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=output_fields)
        writer.writeheader()

        for user in forward_users:
            pos = positives[user]
            neg = negatives[user]
            writer.writerow(
                {
                    "user": user,
                    "direction": "A->B",
                    "positive_board": "A",
                    "negative_board": "B",
                    "board_a_label": board_a_label,
                    "board_b_label": board_b_label,
                    "positive_comment_id": pos["comment_id"],
                    "positive_score": pos["score"],
                    "positive_body": pos["body"],
                    "positive_created_utc": pos["created_utc"],
                    "positive_link_id": pos["link_id"],
                    "positive_parent_id": pos["parent_id"],
                    "positive_permalink": pos.get("permalink"),
                    "negative_comment_id": neg["comment_id"],
                    "negative_score": neg["score"],
                    "negative_body": neg["body"],
                    "negative_created_utc": neg["created_utc"],
                    "negative_link_id": neg["link_id"],
                    "negative_parent_id": neg["parent_id"],
                    "negative_permalink": neg.get("permalink"),
                }
            )

        for user in reverse_users:
            pos = reverse_positives[user]
            neg = reverse_negatives[user]
            writer.writerow(
                {
                    "user": user,
                    "direction": "B->A",
                    "positive_board": "B",
                    "negative_board": "A",
                    "board_a_label": board_a_label,
                    "board_b_label": board_b_label,
                    "positive_comment_id": pos["comment_id"],
                    "positive_score": pos["score"],
                    "positive_body": pos["body"],
                    "positive_created_utc": pos["created_utc"],
                    "positive_link_id": pos["link_id"],
                    "positive_parent_id": pos["parent_id"],
                    "positive_permalink": pos.get("permalink"),
                    "negative_comment_id": neg["comment_id"],
                    "negative_score": neg["score"],
                    "negative_body": neg["body"],
                    "negative_created_utc": neg["created_utc"],
                    "negative_link_id": neg["link_id"],
                    "negative_parent_id": neg["parent_id"],
                    "negative_permalink": neg.get("permalink"),
                }
            )
    print(
        f"Wrote {len(forward_users) + len(reverse_users)} rows to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
