#!/usr/bin/env python3
"""
Aggregate per-user scores from Pushshift-style Reddit comment dumps.

For each comment file in the data directory matching "*_comments", the script
accumulates total score (net votes) and antiscore (downs) per author, writes
the full table to CSV, and generates a plot of the top/bottom 100 users by
score.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-user score/antiscore from Reddit comment dumps."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
        help='Directory containing "*_comments" files (default: data)',
    )
    parser.add_argument(
        "--output-csv",
        default="user_scores.csv",
        type=Path,
        help="Path to write aggregated CSV (default: user_scores.csv)",
    )
    parser.add_argument(
        "--output-figure",
        default=Path("score_plots/user_scores_top_bottom.png"),
        type=Path,
        help=(
            "Path to write top/bottom plot "
            "(default: score_plots/user_scores_top_bottom.png)"
        ),
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


def accumulate_scores(
    comment_files: Iterable[Path], include_deleted: bool
) -> pd.DataFrame:
    stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"score": 0, "antiscore": 0, "comment_count": 0}
    )

    for file_path in comment_files:
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

            antiscore_val = to_int(record.get("downs"))

            stats[author]["score"] += score_val
            stats[author]["antiscore"] += antiscore_val
            stats[author]["comment_count"] += 1

    df = pd.DataFrame.from_dict(stats, orient="index").reset_index()
    df.rename(columns={"index": "user"}, inplace=True)
    return df


def infer_label_size(count: int) -> int:
    if count <= 25:
        return 12
    if count <= 50:
        return 9
    if count <= 100:
        return 7
    return 6


def make_plot(df: pd.DataFrame, output_path: Path) -> None:
    sorted_df = df.sort_values("score", ascending=False)
    top = sorted_df.head(100)
    bottom = sorted_df.tail(100)
    label_size_top = infer_label_size(len(top))
    label_size_bottom = infer_label_size(len(bottom))

    fig, axes = plt.subplots(2, 1, figsize=(12, 18))

    axes[0].barh(top["user"], top["score"], color="#2c7a7b")
    axes[0].invert_yaxis()
    axes[0].set_title("Top 100 Users by Score")
    axes[0].set_xlabel("Score")
    axes[0].tick_params(axis="y", labelsize=label_size_top)

    axes[1].barh(bottom["user"], bottom["score"], color="#c53030")
    axes[1].invert_yaxis()
    axes[1].set_title("Bottom 100 Users by Score")
    axes[1].set_xlabel("Score")
    axes[1].tick_params(axis="y", labelsize=label_size_bottom)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    comment_files = sorted(args.data_dir.glob("*_comments"))
    if not comment_files:
        raise SystemExit(f"No comment files found in {args.data_dir}")

    df = accumulate_scores(comment_files, include_deleted=args.include_deleted)
    df.to_csv(args.output_csv, index=False)

    if not df.empty:
        make_plot(df, args.output_figure)
        print(f"Wrote CSV to {args.output_csv}")
        print(f"Wrote figure to {args.output_figure}")
    else:
        print("No data aggregated; CSV and figure were not written.")


if __name__ == "__main__":
    main()
