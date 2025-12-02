#!/usr/bin/env python3
"""
Create a score visualization for the top/bottom N users from the aggregated CSV.

Use together with compute_user_scores.py, which generates user_scores.csv.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def infer_label_size(count: int) -> int:
    if count <= 25:
        return 12
    if count <= 50:
        return 9
    if count <= 100:
        return 7
    return 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot top/bottom-N user scores from aggregated CSV."
    )
    parser.add_argument(
        "top_n",
        type=int,
        help="Number of top/bottom users to display.",
    )
    parser.add_argument(
        "--csv",
        default="user_scores.csv",
        type=Path,
        help="CSV file with columns user, score (default: user_scores.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to score_plots/user_scores_top_bottom_<N>.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty or "score" not in df.columns or "user" not in df.columns:
        raise SystemExit("CSV missing required columns 'user' and 'score'.")

    df = df[~df["user"].str.contains("bot", case=False, na=False)]
    if df.empty:
        raise SystemExit("No users remaining after bot filtering.")
    top_n = max(args.top_n, 1)
    sorted_df = df.sort_values("score", ascending=False)
    top = sorted_df.head(top_n)
    bottom = sorted_df.tail(top_n)
    label_size_top = infer_label_size(len(top))
    label_size_bottom = infer_label_size(len(bottom))

    fig, axes = plt.subplots(2, 1, figsize=(12, 18))

    axes[0].barh(top["user"], top["score"], color="#2c7a7b")
    axes[0].invert_yaxis()
    axes[0].set_title(f"Top {top_n} Users by Score")
    axes[0].set_xlabel("Score")
    axes[0].tick_params(axis="y", labelsize=label_size_top)

    axes[1].barh(bottom["user"], bottom["score"], color="#c53030")
    axes[1].invert_yaxis()
    axes[1].set_title(f"Bottom {top_n} Users by Score")
    axes[1].set_xlabel("Score")
    axes[1].tick_params(axis="y", labelsize=label_size_bottom)

    plt.tight_layout()
    output_path = args.output or Path("score_plots") / f"user_scores_top_bottom_{top_n}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
