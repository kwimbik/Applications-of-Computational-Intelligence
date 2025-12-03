#!/usr/bin/env python3
"""
Scatter visualization of sentiment contrast with activity weighting (v2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIRECTION_COLORS = {
    "A->B": "#2563eb",
    "B->A": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sentiment contrast scatter from user_sentiment_contrast_v2.csv."
    )
    parser.add_argument(
        "--csv",
        default=Path("user_sentiment_contrast_v2.csv"),
        type=Path,
        help="CSV produced by compare_user_sentiment_v2.py.",
    )
    parser.add_argument(
        "--output",
        default=Path("sentiment_plots_v2") / "sentiment_contrast_v2.png",
        type=Path,
        help="Where to save the plot (default: sentiment_plots_v2/sentiment_contrast_v2.png).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows to plot.",
    )
    parser.add_argument(
        "--annotate-top",
        type=int,
        default=5,
        help="Number of most contrasting users to annotate (default: 5).",
    )
    return parser.parse_args()


def compute_sizes(df: pd.DataFrame) -> np.ndarray:
    min_counts = df[["positive_comment_count", "negative_comment_count"]].min(axis=1)
    return 80 + np.sqrt(min_counts) * 20


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)
    required = {
        "user",
        "direction",
        "positive_score",
        "negative_score",
        "positive_comment_count",
        "negative_comment_count",
        "contrast",
        "board_a_label",
        "board_b_label",
    }
    if not required.issubset(df.columns):
        raise SystemExit("CSV missing required columns for visualization.")

    df = df.dropna(
        subset=[
            "positive_score",
            "negative_score",
            "positive_comment_count",
            "negative_comment_count",
        ]
    )
    if df.empty:
        raise SystemExit("No data rows available for plotting.")

    if args.max_rows and len(df) > args.max_rows:
        df = df.head(args.max_rows)

    sizes = compute_sizes(df)
    colors = df["direction"].map(DIRECTION_COLORS).fillna("#475569")

    board_a_label = df["board_a_label"].iloc[0] if "board_a_label" in df else "Board A"
    board_b_label = df["board_b_label"].iloc[0] if "board_b_label" in df else "Board B"

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        df["positive_score"],
        df["negative_score"],
        s=sizes,
        c=colors,
        alpha=0.7,
        edgecolor="none",
    )
    ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Positive score in {board_a_label} / {board_b_label}", fontsize=13)
    ax.set_ylabel(f"Negative score in counterpart board", fontsize=13)
    ax.set_title("Sentiment Contrast vs Activity (v2)", fontsize=18)
    ax.grid(alpha=0.25)

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{direction} ({label})",
            markerfacecolor=color,
            markersize=10,
        )
        for direction, color, label in [
            ("A->B", DIRECTION_COLORS["A->B"], f"{board_a_label} positive → {board_b_label} negative"),
            ("B->A", DIRECTION_COLORS["B->A"], f"{board_b_label} positive → {board_a_label} negative"),
        ]
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    annotate_df = df.copy()
    annotate_df["abs_contrast"] = annotate_df["contrast"].abs()
    annotate_df.sort_values("abs_contrast", ascending=False, inplace=True)
    for _, row in annotate_df.head(args.annotate_top).iterrows():
        ax.text(
            row["positive_score"],
            row["negative_score"],
            row["user"],
            fontsize=9,
            color="#111827",
            ha="left",
            va="bottom",
        )

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f"Wrote sentiment scatter to {args.output}")


if __name__ == "__main__":
    main()
