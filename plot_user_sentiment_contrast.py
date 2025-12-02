#!/usr/bin/env python3
"""
Visualize contrasting user sentiment across two boards as a bipartite graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch


BOARD_COLORS = {
    "A": {"positive": "#1d4ed8", "negative": "#7c3aed"},
    "B": {"positive": "#ef4444", "negative": "#16a34a"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot bipartite sentiment graph from user_sentiment_contrast.csv."
    )
    parser.add_argument(
        "--csv",
        default=Path("user_sentiment_contrast.csv"),
        type=Path,
        help="CSV produced by compare_user_sentiment.py.",
    )
    parser.add_argument(
        "--output",
        default=Path("sentiment_bipartite.png"),
        type=Path,
        help="Path to save the visualization (default: sentiment_bipartite.png).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows/users to plot.",
    )
    parser.add_argument(
        "--curve-rad",
        type=float,
        default=0.15,
        help="Curvature radius for connecting arcs (default: 0.15).",
    )
    return parser.parse_args()


def node_size(score: float) -> float:
    return 80 + min(abs(score) * 10, 800)


def draw_connection(ax, start, end, curve_rad: float) -> None:
    patch = FancyArrowPatch(
        posA=start,
        posB=end,
        connectionstyle=f"arc3,rad={curve_rad}",
        arrowstyle="-",
        linewidth=1.0,
        color="black",
        alpha=0.8,
    )
    ax.add_patch(patch)


def plot_bipartite(df: pd.DataFrame, output_path: Path, curve_rad: float) -> None:
    df = df.reset_index(drop=True)
    left_x, right_x = 0.0, 1.0
    y_positions = range(len(df))
    board_a_label = df["board_a_label"].iloc[0] if "board_a_label" in df else "Board A"
    board_b_label = df["board_b_label"].iloc[0] if "board_b_label" in df else "Board B"

    fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.2)))

    for y, (_, row) in zip(y_positions, df.iterrows()):
        # Determine which entry belongs to board A / B
        if row["positive_board"] == "A":
            board_a_comment = ("positive", row["positive_score"], row["positive_body"])
            board_b_comment = ("negative", row["negative_score"], row["negative_body"])
        else:
            board_a_comment = ("negative", row["negative_score"], row["negative_body"])
            board_b_comment = ("positive", row["positive_score"], row["positive_body"])

        a_sentiment, a_score, _ = board_a_comment
        b_sentiment, b_score, _ = board_b_comment

        ax.scatter(
            left_x,
            y,
            s=node_size(a_score),
            color=BOARD_COLORS["A"][a_sentiment],
            alpha=0.7,
            edgecolors="none",
        )
        ax.scatter(
            right_x,
            y,
            s=node_size(b_score),
            color=BOARD_COLORS["B"][b_sentiment],
            alpha=0.7,
            edgecolors="none",
        )

        draw_connection(ax, (left_x, y), (right_x, y), curve_rad)
        ax.text(
            left_x - 0.02,
            y,
            f"{row['user']}",
            ha="right",
            va="center",
            fontsize=9,
            color="#4a5568",
        )

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-1, len(df))
    ax.set_xticks([left_x, right_x])
    ax.set_xticklabels([board_a_label, board_b_label], fontsize=14, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylabel("Users", fontsize=12)
    ax.set_title("Cross-board sentiment contrast", fontsize=18)
    ax.grid(axis="y", alpha=0.1)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=label,
        )
        for board, sentiments in BOARD_COLORS.items()
        for label, color in [
            (f"{board_a_label if board=='A' else board_b_label} positive", sentiments["positive"]),
            (f"{board_a_label if board=='A' else board_b_label} negative", sentiments["negative"]),
        ]
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote bipartite plot to {output_path}")


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)
    required = {
        "user",
        "positive_score",
        "negative_score",
        "positive_board",
        "negative_board",
    }
    if not required.issubset(df.columns):
        raise SystemExit("CSV missing required columns for visualization.")

    df = df.dropna(
        subset=[
            "positive_score",
            "negative_score",
            "positive_board",
            "negative_board",
        ]
    )
    if df.empty:
        raise SystemExit("CSV contains no valid rows.")

    if args.max_rows and len(df) > args.max_rows:
        df = df.head(args.max_rows)

    plot_bipartite(df, args.output, args.curve_rad)


if __name__ == "__main__":
    main()
