#!/usr/bin/env python3
"""
Scatter plot of user activity (comment count) vs. average score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import SymLogNorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot user activity vs. average score scatter."
    )
    parser.add_argument(
        "--csv",
        default=Path("user_scatter_data.csv"),
        type=Path,
        help="CSV produced by compute_user_scatter_data.py (default: user_scatter_data.csv).",
    )
    parser.add_argument(
        "--output",
        default=Path("user_scatter") / "user_scatter.png",
        type=Path,
        help="Where to save the plot (default: user_scatter/user_scatter.png).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Random sample size for plotted users (default: 1000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)
    required = {"user", "comment_count", "average_score"}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV missing required columns: {required}")

    df = df.dropna(subset=["comment_count", "average_score"])
    df = df[df["comment_count"] > 0]
    df = df[~df["user"].str.contains("bot", case=False, na=False)]
    if df.empty:
        raise SystemExit("No users remaining to plot after filtering.")

    avg_score = df["average_score"].mean()
    avg_comments = df["comment_count"].mean()

    sample_size = max(1, args.sample_size)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=None)

    fig, ax = plt.subplots(figsize=(12, 8))
    vmin = df["average_score"].min()
    vmax = df["average_score"].max()
    norm = SymLogNorm(linthresh=1.0, vmin=vmin, vmax=vmax, base=10)
    scatter = ax.scatter(
        df["comment_count"],
        df["average_score"],
        c=df["average_score"],
        cmap="coolwarm",
        norm=norm,
        alpha=0.6,
        edgecolor="none",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Comment count (activity, log scale)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("Average score (symlog)")
    ax.set_title(
        f"User Activity vs Average Score — mean score {avg_score:.2f}, mean comments {avg_comments:.0f}"
    )
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Average score (symlog scale)")

    caption = f"Users plotted: {len(df)}"
    ax.text(
        0.99,
        0.01,
        caption,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#4a5568",
    )

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Wrote scatter plot to {args.output}")


if __name__ == "__main__":
    main()
