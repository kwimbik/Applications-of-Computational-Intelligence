#!/usr/bin/env python3
"""
Visualize user lifespan survival curves with an overlaid KDE histogram.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot survival curve and KDE for user lifespans."
    )
    parser.add_argument(
        "--csv",
        default="lifespan_data.csv",
        type=Path,
        help="CSV file produced by compute_user_lifespan.py (default: lifespan_data.csv).",
    )
    parser.add_argument(
        "--output",
        default=Path("Lifespan") / "lifespan_survival.png",
        type=Path,
        help="Where to save the plot (default: Lifespan/lifespan_survival.png).",
    )
    parser.add_argument(
        "--max-days",
        type=float,
        default=None,
        help="Optional cap on lifespan days for the plot (e.g., 365).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Random sample size for users included in the plot (default: 1000).",
    )
    return parser.parse_args()


def survival_curve(lifespans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_spans = np.sort(lifespans)
    unique_spans, counts = np.unique(sorted_spans, return_counts=True)
    cumulative = np.cumsum(counts)
    survivors = lifespans.size - cumulative + counts
    survival = survivors / lifespans.size
    return unique_spans, survival


def gaussian_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(grid)
    std = np.std(values, ddof=1)
    if std == 0:
        std = 1.0
    bandwidth = 1.06 * std * (values.size ** (-1 / 5))
    bandwidth = max(bandwidth, 0.1)
    diffs = (grid[:, None] - values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs ** 2)
    density = kernel.sum(axis=1) / (values.size * bandwidth * np.sqrt(2 * np.pi))
    return density


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if "lifespan_days" not in df.columns:
        raise SystemExit("CSV missing required column 'lifespan_days'.")

    df["lifespan_days"] = pd.to_numeric(df["lifespan_days"], errors="coerce")
    df = df[df["lifespan_days"].notna() & (df["lifespan_days"] >= 0)]
    if df.empty:
        raise SystemExit("No valid lifespan data to plot.")

    if args.max_days is not None:
        df = df[df["lifespan_days"] <= args.max_days]
        if df.empty:
            raise SystemExit("All lifespans were filtered out by max-days.")

    sample_size = max(1, args.sample_size)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=None)

    lifespans = df["lifespan_days"].to_numpy(dtype=float)

    x_survival, y_survival = survival_curve(lifespans)
    kde_grid = np.linspace(0, lifespans.max(), 500)
    kde_values = gaussian_kde(lifespans, kde_grid)

    fig, ax_survival = plt.subplots(figsize=(12, 8))

    ax_survival.plot(
        x_survival,
        y_survival,
        color="#1d4ed8",
        linewidth=2.5,
        label="Survival Curve",
    )
    ax_survival.set_xlabel("Lifespan since first comment (days)")
    ax_survival.set_ylabel("Survival probability")
    ax_survival.set_ylim(0, 1.05)

    max_span = float(lifespans.max())
    mid_span = max_span / 2
    tick_positions = [0.0, mid_span, max_span]
    first_dates = pd.to_datetime(df.get("first_comment_utc"), utc=True, errors="coerce")
    base_date = first_dates.min()
    tick_labels = []
    for pos in tick_positions:
        label = f"{pos:.0f} d"
        if pd.notna(base_date):
            tick_date = (base_date + pd.to_timedelta(pos, unit="D")).date()
            label += f"\n({tick_date.isoformat()})"
        tick_labels.append(label)

    ax_survival.set_xticks(tick_positions)
    ax_survival.set_xticklabels(tick_labels)
    ax_survival.grid(alpha=0.3)

    ax_density = ax_survival.twinx()
    ax_density.fill_between(
        kde_grid, kde_values, color="#f97316", alpha=0.3, label="KDE density"
    )
    ax_density.set_ylabel("Density (smoothed)")

    survival_line = ax_survival.get_lines()[0]
    density_patch = ax_density.collections[0]
    ax_survival.legend(
        [survival_line, density_patch], ["Survival Curve", "KDE density"], loc="upper right"
    )

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Wrote survival plot to {args.output}")


if __name__ == "__main__":
    main()
