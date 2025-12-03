#!/usr/bin/env python3
"""
Plot PCA/UMAP scatter of posts colored by topic clusters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize topic clusters via PCA/UMAP scatter plots."
    )
    parser.add_argument(
        "--csv",
        default=Path("topic_scatter_data.csv"),
        type=Path,
        help="CSV produced by compute_topic_scatter_data.py (default: topic_scatter_data.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("topic_scatter"),
        type=Path,
        help="Directory to store generated plots (default: topic_scatter).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of posts to plot (default: 5000).",
    )
    parser.add_argument(
        "--use-umap",
        action="store_true",
        help="Use UMAP for dimensionality reduction (fallback to PCA if unavailable).",
    )
    return parser.parse_args()


def ensure_projection(df: pd.DataFrame, sample_size: int, use_umap: bool):
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=None)

    if "pca_component_1" in df.columns and "pca_component_2" in df.columns:
        coords = df[["pca_component_1", "pca_component_2"]].to_numpy()
    else:
        raise SystemExit("CSV missing PCA components. Rerun compute_topic_scatter_data.py.")

    if use_umap and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, metric="cosine")
        coords = reducer.fit_transform(coords)
    elif use_umap:
        print("UMAP not installed; falling back to PCA projection.")

    return df, coords


def plot_scatter(df: pd.DataFrame, coords: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=df["cluster"],
        cmap="tab10",
        alpha=0.7,
        s=12,
        edgecolor="none",
    )
    ax.set_title(title, fontsize=18)
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_xlabel("Component 1 (symlog)", fontsize=14)
    ax.set_ylabel("Component 2", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(alpha=0.2)

    legend_handles = []
    seen = set()
    for cluster, label in zip(df["cluster"], df["cluster_label"]):
        if cluster in seen:
            continue
        seen.add(cluster)
        color = scatter.cmap(scatter.norm(cluster))
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                label=label,
            )
        )
    ax.legend(
        handles=legend_handles,
        title="Cluster descriptors",
        loc="best",
        fontsize=12,
        title_fontsize=14,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)
    required = {"cluster", "pca_component_1", "pca_component_2", "cluster_label"}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV missing required columns: {required}")

    df = df.dropna(subset=["cluster", "pca_component_1", "pca_component_2"])
    if df.empty:
        raise SystemExit("No data rows available for plotting.")

    sampled_df, coords = ensure_projection(df, args.sample_size, args.use_umap)

    method = "UMAP" if args.use_umap and HAS_UMAP else "PCA"
    output_base = args.output_dir / f"topic_scatter_{method.lower()}.png"
    plot_scatter(sampled_df, coords, output_base, f"Topic Scatter ({method})")


if __name__ == "__main__":
    main()
