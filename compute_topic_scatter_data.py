#!/usr/bin/env python3
"""
Extract post text and vector representations for topic scatter visualizations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample posts and compute TF-IDF vectors for topic scatter plots."
    )
    parser.add_argument(
        "submission_file",
        type=Path,
        help="Path to a *_submissions dump file.",
    )
    parser.add_argument(
        "--output-csv",
        default=Path("topic_scatter_data.csv"),
        type=Path,
        help="Where to write the sampled posts with embeddings (default: topic_scatter_data.csv).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of posts to sample for analysis (default: 5000).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=10,
        help="Number of clusters for KMeans (default: 10).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum TF-IDF features (default: 5000).",
    )
    return parser.parse_args()


def iter_records(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_posts(file_path: Path) -> pd.DataFrame:
    rows = []
    for record in iter_records(file_path):
        title = record.get("title", "")
        selftext = record.get("selftext", "")
        body = record.get("body", "")

        text_parts = []
        if title:
            text_parts.append(title)
        for part in (selftext, body):
            if part and part not in ("[deleted]", "[removed]"):
                text_parts.append(part)

        text = "\n\n".join(text_parts).strip()
        if not text:
            continue
        rows.append(
            {
                "id": record.get("id"),
                "title": title,
                "selftext": selftext,
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.submission_file.exists():
        raise SystemExit(f"Submission file not found: {args.submission_file}")

    df = collect_posts(args.submission_file)
    if df.empty:
        raise SystemExit("No text posts found to process.")

    if len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42)

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    pca = PCA(n_components=min(50, tfidf_matrix.shape[1]))
    dense_vectors = pca.fit_transform(tfidf_matrix.toarray())

    clusterer = KMeans(n_clusters=args.clusters, random_state=42, n_init="auto")
    cluster_labels = clusterer.fit_predict(dense_vectors)

    df = df.reset_index(drop=True)
    df["pca_component_1"] = dense_vectors[:, 0]
    df["pca_component_2"] = dense_vectors[:, 1] if dense_vectors.shape[1] > 1 else 0
    df["cluster"] = cluster_labels

    df.to_csv(args.output_csv, index=False)
    print(f"Wrote topic data to {args.output_csv}")


if __name__ == "__main__":
    main()
