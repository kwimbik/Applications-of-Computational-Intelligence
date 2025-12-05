#!/usr/bin/env python3
"""
Extract post text and vector representations for topic scatter visualizations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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
    parser.add_argument(
        "--top-terms",
        type=int,
        default=10,
        help="Number of top-weighted terms to display per PCA component (default: 10).",
    )
    parser.add_argument(
        "--cluster-terms",
        type=int,
        default=5,
        help="Number of terms to summarize each cluster (default: 5).",
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


def describe_components(
    pca_model: PCA, feature_names, top_terms: int
) -> List[Dict[str, List[str]]]:
    top_terms = max(1, top_terms)
    descriptions: List[Dict[str, List[str]]] = []
    for idx, component in enumerate(pca_model.components_[:2], start=1):
        sorted_idx = component.argsort()
        top_pos = sorted_idx[-top_terms:][::-1]
        top_neg = sorted_idx[:top_terms]
        terms_pos = [feature_names[i] for i in top_pos]
        terms_neg = [feature_names[i] for i in top_neg]
        print(f"PCA component {idx}:")
        print("  High positive terms:", ", ".join(terms_pos))
        print("  High negative terms:", ", ".join(terms_neg))
        descriptions.append(
            {
                "component": idx,
                "positive_terms": terms_pos,
                "negative_terms": terms_neg,
            }
        )
    return descriptions


def describe_clusters(
    tfidf_matrix, cluster_labels, feature_names, top_terms: int
) -> Dict[int, str]:
    summaries: Dict[int, str] = {}
    top_terms = max(1, top_terms)
    for cluster in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster
        if mask.sum() == 0:
            continue
        cluster_mean = tfidf_matrix[mask].mean(axis=0).A1
        top_idx = cluster_mean.argsort()[-top_terms:][::-1]
        terms = [feature_names[i] for i in top_idx]
        label = ", ".join(terms)
        summaries[cluster] = label
        print(f"Cluster {cluster}: {label}")
    return summaries


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
    feature_names = vectorizer.get_feature_names_out()

    pca = PCA(n_components=min(50, tfidf_matrix.shape[1]))
    dense_vectors = pca.fit_transform(tfidf_matrix.toarray())
    component_descriptions = describe_components(pca, feature_names, args.top_terms)

    clusterer = KMeans(n_clusters=args.clusters, random_state=42, n_init="auto")
    cluster_labels = clusterer.fit_predict(dense_vectors)

    cluster_terms = describe_clusters(
        tfidf_matrix, cluster_labels, feature_names, args.cluster_terms
    )

    df = df.reset_index(drop=True)
    df["pca_component_1"] = dense_vectors[:, 0]
    df["pca_component_2"] = dense_vectors[:, 1] if dense_vectors.shape[1] > 1 else 0
    component_labels = {}
    for comp in component_descriptions:
        preview = 3
        pos = ", ".join(comp["positive_terms"][:preview])
        neg = ", ".join(comp["negative_terms"][:preview])
        component_labels[comp["component"]] = f"Component {comp['component']}: {pos} <-> {neg}"
    df["pca_component_1_desc"] = component_labels.get(1, "Component 1")
    df["pca_component_2_desc"] = component_labels.get(2, "Component 2")
    df["cluster"] = cluster_labels
    df["cluster_label"] = df["cluster"].map(
        lambda c: cluster_terms.get(c, f"Cluster {c}")
    )

    df.to_csv(args.output_csv, index=False)
    print(f"Wrote topic data to {args.output_csv}")


if __name__ == "__main__":
    main()
