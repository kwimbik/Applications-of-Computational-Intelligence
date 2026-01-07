#!/usr/bin/env python3
"""Community detection from Reddit-like exports (JSONL/JSON/CSV, optional .gz)."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple


def open_text(path: str) -> io.TextIOBase:
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def detect_format(path: str) -> str:
    """Return 'jsonl', 'json', or 'csv' based on leading content."""
    with open_text(path) as handle:
        first = None
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            first = stripped
            break
        if first is None:
            return "jsonl"
        lead = first[0]
        if lead == "[":
            return "json"
        if lead == "{":
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("{"):
                    return "jsonl"
                return "json"
            return "jsonl"
        return "csv"


def iter_jsonl(path: str, stats: Dict[str, int]) -> Iterator[dict]:
    with open_text(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue
            if isinstance(obj, dict):
                yield obj


def iter_json(path: str, stats: Dict[str, int]) -> Iterator[dict]:
    with open_text(path) as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            stats["json_errors"] += 1
            return
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj
    elif isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            for obj in data["data"]:
                if isinstance(obj, dict):
                    yield obj
        else:
            yield data


def iter_csv(path: str) -> Iterator[dict]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def iter_records(path: str, stats: Dict[str, int]) -> Iterator[dict]:
    fmt = detect_format(path)
    if fmt == "jsonl":
        yield from iter_jsonl(path, stats)
    elif fmt == "json":
        yield from iter_json(path, stats)
    else:
        yield from iter_csv(path)


def gather_files(inputs: List[str]) -> List[str]:
    files: List[str] = []
    for entry in inputs:
        if os.path.isdir(entry):
            for root, _, filenames in os.walk(entry):
                for name in filenames:
                    files.append(os.path.join(root, name))
        else:
            files.append(entry)
    return files


def is_probably_text(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            chunk = handle.read(1024)
    except OSError:
        return False
    return b"\x00" not in chunk


def normalize_comment_id(record: dict) -> str | None:
    name = record.get("name")
    if name and isinstance(name, str):
        return name
    cid = record.get("id")
    if cid and isinstance(cid, str):
        return f"t1_{cid}"
    return None


def normalize_submission_id(record: dict) -> str | None:
    name = record.get("name")
    if name and isinstance(name, str):
        return name
    sid = record.get("id")
    if sid and isinstance(sid, str):
        return f"t3_{sid}"
    return None


def add_edge(edges: Dict[Tuple[str, str], int], src: str, dst: str) -> None:
    if src == dst:
        return
    edges[(src, dst)] += 1


def build_graph(paths: List[str], verbose: bool) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int]]:
    edges: Dict[Tuple[str, str], int] = defaultdict(int)
    id_to_author: Dict[str, str] = {}
    pending: Dict[str, List[str]] = defaultdict(list)
    stats = defaultdict(int)

    for path in paths:
        if not is_probably_text(path):
            stats["skipped_binary"] += 1
            continue
        if verbose:
            print(f"Reading {path}", file=sys.stderr)
        for record in iter_records(path, stats):
            stats["records"] += 1
            if not isinstance(record, dict):
                continue
            author = record.get("author")
            if not author or author == "[deleted]":
                stats["skipped_authors"] += 1
                continue
            if "parent_id" in record:
                stats["comments"] += 1
                comment_id = normalize_comment_id(record)
                if comment_id:
                    id_to_author[comment_id] = author
                    if comment_id in pending:
                        for child_author in pending.pop(comment_id):
                            add_edge(edges, child_author, author)
                parent_id = record.get("parent_id")
                if parent_id:
                    parent_author = id_to_author.get(parent_id)
                    if parent_author:
                        add_edge(edges, author, parent_author)
                    else:
                        pending[parent_id].append(author)
                else:
                    stats["missing_parent"] += 1
            else:
                submission_id = normalize_submission_id(record)
                if submission_id:
                    stats["submissions"] += 1
                    id_to_author[submission_id] = author
                    if submission_id in pending:
                        for child_author in pending.pop(submission_id):
                            add_edge(edges, child_author, author)

    stats["unresolved_parents"] = sum(len(v) for v in pending.values())
    stats["edges"] = len(edges)
    stats["users"] = len({u for edge in edges for u in edge})
    return edges, stats


def build_adjacency(edges: Dict[Tuple[str, str], int]) -> Dict[str, Dict[str, int]]:
    adjacency: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (src, dst), weight in edges.items():
        adjacency[src][dst] += weight
        adjacency[dst][src] += weight
    return adjacency


def label_propagation(adjacency: Dict[str, Dict[str, int]], max_iter: int, seed: int) -> Dict[str, str]:
    nodes = list(adjacency.keys())
    labels = {node: node for node in nodes}
    rng = random.Random(seed)

    for _ in range(max_iter):
        rng.shuffle(nodes)
        changes = 0
        for node in nodes:
            neighbors = adjacency.get(node)
            if not neighbors:
                continue
            scores: Dict[str, int] = defaultdict(int)
            for neighbor, weight in neighbors.items():
                scores[labels[neighbor]] += weight
            best_weight = max(scores.values())
            best_labels = [label for label, weight in scores.items() if weight == best_weight]
            best_label = min(best_labels)
            if labels[node] != best_label:
                labels[node] = best_label
                changes += 1
        if changes == 0:
            break
    return labels


def communities_from_labels(labels: Dict[str, str]) -> Dict[str, List[str]]:
    communities: Dict[str, List[str]] = defaultdict(list)
    for node, label in labels.items():
        communities[label].append(node)
    for members in communities.values():
        members.sort()
    return communities


def write_communities(path: str, communities: Dict[str, List[str]], min_size: int) -> List[Tuple[str, List[str]]]:
    filtered = [(label, members) for label, members in communities.items() if len(members) >= min_size]
    filtered.sort(key=lambda item: (-len(item[1]), item[0]))
    with open(path, "w", encoding="utf-8") as handle:
        for idx, (label, members) in enumerate(filtered, 1):
            handle.write(f"Community {idx} (size={len(members)} label={label})\n")
            for member in members:
                handle.write(f"  {member}\n")
            handle.write("\n")
    return filtered


def write_dot(path: str, adjacency: Dict[str, Dict[str, int]], communities: List[Tuple[str, List[str]]],
              max_nodes: int) -> None:
    palette = [
        "#f2c14e",
        "#ff6f59",
        "#4d9de0",
        "#7bd389",
        "#c1a5a9",
        "#1b998b",
        "#e84855",
        "#f9c80e",
    ]

    node_to_community = {}
    for idx, (_, members) in enumerate(communities):
        for member in members:
            node_to_community[member] = idx

    nodes = list(node_to_community.keys())
    if max_nodes and len(nodes) > max_nodes:
        degrees = {node: sum(adjacency.get(node, {}).values()) for node in nodes}
        nodes = [node for node, _ in sorted(degrees.items(), key=lambda item: (-item[1], item[0]))[:max_nodes]]
        node_to_community = {node: node_to_community[node] for node in nodes}

    node_set = set(nodes)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("graph communities {\n")
        handle.write("  graph [overlap=false, splines=true];\n")
        handle.write("  node [shape=circle, style=filled, fontname=Helvetica];\n")

        for idx, (_, members) in enumerate(communities):
            cluster_nodes = [m for m in members if m in node_set]
            if not cluster_nodes:
                continue
            color = palette[idx % len(palette)]
            handle.write(f"  subgraph cluster_{idx} {{\n")
            handle.write(f"    color=\"{color}\";\n")
            for member in cluster_nodes:
                safe = member.replace("\"", "\\\"")
                handle.write(f"    \"{safe}\";\n")
            handle.write("  }\n")

        seen = set()
        for src in node_set:
            for dst, weight in adjacency.get(src, {}).items():
                if dst not in node_set:
                    continue
                if src == dst:
                    continue
                key = tuple(sorted((src, dst)))
                if key in seen:
                    continue
                seen.add(key)
                handle.write(f"  \"{src}\" -- \"{dst}\" [penwidth={1 + weight / 2}];\n")

        handle.write("}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect user communities from Reddit exports.")
    parser.add_argument("--inputs", nargs="+", default=["data"], help="Files or directories to scan.")
    parser.add_argument("--communities-out", default="communities.txt", help="Output text file.")
    parser.add_argument("--dot-out", default="communities.dot", help="Graphviz DOT output.")
    parser.add_argument("--min-community-size", type=int, default=2, help="Minimum community size to include.")
    parser.add_argument("--max-iter", type=int, default=20, help="Label propagation iterations.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for label propagation.")
    parser.add_argument("--dot-max-nodes", type=int, default=0, help="Limit nodes in DOT (0 = no limit).")
    parser.add_argument("--verbose", action="store_true", help="Verbose progress output.")
    args = parser.parse_args()

    paths = gather_files(args.inputs)
    if not paths:
        print("No input files found.", file=sys.stderr)
        return 1

    edges, stats = build_graph(paths, args.verbose)
    if not edges:
        print("No interactions detected. Check input files.", file=sys.stderr)
        return 1

    adjacency = build_adjacency(edges)
    labels = label_propagation(adjacency, args.max_iter, args.seed)
    communities = communities_from_labels(labels)

    filtered = write_communities(args.communities_out, communities, args.min_community_size)
    write_dot(args.dot_out, adjacency, filtered, args.dot_max_nodes)

    print("Processed records:", stats.get("records", 0))
    print("Comments:", stats.get("comments", 0))
    print("Submissions:", stats.get("submissions", 0))
    print("Edges:", stats.get("edges", 0))
    print("Users:", stats.get("users", 0))
    print("Unresolved parent refs:", stats.get("unresolved_parents", 0))
    print("Communities written:", len(filtered))
    print(f"Communities file: {args.communities_out}")
    print(f"DOT file: {args.dot_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
