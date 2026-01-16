#!/usr/bin/env python3
"""Find an NBA cross-board community, run PageRank, and render a DOT graph."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple


def open_text(path: str) -> io.TextIOBase:
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def detect_format(path: str) -> str:
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


def iter_jsonl(path: str) -> Iterator[dict]:
    with open_text(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def iter_json(path: str) -> Iterator[dict]:
    with open_text(path) as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
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


def iter_records(path: str) -> Iterator[dict]:
    fmt = detect_format(path)
    if fmt == "jsonl":
        yield from iter_jsonl(path)
    elif fmt == "json":
        yield from iter_json(path)
    else:
        yield from iter_csv(path)


def gather_files(inputs: List[str]) -> List[str]:
    files: List[str] = []
    for entry in inputs:
        if os.path.isdir(entry):
            for root, _, filenames in os.walk(entry):
                for name in filenames:
                    lower = name.lower()
                    has_ext = "." in name
                    if has_ext and not (
                        lower.endswith(".json")
                        or lower.endswith(".jsonl")
                        or lower.endswith(".csv")
                        or lower.endswith(".gz")
                    ):
                        continue
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


def parse_timestamp(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def infer_board(path: str, data_root: str) -> str:
    try:
        rel = os.path.relpath(path, data_root)
    except ValueError:
        rel = path
    parts = rel.split(os.sep)
    if parts:
        return parts[0]
    return os.path.basename(os.path.dirname(path))


def collect_user_stats(paths: List[str], data_root: str, max_records: int, max_records_per_file: int,
                       max_files: int, verbose: bool) -> Tuple[Dict[str, set[str]], Dict[str, int]]:
    user_boards: Dict[str, set[str]] = defaultdict(set)
    user_activity: Dict[str, int] = defaultdict(int)
    total_records = 0
    files_read = 0
    for path in paths:
        if not is_probably_text(path):
            continue
        board = infer_board(path, data_root)
        if verbose:
            print(f"Scanning authors in {path}", file=sys.stderr)
        per_file = 0
        for record in iter_records(path):
            if not isinstance(record, dict):
                continue
            author = record.get("author")
            if not author or author == "[deleted]":
                continue
            user_boards[author].add(board)
            user_activity[author] += 1
            total_records += 1
            per_file += 1
            if max_records and total_records >= max_records:
                return user_boards, user_activity
            if max_records_per_file and per_file >= max_records_per_file:
                break
        files_read += 1
        if max_files and files_read >= max_files:
            break
    return user_boards, user_activity


def select_candidates(user_boards: Dict[str, set[str]], user_activity: Dict[str, int],
                      min_boards: int, max_candidates: int) -> List[str]:
    eligible = [user for user, boards in user_boards.items() if len(boards) >= min_boards]
    eligible.sort(key=lambda user: (-user_activity.get(user, 0), user))
    if max_candidates and len(eligible) > max_candidates:
        return eligible[:max_candidates]
    return eligible


def build_edge_stats(paths: List[str], member_set: set[str], max_records: int, max_records_per_file: int,
                     max_files: int, verbose: bool) -> Dict[Tuple[str, str], Dict[str, int]]:
    id_to_author: Dict[str, str] = {}
    pending: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    edge_stats: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"weight": 0, "latest": 0})
    total_records = 0
    files_read = 0

    for path in paths:
        if not is_probably_text(path):
            continue
        if verbose:
            print(f"Reading {path}", file=sys.stderr)
        per_file = 0
        for record in iter_records(path):
            if not isinstance(record, dict):
                continue
            author = record.get("author")
            if not author or author == "[deleted]":
                continue

            timestamp = parse_timestamp(record.get("created_utc") or record.get("created")) or 0

            if "parent_id" in record:
                if author in member_set:
                    comment_id = normalize_comment_id(record)
                    if comment_id:
                        id_to_author[comment_id] = author
                        if comment_id in pending:
                            for child_author, child_time in pending.pop(comment_id):
                                if child_author in member_set:
                                    key = tuple(sorted((child_author, author)))
                                    stats = edge_stats[key]
                                    stats["weight"] += 1
                                    stats["latest"] = max(stats["latest"], child_time)
                    parent_id = record.get("parent_id")
                    if parent_id:
                        parent_author = id_to_author.get(parent_id)
                        if parent_author and parent_author in member_set:
                            key = tuple(sorted((author, parent_author)))
                            stats = edge_stats[key]
                            stats["weight"] += 1
                            stats["latest"] = max(stats["latest"], timestamp)
                        else:
                            pending[parent_id].append((author, timestamp))
                else:
                    comment_id = normalize_comment_id(record)
                    if comment_id and author in member_set:
                        id_to_author[comment_id] = author
            else:
                if author in member_set:
                    submission_id = normalize_submission_id(record)
                    if submission_id:
                        id_to_author[submission_id] = author
                        if submission_id in pending:
                            for child_author, child_time in pending.pop(submission_id):
                                if child_author in member_set:
                                    key = tuple(sorted((child_author, author)))
                                    stats = edge_stats[key]
                                    stats["weight"] += 1
                                    stats["latest"] = max(stats["latest"], child_time)
            total_records += 1
            per_file += 1
            if max_records and total_records >= max_records:
                return edge_stats
            if max_records_per_file and per_file >= max_records_per_file:
                break
        files_read += 1
        if max_files and files_read >= max_files:
            break

    return edge_stats


def build_weighted_adjacency(edge_stats: Dict[Tuple[str, str], Dict[str, int]],
                             nodes: Iterable[str]) -> Dict[str, Dict[str, int]]:
    adjacency: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for node in nodes:
        adjacency[node]  # ensure node exists
    for (a, b), stats in edge_stats.items():
        weight = stats["weight"]
        adjacency[a][b] += weight
        adjacency[b][a] += weight
    return adjacency


def pagerank(adjacency: Dict[str, Dict[str, int]], damping: float = 0.85,
             max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]:
    nodes = list(adjacency.keys())
    if not nodes:
        return {}
    n = len(nodes)
    scores = {node: 1.0 / n for node in nodes}
    out_weight = {node: sum(neighbors.values()) for node, neighbors in adjacency.items()}

    for _ in range(max_iter):
        next_scores = {node: (1.0 - damping) / n for node in nodes}
        for node in nodes:
            if out_weight[node] == 0:
                share = damping * scores[node] / n
                for target in nodes:
                    next_scores[target] += share
                continue
            for neighbor, weight in adjacency[node].items():
                next_scores[neighbor] += damping * scores[node] * (weight / out_weight[node])
        diff = max(abs(next_scores[node] - scores[node]) for node in nodes)
        scores = next_scores
        if diff < tol:
            break
    return scores


def mix_color(color_a: str, color_b: str, t: float) -> str:
    def to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def to_hex(rgb: Tuple[int, int, int]) -> str:
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    a = to_rgb(color_a)
    b = to_rgb(color_b)
    rgb = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return to_hex(rgb)


def write_dot(path: str, members: List[str], edge_stats: Dict[Tuple[str, str], Dict[str, int]],
              pr_scores: Dict[str, float]) -> None:
    if not members:
        raise SystemExit("No community members to render.")

    weights = [stats["weight"] for stats in edge_stats.values()] or [1]
    min_w, max_w = min(weights), max(weights)

    times = [stats["latest"] for stats in edge_stats.values() if stats["latest"]]
    min_t, max_t = (min(times), max(times)) if times else (0, 0)

    pr_values = [pr_scores.get(member, 0.0) for member in members] or [0.0]
    min_pr, max_pr = min(pr_values), max(pr_values)

    def scale_weight(weight: int) -> float:
        if max_w == min_w:
            return 1.5
        return 1.0 + 4.0 * (weight - min_w) / (max_w - min_w)

    def scale_color(timestamp: int) -> str:
        if max_t == min_t:
            return "#4d9de0"
        t = (timestamp - min_t) / (max_t - min_t)
        return mix_color("#bfe1f4", "#1d3557", t)

    def scale_node(pr_value: float) -> float:
        if max_pr == min_pr:
            return 0.9
        return 0.6 + 1.0 * (pr_value - min_pr) / (max_pr - min_pr)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("graph community {\n")
        handle.write("  graph [overlap=false, splines=true];\n")
        handle.write("  node [shape=circle, style=filled, fontname=Helvetica, fillcolor=\"#f2c14e\"];\n")
        for member in members:
            safe = member.replace("\"", "\\\"")
            size = scale_node(pr_scores.get(member, 0.0))
            handle.write(f"  \"{safe}\" [fixedsize=true, width={size:.2f}, height={size:.2f}];\n")
        for (a, b), stats in edge_stats.items():
            weight = stats["weight"]
            color = scale_color(stats["latest"])
            penwidth = scale_weight(weight)
            handle.write(
                f"  \"{a}\" -- \"{b}\" [label=\"{weight}\", color=\"{color}\", penwidth={penwidth:.2f}];\n"
            )
        handle.write("}\n")


def write_pagerank(path: str, pr_scores: Dict[str, float]) -> None:
    ranked = sorted(pr_scores.items(), key=lambda item: (-item[1], item[0]))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("User\tPageRank\n")
        for user, score in ranked:
            handle.write(f"{user}\t{score:.6f}\n")


def default_nba_inputs(data_root: str) -> List[str]:
    if not os.path.isdir(data_root):
        return []
    team_names = {"lakers", "bostonceltics", "celtics"}
    inputs = []
    for name in os.listdir(data_root):
        path = os.path.join(data_root, name)
        if not (os.path.isdir(path) or os.path.isfile(path)):
            continue
        lowered = name.lower()
        if "nba" in lowered or lowered in team_names:
            if "copy" in lowered:
                continue
            inputs.append(path)
    return sorted(inputs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find an NBA cross-board community, run PageRank, and render a DOT graph."
    )
    parser.add_argument("--data-root", default="data", help="Root directory holding board exports.")
    parser.add_argument("--inputs", nargs="+", default=None, help="Specific files or directories to scan.")
    parser.add_argument("--min-community-size", type=int, default=10, help="Minimum community size.")
    parser.add_argument("--max-community-size", type=int, default=50, help="Maximum community size.")
    parser.add_argument("--min-board-count", type=int, default=2, help="Minimum boards per user (cross-board).")
    parser.add_argument("--max-candidate-users", type=int, default=200,
                        help="Cap cross-board users before PageRank (0 = no cap).")
    parser.add_argument("--max-records", type=int, default=200000,
                        help="Stop after this many records per pass (0 = no limit).")
    parser.add_argument("--max-records-per-file", type=int, default=50000,
                        help="Stop after this many records per file (0 = no limit).")
    parser.add_argument("--max-files", type=int, default=0, help="Stop after this many files (0 = no limit).")
    parser.add_argument("--dot-out", default="nba_community_pagerank.dot", help="Output DOT file.")
    parser.add_argument("--pagerank-out", default="nba_community_pagerank.txt", help="Output PageRank text file.")
    parser.add_argument("--members-out", default="nba_community_members.txt", help="Output community members file.")
    parser.add_argument("--verbose", action="store_true", help="Verbose progress output.")
    args = parser.parse_args()

    inputs = args.inputs if args.inputs else default_nba_inputs(args.data_root)
    if not inputs:
        print("No NBA inputs found. Provide --inputs or check --data-root.", file=sys.stderr)
        return 1

    paths = gather_files(inputs)
    if not paths:
        print("No input files found.", file=sys.stderr)
        return 1

    user_boards, user_activity = collect_user_stats(
        paths, args.data_root, args.max_records, args.max_records_per_file, args.max_files, args.verbose
    )
    candidates = select_candidates(
        user_boards, user_activity, args.min_board_count, args.max_candidate_users
    )
    if len(candidates) < args.min_community_size:
        print("Not enough cross-board users for the requested size.", file=sys.stderr)
        return 1

    member_set = set(candidates)
    edge_stats = build_edge_stats(
        paths, member_set, args.max_records, args.max_records_per_file, args.max_files, args.verbose
    )
    if not edge_stats:
        print("No edges found among cross-board candidates.", file=sys.stderr)
        return 1

    weighted_adj = build_weighted_adjacency(edge_stats, candidates)
    pr_scores = pagerank(weighted_adj)

    ranked = sorted(pr_scores.items(), key=lambda item: (-item[1], item[0]))
    target_size = min(len(ranked), args.max_community_size)
    if target_size < args.min_community_size:
        print("Not enough connected users for the requested size.", file=sys.stderr)
        return 1
    members = [user for user, _ in ranked[:target_size]]
    member_set = set(members)
    edge_stats = {key: stats for key, stats in edge_stats.items() if key[0] in member_set and key[1] in member_set}
    weighted_adj = build_weighted_adjacency(edge_stats, members)
    pr_scores = pagerank(weighted_adj)

    with open(args.members_out, "w", encoding="utf-8") as handle:
        handle.write(f"Size: {len(members)}\n")
        handle.write(f"Min boards per user: {args.min_board_count}\n")
        handle.write(f"Candidate users: {len(candidates)}\n")
        for member in members:
            handle.write(f"{member}\n")

    write_pagerank(args.pagerank_out, pr_scores)
    write_dot(args.dot_out, members, edge_stats, pr_scores)

    print(f"Selected community size: {len(members)}")
    print(f"Members file: {args.members_out}")
    print(f"PageRank file: {args.pagerank_out}")
    print(f"DOT file: {args.dot_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
