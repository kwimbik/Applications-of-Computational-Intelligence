#!/usr/bin/env python3
"""Compute centrality metrics for a selected community."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import sys
from collections import defaultdict, deque
from typing import Dict, Iterable, Iterator, List, Tuple


def open_text(path: str) -> io.TextIOBase:
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    return open(path, 'rt', encoding='utf-8', errors='replace')


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
            return 'jsonl'
        lead = first[0]
        if lead == '[':
            return 'json'
        if lead == '{':
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('{'):
                    return 'jsonl'
                return 'json'
            return 'jsonl'
        return 'csv'


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
        if 'data' in data and isinstance(data['data'], list):
            for obj in data['data']:
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
    if fmt == 'jsonl':
        yield from iter_jsonl(path)
    elif fmt == 'json':
        yield from iter_json(path)
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
        with open(path, 'rb') as handle:
            chunk = handle.read(1024)
    except OSError:
        return False
    return b'\x00' not in chunk


def normalize_comment_id(record: dict) -> str | None:
    name = record.get('name')
    if name and isinstance(name, str):
        return name
    cid = record.get('id')
    if cid and isinstance(cid, str):
        return f't1_{cid}'
    return None


def normalize_submission_id(record: dict) -> str | None:
    name = record.get('name')
    if name and isinstance(name, str):
        return name
    sid = record.get('id')
    if sid and isinstance(sid, str):
        return f't3_{sid}'
    return None


def load_community_members(path: str, idx: int) -> List[str]:
    members: List[str] = []
    header = f'Community {idx} '
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            if line.startswith(header):
                for line in handle:
                    if not line.strip():
                        return members
                    if line.startswith('  '):
                        members.append(line.strip())
                return members
    return members


def build_edge_weights(paths: List[str], member_set: set[str], verbose: bool) -> Dict[Tuple[str, str], int]:
    id_to_author: Dict[str, str] = {}
    pending: Dict[str, List[str]] = defaultdict(list)
    edges: Dict[Tuple[str, str], int] = defaultdict(int)

    for path in paths:
        if not is_probably_text(path):
            continue
        if verbose:
            print(f"Reading {path}", file=sys.stderr)
        for record in iter_records(path):
            if not isinstance(record, dict):
                continue
            author = record.get('author')
            if not author or author == '[deleted]':
                continue

            if 'parent_id' in record:
                if author in member_set:
                    comment_id = normalize_comment_id(record)
                    if comment_id:
                        id_to_author[comment_id] = author
                        if comment_id in pending:
                            for child_author in pending.pop(comment_id):
                                if child_author in member_set:
                                    key = tuple(sorted((child_author, author)))
                                    edges[key] += 1
                    parent_id = record.get('parent_id')
                    if parent_id:
                        parent_author = id_to_author.get(parent_id)
                        if parent_author and parent_author in member_set:
                            key = tuple(sorted((author, parent_author)))
                            edges[key] += 1
                        else:
                            pending[parent_id].append(author)
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
                            for child_author in pending.pop(submission_id):
                                if child_author in member_set:
                                    key = tuple(sorted((child_author, author)))
                                    edges[key] += 1

    return edges


def build_adjacency(edges: Dict[Tuple[str, str], int]) -> Dict[str, Dict[str, int]]:
    adjacency: Dict[str, Dict[str, int]] = defaultdict(dict)
    for (a, b), w in edges.items():
        adjacency[a][b] = w
        adjacency[b][a] = w
    return adjacency


def dijkstra(adjacency: Dict[str, Dict[str, int]], source: str) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, int], List[str]]:
    dist: Dict[str, float] = {source: 0.0}
    preds: Dict[str, List[str]] = defaultdict(list)
    sigma: Dict[str, int] = defaultdict(int)
    sigma[source] = 1
    visited = set()
    order: List[str] = []
    # naive priority queue
    while True:
        min_node = None
        min_dist = float('inf')
        for node, d in dist.items():
            if node in visited:
                continue
            if d < min_dist:
                min_dist = d
                min_node = node
        if min_node is None:
            break
        visited.add(min_node)
        order.append(min_node)
        for neighbor, weight in adjacency.get(min_node, {}).items():
            length = min_dist + (1.0 / weight if weight else 1.0)
            if neighbor not in dist or length < dist[neighbor]:
                dist[neighbor] = length
                preds[neighbor] = [min_node]
                sigma[neighbor] = sigma[min_node]
            elif abs(length - dist[neighbor]) < 1e-12:
                preds[neighbor].append(min_node)
                sigma[neighbor] += sigma[min_node]
    return dist, preds, sigma, order


def betweenness_centrality(adjacency: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    nodes = list(adjacency.keys())
    cb = {node: 0.0 for node in nodes}
    for s in nodes:
        dist, preds, sigma, order = dijkstra(adjacency, s)
        delta = {node: 0.0 for node in nodes}
        for w in reversed(order):
            for v in preds.get(w, []):
                if sigma[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    # normalize for undirected graphs
    for node in cb:
        cb[node] /= 2.0
    return cb


def closeness_centrality(adjacency: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    nodes = list(adjacency.keys())
    closeness = {}
    for node in nodes:
        dist, _, _, _ = dijkstra(adjacency, node)
        if len(dist) <= 1:
            closeness[node] = 0.0
            continue
        total = sum(dist.values())
        if total > 0:
            closeness[node] = (len(dist) - 1) / total
        else:
            closeness[node] = 0.0
    return closeness


def eigenvector_centrality(adjacency: Dict[str, Dict[str, int]], max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]:
    nodes = list(adjacency.keys())
    if not nodes:
        return {}
    scores = {node: 1.0 for node in nodes}
    for _ in range(max_iter):
        next_scores = {node: 0.0 for node in nodes}
        for node in nodes:
            for neighbor, weight in adjacency.get(node, {}).items():
                next_scores[node] += scores[neighbor] * weight
        norm = sum(value * value for value in next_scores.values()) ** 0.5
        if norm == 0:
            break
        for node in nodes:
            next_scores[node] /= norm
        diff = max(abs(next_scores[node] - scores[node]) for node in nodes)
        scores = next_scores
        if diff < tol:
            break
    return scores


def connected_components(adjacency: Dict[str, Dict[str, int]]) -> List[List[str]]:
    seen = set()
    components = []
    for node in adjacency:
        if node in seen:
            continue
        comp = []
        queue = deque([node])
        seen.add(node)
        while queue:
            current = queue.popleft()
            comp.append(current)
            for neighbor in adjacency.get(current, {}):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(comp)
    return components


def write_report(path: str, community_idx: int, members: List[str], edges: Dict[Tuple[str, str], int],
                 adjacency: Dict[str, Dict[str, int]], degree: Dict[str, int], strength: Dict[str, int],
                 betweenness: Dict[str, float], closeness: Dict[str, float], eigenvector: Dict[str, float]) -> None:
    n = len(members)
    m = len(edges)
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0.0
    avg_weight = sum(edges.values()) / m if m else 0.0
    avg_degree = sum(degree.values()) / n if n else 0.0
    components = connected_components(adjacency)
    components.sort(key=len, reverse=True)

    def top_items(metric: Dict[str, float], limit: int = 10) -> List[Tuple[str, float]]:
        return sorted(metric.items(), key=lambda item: (-item[1], item[0]))[:limit]

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(f'Community {community_idx} centrality report\n')
        handle.write(f'Nodes: {n}\n')
        handle.write(f'Edges: {m}\n')
        handle.write(f'Density: {density:.4f}\n')
        handle.write(f'Average degree: {avg_degree:.2f}\n')
        handle.write(f'Average edge weight: {avg_weight:.2f}\n')
        handle.write(f'Connected components: {len(components)}\n')
        if components:
            handle.write(f'Largest component size: {len(components[0])}\n')
        handle.write('\nTop 10 by weighted degree (strength):\n')
        for node, value in top_items(strength):
            handle.write(f'  {node}\t{value}\n')
        handle.write('\nTop 10 by degree (unweighted):\n')
        for node, value in top_items(degree):
            handle.write(f'  {node}\t{value}\n')
        handle.write('\nTop 10 by betweenness (weighted):\n')
        for node, value in top_items(betweenness):
            handle.write(f'  {node}\t{value:.4f}\n')
        handle.write('\nTop 10 by closeness (weighted):\n')
        for node, value in top_items(closeness):
            handle.write(f'  {node}\t{value:.6f}\n')
        handle.write('\nTop 10 by eigenvector centrality (weighted):\n')
        for node, value in top_items(eigenvector):
            handle.write(f'  {node}\t{value:.6f}\n')


def main() -> int:
    parser = argparse.ArgumentParser(description='Centrality analysis for a community.')
    parser.add_argument('--inputs', nargs='+', default=['data'], help='Files or directories to scan.')
    parser.add_argument('--communities-file', default='communities_got.txt', help='Communities text file.')
    parser.add_argument('--community-index', type=int, required=True, help='Community index to analyze.')
    parser.add_argument('--report-out', default='community_centrality_report.txt', help='Output report file.')
    parser.add_argument('--verbose', action='store_true', help='Verbose progress output.')
    args = parser.parse_args()

    members = load_community_members(args.communities_file, args.community_index)
    if not members:
        print('Community not found or empty.', file=sys.stderr)
        return 1

    member_set = set(members)
    paths = gather_files(args.inputs)
    edges = build_edge_weights(paths, member_set, args.verbose)
    if not edges:
        print('No edges found for this community.', file=sys.stderr)
        return 1

    adjacency = build_adjacency(edges)
    degree = {node: len(neighbors) for node, neighbors in adjacency.items()}
    strength = {node: sum(neighbors.values()) for node, neighbors in adjacency.items()}

    betweenness = betweenness_centrality(adjacency)
    closeness = closeness_centrality(adjacency)
    eigenvector = eigenvector_centrality(adjacency)

    write_report(args.report_out, args.community_index, members, edges, adjacency,
                 degree, strength, betweenness, closeness, eigenvector)
    print(f'Wrote report to {args.report_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
