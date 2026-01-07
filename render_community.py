#!/usr/bin/env python3
"""Render a single community with edge weights and recency colors."""

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


def mix_color(color_a: str, color_b: str, t: float) -> str:
    def to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def to_hex(rgb: Tuple[int, int, int]) -> str:
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    a = to_rgb(color_a)
    b = to_rgb(color_b)
    rgb = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return to_hex(rgb)


def build_edge_stats(paths: List[str], member_set: set[str], verbose: bool) -> Dict[Tuple[str, str], Dict[str, int]]:
    id_to_author: Dict[str, str] = {}
    pending: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    edge_stats: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"weight": 0, "latest": 0})

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

            timestamp = parse_timestamp(record.get('created_utc') or record.get('created')) or 0

            if 'parent_id' in record:
                if author in member_set:
                    comment_id = normalize_comment_id(record)
                    if comment_id:
                        id_to_author[comment_id] = author
                        if comment_id in pending:
                            for child_author, child_time in pending.pop(comment_id):
                                if child_author in member_set:
                                    key = tuple(sorted((child_author, author)))
                                    stats = edge_stats[key]
                                    stats['weight'] += 1
                                    stats['latest'] = max(stats['latest'], child_time)
                    parent_id = record.get('parent_id')
                    if parent_id:
                        parent_author = id_to_author.get(parent_id)
                        if parent_author and parent_author in member_set:
                            key = tuple(sorted((author, parent_author)))
                            stats = edge_stats[key]
                            stats['weight'] += 1
                            stats['latest'] = max(stats['latest'], timestamp)
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
                                    stats['weight'] += 1
                                    stats['latest'] = max(stats['latest'], child_time)

    return edge_stats


def write_dot(path: str, members: List[str], edge_stats: Dict[Tuple[str, str], Dict[str, int]]) -> None:
    if not edge_stats:
        raise SystemExit('No edges found for community.')

    weights = [stats['weight'] for stats in edge_stats.values()]
    min_w, max_w = min(weights), max(weights)

    times = [stats['latest'] for stats in edge_stats.values() if stats['latest']]
    min_t, max_t = (min(times), max(times)) if times else (0, 0)

    def scale_weight(weight: int) -> float:
        if max_w == min_w:
            return 1.5
        return 1.0 + 4.0 * (weight - min_w) / (max_w - min_w)

    def scale_color(timestamp: int) -> str:
        if max_t == min_t:
            return '#4d9de0'
        t = (timestamp - min_t) / (max_t - min_t)
        return mix_color('#bfe1f4', '#1d3557', t)

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('graph community {\n')
        handle.write('  graph [overlap=false, splines=true];\n')
        handle.write('  node [shape=circle, style=filled, fontname=Helvetica, fillcolor="#f2c14e"];\n')
        for member in members:
            safe = member.replace('"', '\\"')
            handle.write(f'  "{safe}";\n')
        for (a, b), stats in edge_stats.items():
            weight = stats['weight']
            color = scale_color(stats['latest'])
            penwidth = scale_weight(weight)
            handle.write(
                f'  "{a}" -- "{b}" [label="{weight}", color="{color}", penwidth={penwidth:.2f}];\n'
            )
        handle.write('}\n')


def main() -> int:
    parser = argparse.ArgumentParser(description='Render a community with weighted, colored edges.')
    parser.add_argument('--inputs', nargs='+', default=['data'], help='Files or directories to scan.')
    parser.add_argument('--communities-file', default='communities_got.txt', help='Communities text file.')
    parser.add_argument('--community-index', type=int, required=True, help='Community index to render.')
    parser.add_argument('--dot-out', default='community_weighted.dot', help='Output DOT file.')
    parser.add_argument('--verbose', action='store_true', help='Verbose progress output.')
    args = parser.parse_args()

    members = load_community_members(args.communities_file, args.community_index)
    if not members:
        print('Community not found or empty.', file=sys.stderr)
        return 1

    member_set = set(members)
    paths = gather_files(args.inputs)
    edge_stats = build_edge_stats(paths, member_set, args.verbose)
    if not edge_stats:
        print('No edges found for this community.', file=sys.stderr)
        return 1

    write_dot(args.dot_out, members, edge_stats)
    print(f'Wrote {args.dot_out} with {len(members)} nodes and {len(edge_stats)} edges.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
