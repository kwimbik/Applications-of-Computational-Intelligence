# Applications of Computational Intelligence

## Community detection from Reddit exports

This repo includes Reddit comment/submission exports under `data/` (one JSON object per line). The script below builds a user-user interaction graph and detects communities via label propagation.

### Usage

```bash
python detect_communities.py --inputs data \
  --communities-out communities.txt \
  --dot-out communities.dot \
  --min-community-size 3 \
  --max-iter 25
```

You can point `--inputs` at specific files or multiple directories. Files may be JSONL, JSON, or CSV, and may be gzip-compressed (`.gz`).

### Assumptions and interaction signals

- Comment rows include `author`, `id`/`name`, and `parent_id`.
- Submission rows include `author` and `id`/`name`.
- A user is considered to interact with another when they reply to a parent comment or submission (`parent_id`), producing a directed edge from replier to replied-to author.
- Entries without authors or with `author == "[deleted]"` are skipped.
- The DOT output groups nodes by community; use Graphviz tools such as `dot -Tpng communities.dot -o communities.png` to render.
