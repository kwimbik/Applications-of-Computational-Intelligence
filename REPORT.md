# NBA Community Analysis Notes

## Data Origin
- Source: Reddit comment and submission exports stored under `data/`.
- Boards used: NBA-focused subreddits (e.g., `nba`, `nbadiscussion`, `lakers`, `bostonceltics`, `NBA_Draft`).
- Format: JSONL/CSV with fields such as `author`, `id`/`name`, `parent_id`, and `created_utc`.

## Board Wars and Conflict
**Definition**
- "Board wars" are modeled as cross-board contention where users express contrasting sentiment or stance across two boards.

**How conflict was found**
- We compare user sentiment across two boards and highlight users whose sentiment polarity differs between them.
- The visual is a bipartite-style plot that connects users’ sentiment distributions between boards to show contrast clusters.

**Data used**
- Comment text for sentiment scoring.
- Board-specific comment exports for both sides of the comparison.
- Output shown in `sentiment_bipartite.png`.

## Single-Board Community Detection
**How the community was found**
- Build a user–user interaction graph from replies within a single board.
- An interaction is created when user A replies to user B’s comment or submission.
- Communities are detected with label propagation on the resulting graph.

**Data used**
- One board’s comments/submissions (e.g., `data/freefolk_comments` or `data/gameofthrones_comments`).

**What the graph represents**
- Nodes: users.
- Edges: reply interactions between users.
- Edge weight: number of replies (interaction frequency).

**Directed or undirected?**
- **Undirected.** The raw interaction is directed (replier → parent author), but for community detection it is symmetrized to capture mutual interaction strength and to simplify clustering.

## Inter-Board Community (Cross-Board Graph)
**How the community was found**
- Identify **cross-board users**: authors active in 2+ NBA boards.
- Build a graph of replies among these users across all NBA boards.
- Select a 10–50 user subset and rank it with PageRank.

**Data used**
- Cross-board NBA comment/submission exports (e.g., `nba_comments`, `nbadiscussion_comments`, `lakers_comments`, `bostonceltics_comments`, `NBA_Draft_comments`).

**What the graph represents**
- Nodes: cross-board users.
- Edges: replies between those users (weighted by reply count).

**Directed or undirected?**
- **Undirected.** Reply interactions are aggregated into an undirected weighted edge to represent the strength of interaction regardless of direction.

## PageRank Logic and Purpose
**Why PageRank**
- Highlights influential users within the selected cross-board community.
- Accounts for both **how many** interactions a user receives and **who** those interactions are connected to.

**How it works (high level)**
- Each user starts with equal score.
- Scores are redistributed through the network over weighted edges.
- Users connected to other high-scoring users gain more rank.

**Example edges (from `nba_community_pagerank.dot`)**
- `Barncore` — `xychosis` (weight 20)
- `Barncore` — `deadskin` (weight 22)
- `Barncore` — `rps215` (weight 38)
- `endubs` — `rps215` (weight 8)

**Output files**
- `nba_community_pagerank.txt`: ranked list of users by PageRank.
- `nba_community_pagerank.dot`: graph with edge weights and PageRank-scaled node sizes.
- `nba_community_pagerank.png`: rendered visualization.
