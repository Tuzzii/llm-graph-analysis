"""
make_bm_prompts_facebook.py
Author: Tran Le Vinh Buu
Purpose: Generate 50 Bipartite Matching OOD prompts from Facebook dataset.
Output:
  - buu_bipartite_facebook.json
  - buu_bm_fb_checks.csv
Notes:
  - Construct bipartite graph by sampling subgraph, partitioning nodes,
    then keeping/adding cross-edges. Ground truth: maximum matching (pairs).
"""

import json, random, networkx as nx, numpy as np, pandas as pd
from datetime import datetime
from tqdm import tqdm
import os

NUM_PROMPTS = 50
SEED = 20251102
random.seed(SEED)
np.random.seed(SEED)

OUT_JSON = "buu_bipartite_facebook.json"
OUT_CSV  = "buu_bm_fb_checks.csv"

G_fb = nx.read_edgelist("data/facebook_combined.txt")
G_fb = nx.relabel_nodes(G_fb, lambda x: str(x))
print(f"Facebook graph: {G_fb.number_of_nodes()} nodes, {G_fb.number_of_edges()} edges")

def get_subgraph(G, center, size=30):
    nodes = list(nx.single_source_shortest_path_length(G, center, cutoff=2).keys())
    if len(nodes) > size:
        nodes = random.sample(nodes, size)
    SG = G.subgraph(nodes).copy()
    if SG.number_of_nodes() < 4:
        SG = G.subgraph(random.sample(list(G.nodes()), size)).copy()
    return SG

def make_question_text(nodes, edges, A, B):
    e_text = ", ".join([f"{u}-{v}" for u,v in edges])
    return (f"Given the bipartite graph with left partition {A} and right partition {B}, "
            f"and edges {e_text}, what is the size of a maximum matching? Provide one maximum matching as pairs.")

prompts, checks = [], []
nodes_all = list(G_fb.nodes())
os.makedirs("results", exist_ok=True)

attempts = 0
max_attempts = NUM_PROMPTS * 12

while len(prompts) < NUM_PROMPTS and attempts < max_attempts:
    attempts += 1
    seed = SEED + attempts * 13
    random.seed(seed)

    center = random.choice(nodes_all)
    SG = get_subgraph(G_fb, center, size=random.randint(20,40))
    if SG.number_of_nodes() < 4:
        continue

    nodes = list(SG.nodes())
    random.shuffle(nodes)
    # split into two partitions (roughly half)
    mid = max(1, len(nodes)//2)
    A = nodes[:mid]
    B = nodes[mid:]

    # build bipartite graph with crossing edges only
    BG = nx.Graph()
    BG.add_nodes_from(A, bipartite=0)
    BG.add_nodes_from(B, bipartite=1)

    # add existing crossing edges from SG
    for u,v in SG.edges():
        if (u in A and v in B) or (u in B and v in A):
            # ensure edge orientation u in A -> v in B for storage
            if u in A and v in B:
                BG.add_edge(u, v)
            else:
                BG.add_edge(v, u)

    # if too few edges, add random cross edges to make matching meaningful
    if BG.number_of_edges() < max(1, int(0.1 * SG.number_of_edges())):
        # add up to 15 random cross edges
        tries = 0
        while BG.number_of_edges() < min( max(2, int(0.1 * SG.number_of_edges())), 15) and tries < 200:
            tries += 1
            u = random.choice(A)
            v = random.choice(B)
            if not BG.has_edge(u, v):
                BG.add_edge(u, v)

    if BG.number_of_edges() == 0:
        continue

    # compute maximum matching using networkx (returns set/dict)
    try:
        matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(BG, top_nodes=A)
    except Exception:
        # fallback to general maximum_matching
        matching = nx.algorithms.matching.max_weight_matching(BG, maxcardinality=True)

    # hopcroft_karp_matching returns dict mapping both ways, we want pairs (u in A -> v)
    pairs = []
    if isinstance(matching, dict):
        # keep only A -> B entries
        for u in A:
            v = matching.get(u)
            if v is not None:
                pairs.append((u, v))
    else:
        # set of pairs in tuple form
        for a,b in matching:
            # ensure a in A
            if a in A and b in B:
                pairs.append((a,b))
            elif b in A and a in B:
                pairs.append((b,a))

    match_size = len(pairs)
    gt = {"match_size": match_size, "pairs": pairs, "status": "ok"}

    rec = {
        "id": f"FB_BM_{len(prompts):03d}",
        "task": "bipartite_matching",
        "dataset": "facebook_combined",
        "seed": seed,
        "graph": {
            "directed": False,
            "left": A,
            "right": B,
            "edges": [{"u": u, "v": v} for u,v in BG.edges()],
        },
        "question": make_question_text(list(BG.nodes()), list(BG.edges()), A, B),
        "ground_truth": gt,
        "notes": {
            "node_count": BG.number_of_nodes(),
            "edge_count": BG.number_of_edges(),
            "timestamp": datetime.now().isoformat()
        }
    }

    prompts.append(rec)
    checks.append({
        "id": rec["id"],
        "n_left": len(A),
        "n_right": len(B),
        "n_edges": BG.number_of_edges(),
        "match_size": match_size,
        "status": gt["status"]
    })

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(prompts, f, indent=2, ensure_ascii=False)

pd.DataFrame(checks).to_csv(OUT_CSV, index=False)
print(f"✅ Done: {len(prompts)} prompts saved to {OUT_JSON}")
print(f"ℹ️ Attempts used: {attempts}/{max_attempts}")
