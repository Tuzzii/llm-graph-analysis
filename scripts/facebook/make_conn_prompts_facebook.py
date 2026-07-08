"""
make_conn_prompts_facebook.py
Author: Tran Le Vinh Buu
Purpose: Generate 50 Connectivity OOD prompts from Facebook dataset.
Output:
  - buu_connectivity_facebook.json
  - buu_conn_fb_checks.csv
Notes:
  - Each prompt: graph (undirected subgraph), source, target, question, ground_truth (connected: bool, path if exists)
  - Reproducible via seeds.
"""

import json, random, networkx as nx, numpy as np, pandas as pd
from datetime import datetime
from tqdm import tqdm
import os

NUM_PROMPTS = 50
SEED = 20251101
random.seed(SEED)
np.random.seed(SEED)

OUT_JSON = "buu_connectivity_facebook.json"
OUT_CSV  = "buu_conn_fb_checks.csv"

# load graph
G_fb = nx.read_edgelist("data/facebook_combined.txt")
G_fb = nx.relabel_nodes(G_fb, lambda x: str(x))
print(f"Facebook graph: {G_fb.number_of_nodes()} nodes, {G_fb.number_of_edges()} edges")

# helper: sample subgraph around a center
def get_subgraph(G, center, size=30):
    nodes = list(nx.single_source_shortest_path_length(G, center, cutoff=2).keys())
    if len(nodes) > size:
        nodes = random.sample(nodes, size)
    SG = G.subgraph(nodes).copy()
    if SG.number_of_nodes() < 2:
        SG = G.subgraph(random.sample(list(G.nodes()), size)).copy()
    return SG

def make_question_text(nodes, edges, src, tgt):
    e_text = ", ".join([f"{u}-{v}" for u,v in edges])
    return (f"Given the undirected subgraph of the Facebook network with nodes {nodes} and edges {e_text}, "
            f"are nodes {src} and {tgt} connected? If yes, provide one shortest path between them.")

prompts, checks = [], []
nodes_all = list(G_fb.nodes())
os.makedirs("results", exist_ok=True)

attempts = 0
max_attempts = NUM_PROMPTS * 10

while len(prompts) < NUM_PROMPTS and attempts < max_attempts:
    attempts += 1
    seed = SEED + attempts * 11
    random.seed(seed)

    center = random.choice(nodes_all)
    SG = get_subgraph(G_fb, center, size=random.randint(20, 40))
    if SG.number_of_nodes() < 2:
        continue

    src = random.choice(list(SG.nodes()))
    tgt = random.choice(list(SG.nodes()))
    if src == tgt:
        # choose different target
        candidates = [n for n in SG.nodes() if n != src]
        if not candidates:
            continue
        tgt = random.choice(candidates)

    # compute ground truth
    try:
        connected = nx.has_path(SG, src, tgt)
        path = None
        if connected:
            # get one shortest path
            path = nx.shortest_path(SG, src, tgt)
    except Exception as e:
        connected = False
        path = None

    gt = {"connected": bool(connected), "path": path, "status": "ok"}

    rec = {
        "id": f"FB_CONN_{len(prompts):03d}",
        "task": "connectivity",
        "dataset": "facebook_combined",
        "seed": seed,
        "graph": {
            "directed": False,
            "nodes": list(SG.nodes()),
            "edges": [{"u": u, "v": v} for u,v in SG.edges()],
            "source": src,
            "target": tgt
        },
        "question": make_question_text(list(SG.nodes()), list(SG.edges()), src, tgt),
        "ground_truth": gt,
        "notes": {
            "node_count": SG.number_of_nodes(),
            "edge_count": SG.number_of_edges(),
            "timestamp": datetime.now().isoformat()
        }
    }

    prompts.append(rec)
    checks.append({
        "id": rec["id"],
        "n_nodes": SG.number_of_nodes(),
        "n_edges": SG.number_of_edges(),
        "connected": gt["connected"],
        "status": gt["status"]
    })

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(prompts, f, indent=2, ensure_ascii=False)

pd.DataFrame(checks).to_csv(OUT_CSV, index=False)
print(f"✅ Done: {len(prompts)} prompts saved to {OUT_JSON}")
print(f"ℹ️ Attempts used: {attempts}/{max_attempts}")
