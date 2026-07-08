"""
make_sp_prompts_facebook.py
Author: Tran Le Vinh Buu
Purpose: Generate 50 Shortest Path OOD prompts from Facebook dataset.
Paper-based spec: NLGraph (OOD graph reasoning benchmark)
"""

import json, random, networkx as nx, numpy as np, pandas as pd
from datetime import datetime
from tqdm import tqdm
import os

# ==========================
# CONFIGURATION
# ==========================
NUM_PROMPTS = 50
SEED = 20251029
random.seed(SEED)
np.random.seed(SEED)

OUT_JSON = "buu_shortestpath_facebook.json"
OUT_CSV  = "buu_sp_fb_checks.csv"

# ==========================
# LOAD FACEBOOK GRAPH
# ==========================
G_fb = nx.read_edgelist("data/facebook_combined.txt")
print(f"Facebook graph: {G_fb.number_of_nodes()} nodes, {G_fb.number_of_edges()} edges")

# Ensure node labels are strings (for JSON compatibility)
G_fb = nx.relabel_nodes(G_fb, lambda x: str(x))

# ==========================
# HELPER FUNCTIONS
# ==========================
def get_subgraph(G, center, size=25):
    """Lấy 1 subgraph quanh node 'center', trong bán kính 2 bước."""
    nodes = list(nx.single_source_shortest_path_length(G, center, cutoff=2).keys())
    if len(nodes) > size:
        nodes = random.sample(nodes, size)
    SG = G.subgraph(nodes).copy()
    # Fallback: nếu subgraph quá nhỏ
    if SG.number_of_nodes() < 3:
        SG = G.subgraph(random.sample(list(G.nodes()), size)).copy()
    return SG


def make_question_text(nodes, edges, src, tgt):
    e_text = ", ".join([f"{u}-{v}" for u, v in edges])
    return (
        f"Given the subgraph of the Facebook social network with nodes {nodes} and edges {e_text}, "
        f"what is the length of the shortest path between node {src} and node {tgt}?"
    )


def compute_shortest_path(SG, s, t):
    """Tính shortest path (độ dài + đường đi), xử lý lỗi nếu không tìm thấy đường."""
    try:
        dist = nx.shortest_path_length(SG, s, t)
        path = nx.shortest_path(SG, s, t)
        return {"distance": dist, "path": path, "status": "ok"}
    except nx.NetworkXNoPath:
        return {"distance": None, "path": None, "status": "no_path"}
    except Exception as e:
        return {"distance": None, "path": None, "status": str(e)}

# ==========================
# GENERATE PROMPTS
# ==========================
prompts, checks = [], []
nodes_all = list(G_fb.nodes())
os.makedirs("results", exist_ok=True)

print("Generating Facebook SP prompts...")

attempts = 0
max_attempts = NUM_PROMPTS * 10  # thử lại nhiều lần nếu graph kém liên kết

while len(prompts) < NUM_PROMPTS and attempts < max_attempts:
    attempts += 1
    seed = SEED + attempts * 17
    random.seed(seed)

    # Lấy node trung tâm -> tạo subgraph
    center = random.choice(nodes_all)
    SG = get_subgraph(G_fb, center, size=random.randint(20, 35))

    if SG.number_of_nodes() < 3:
        continue

    src = random.choice(list(SG.nodes()))
    reachable_nodes = nx.single_source_shortest_path_length(SG, src)
    t_candidates = [n for n in reachable_nodes.keys() if n != src]

    if not t_candidates:
        continue

    tgt = random.choice(t_candidates)
    gt = compute_shortest_path(SG, src, tgt)

    if gt["status"] != "ok" or gt["distance"] is None:
        continue

    rec = {
        "id": f"FB_SP_{len(prompts):03d}",
        "task": "shortest_path",
        "dataset": "facebook_combined",
        "seed": seed,
        "graph": {
            "directed": False,
            "nodes": list(SG.nodes()),
            "edges": [{"u": u, "v": v} for u, v in SG.edges()],
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
        "distance": gt["distance"],
        "status": gt["status"]
    })

# ==========================
# SAVE OUTPUTS
# ==========================
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(prompts, f, indent=2, ensure_ascii=False)

pd.DataFrame(checks).to_csv(OUT_CSV, index=False)

print(f"✅ Done: {len(prompts)} prompts saved to {OUT_JSON}")
print(f"ℹ️ Attempts used: {attempts}/{max_attempts}")
