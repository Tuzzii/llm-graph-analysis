"""
make_mf_prompts_facebook.py
Author: Tran Le Vinh Buu
Purpose: Generate 50 Maximum Flow OOD prompts from Facebook dataset.
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

OUT_JSON = "buu_maxflow_facebook.json"
OUT_CSV  = "buu_mf_fb_checks.csv"

# ==========================
# LOAD FACEBOOK GRAPH
# ==========================
G_fb = nx.read_edgelist("data/facebook_combined.txt")
print(f"Facebook graph: {G_fb.number_of_nodes()} nodes, {G_fb.number_of_edges()} edges")

# Đổi node label thành string
G_fb = nx.relabel_nodes(G_fb, lambda x: str(x))

# ==========================
# HELPER FUNCTIONS
# ==========================
def get_directed_subgraph(G, center, size=25):
    """Sinh subgraph có hướng với capacity ngẫu nhiên."""
    nodes = list(nx.single_source_shortest_path_length(G, center, cutoff=2).keys())
    if len(nodes) > size:
        nodes = random.sample(nodes, size)
    SG_undirected = G.subgraph(nodes).copy()
    DG = nx.DiGraph()
    DG.add_nodes_from(SG_undirected.nodes())

    # thêm cả hai hướng cho mỗi cạnh
    for (u, v) in SG_undirected.edges():
        DG.add_edge(u, v, capacity=random.randint(1, 10))
        DG.add_edge(v, u, capacity=random.randint(1, 10))
    return DG


def make_question_text(nodes, edges, s, t):
    e_text = ", ".join([f"{u}->{v} (cap {d['capacity']})" for u, v, d in edges])
    return (
        f"In the directed graph with nodes {nodes} and edges {e_text}, "
        f"determine the maximum flow from source {s} to sink {t}."
    )


def compute_max_flow(DG, s, t):
    """Tính maximum flow và flow dict."""
    try:
        flow_value, flow_dict = nx.maximum_flow(DG, s, t)
        return {"flow_value": flow_value, "flow_dict": flow_dict, "status": "ok"}
    except nx.NetworkXUnbounded:
        return {"flow_value": None, "flow_dict": None, "status": "unbounded"}
    except nx.NetworkXError as e:
        return {"flow_value": None, "flow_dict": None, "status": str(e)}
    except Exception as e:
        return {"flow_value": None, "flow_dict": None, "status": str(e)}

# ==========================
# GENERATE PROMPTS
# ==========================
prompts, checks = [], []
nodes_all = list(G_fb.nodes())
os.makedirs("results", exist_ok=True)

print("Generating Facebook Maximum Flow prompts...")

attempts = 0
max_attempts = NUM_PROMPTS * 10

while len(prompts) < NUM_PROMPTS and attempts < max_attempts:
    attempts += 1
    seed = SEED + attempts * 31
    random.seed(seed)

    center = random.choice(nodes_all)
    DG = get_directed_subgraph(G_fb, center, size=random.randint(20, 35))

    if DG.number_of_nodes() < 3:
        continue

    src = random.choice(list(DG.nodes()))
    reachable_nodes = nx.single_source_shortest_path_length(DG, src)
    t_candidates = [n for n in reachable_nodes.keys() if n != src]
    if not t_candidates:
        continue
    tgt = random.choice(t_candidates)

    gt = compute_max_flow(DG, src, tgt)
    if gt["status"] != "ok" or gt["flow_value"] is None:
        continue

    rec = {
        "id": f"FB_MF_{len(prompts):03d}",
        "task": "maximum_flow",
        "dataset": "facebook_combined",
        "seed": seed,
        "graph": {
            "directed": True,
            "nodes": list(DG.nodes()),
            "edges": [{"u": u, "v": v, "capacity": d["capacity"]} for u, v, d in DG.edges(data=True)],
            "source": src,
            "target": tgt
        },
        "question": make_question_text(list(DG.nodes()), list(DG.edges(data=True)), src, tgt),
        "ground_truth": gt,
        "notes": {
            "node_count": DG.number_of_nodes(),
            "edge_count": DG.number_of_edges(),
            "timestamp": datetime.now().isoformat()
        }
    }

    prompts.append(rec)
    checks.append({
        "id": rec["id"],
        "n_nodes": DG.number_of_nodes(),
        "n_edges": DG.number_of_edges(),
        "flow_value": gt["flow_value"],
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
