"""
Run evaluation for all tasks + summary table.
"""

import subprocess
import pandas as pd
import glob, os, json, re
from logger import SimpleLogger

TASKS = ["connectivity", "shortest_path", "maximum_flow", "bipartite_matching"]

if __name__ == "__main__":
    logger = SimpleLogger("evaluate_all")
    logger.log_start()
    
    print("\n▶ Running all evaluation scripts...\n")
    logger.log("Running individual task evaluations")

    subprocess.call("python evaluation/evaluate_connectivity.py", shell=True)
    subprocess.call("python evaluation/evaluate_shortestpath.py", shell=True)
    subprocess.call("python evaluation/evaluate_maxflow.py", shell=True)
    subprocess.call("python evaluation/evaluate_bipartite.py", shell=True)

    print("\n▶ Building summary table...\n")
    logger.log("Building summary table")

    rows = []
    for f in glob.glob("results/answers/*/*/*.jsonl.csv"):
        # Handle both forward and backward slashes
        parts = f.replace("\\", "/").split("/")
        if len(parts) < 5:
            continue
        model = parts[2]
        task = parts[3]
        prompt = parts[4].replace(".jsonl.csv","")

        df = pd.read_csv(f)
        acc = df["correct"].mean()
        invalid = df["invalid"].mean()
        latency = df["elapsed"].mean()

        rows.append({
            "model": model,
            "task": task,
            "prompt": prompt,
            "accuracy": acc,
            "invalid_rate": invalid,
            "avg_latency": latency
        })

    out = pd.DataFrame(rows)
    out.to_csv("results/summary_all.csv", index=False)
    
    logger.log(f"Summary table created with {len(out)} rows")
    logger.log(f"Saved to: results/summary_all.csv")
    logger.log(f"\nModels evaluated: {out['model'].unique().tolist()}")
    logger.log(f"Tasks evaluated: {out['task'].unique().tolist()}")
    logger.log_complete()
    
    print(out)
