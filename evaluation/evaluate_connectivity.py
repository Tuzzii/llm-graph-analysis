"""
Connectivity Evaluation
Success = model answers YES if reachable, NO if not reachable.
"""

import json, glob, re
import pandas as pd
import networkx as nx

def extract_answer(text):
    if text is None:
        return None
    text = text.lower()
    if "yes" in text: return True
    if "no" in text: return False
    return None

def evaluate_file(path):
    correct = 0
    total = 0
    invalid = 0

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            gt = item["ground_truth"]["connected"]
            pred = extract_answer(item["model_output"])

            total += 1
            if pred is None:
                invalid += 1
            if pred == gt:
                correct += 1

            rows.append({
                "id": item["id"],
                "pred": pred,
                "gt": gt,
                "correct": pred == gt,
                "invalid": pred is None,
                "elapsed": item["elapsed"]
            })

    return correct, total, invalid, pd.DataFrame(rows)

if __name__ == "__main__":
    files = glob.glob("results/answers/*/connectivity/*.jsonl")
    for f in files:
        correct, total, invalid, df = evaluate_file(f)
        print("\n====", f)
        print("Accuracy:", correct/total)
        print("Invalid rate:", invalid/total)
        df.to_csv(f + ".csv", index=False)
