"""
Bipartite Matching Evaluation
Success if maximum matching size is correct.
"""

import json, glob, re
import pandas as pd

def extract_value(text):
    if text is None:
        return None
    nums = re.findall(r"\d+", text)
    if not nums: return None
    return int(nums[0])

def evaluate_file(path):
    correct = 0
    total = 0
    invalid = 0

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            gt = item["ground_truth"].get("matching_size") or item["ground_truth"].get("max_matching")
            pred = extract_value(item["model_output"])

            total += 1
            if pred is None:
                invalid += 1

            rows.append({
                "id": item["id"],
                "pred": pred,
                "gt": gt,
                "correct": pred == gt,
                "invalid": pred is None,
                "elapsed": item["elapsed"]
            })

            if pred == gt:
                correct += 1

    return correct, total, invalid, pd.DataFrame(rows)

if __name__ == "__main__":
    files = glob.glob("results/answers/*/bipartite_matching/*.jsonl")

    for f in files:
        correct, total, invalid, df = evaluate_file(f)
        print("\n====", f)
        print("Accuracy:", correct/total)
        print("Invalid:", invalid/total)
        df.to_csv(f + ".csv", index=False)
