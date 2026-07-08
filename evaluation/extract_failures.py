# evaluation/extract_failures.py
import os, glob, json, re, csv
from collections import defaultdict
import pandas as pd
from logger import SimpleLogger

logger = SimpleLogger("extract_failures")
logger.log_start()

ANS_DIR = "results/answers"
OUT_CAND = "analysis/error_taxonomy_candidates.csv"
os.makedirs("analysis", exist_ok=True)

def parse_number(text):
    if not text: return None
    nums = re.findall(r"-?\d+", text)
    if not nums: return None
    return int(nums[0])

def is_parsable(item):
    mo = item.get("model_output")
    if mo is None: return False
    mo = str(mo).strip()
    # for connectivity look for yes/no
    if item["task"] == "connectivity":
        t = mo.lower()
        if "yes" in t or "no" in t:
            return True
        return False
    # for others look for any number
    return parse_number(mo) is not None

rows = []
for path in glob.glob(f"{ANS_DIR}/*/*/*.jsonl"):
    # Handle both forward and backward slashes (Windows)
    path_normalized = path.replace("\\", "/")
    parts = path_normalized.split("/")
    if len(parts) < 5:
        continue
    model = parts[2]
    task = parts[3]
    promptfile = parts[-1]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            gt = item.get("ground_truth")
            parsed_ok = is_parsable(item)
            pred_num = None
            if item.get("model_output"):
                pred_num = parse_number(item["model_output"])
            # determine correctness for constant types (use strict comparison when possible)
            correct = False
            if task == "connectivity":
                # consider None as unknown; ground_truth uses 'connected'
                if parsed_ok:
                    txt = item["model_output"].lower()
                    pred_bool = "yes" in txt
                    correct = (pred_bool == bool(gt.get("connected")))
            elif task == "shortest_path":
                if pred_num is not None and gt.get("distance") is not None:
                    correct = (pred_num == gt.get("distance"))
            elif task == "maximum_flow":
                if pred_num is not None and gt.get("flow_value") is not None:
                    correct = (pred_num == gt.get("flow_value"))
            elif task == "bipartite_matching":
                # ground truth key: could be match_size or matching_size
                ms = gt.get("match_size") or gt.get("matching_size") or gt.get("match_size")
                if pred_num is not None and ms is not None:
                    correct = (pred_num == ms)
            rows.append({
                "file": path,
                "id": item.get("id"),
                "model": model,
                "task": task,
                "prompt_type": item.get("prompt_type"),
                "elapsed": item.get("elapsed"),
                "parsed_ok": parsed_ok,
                "pred_num": pred_num,
                "ground_truth_raw": json.dumps(gt, ensure_ascii=False),
                "correct": correct,
                "model_output": item.get("model_output")
            })

df = pd.DataFrame(rows)
# Basic heuristics: failures = not correct OR not parsed
candidates = df[(df["correct"]==False) | (df["parsed_ok"]==False)].copy()

# add simple auto-suggest labels (format_error if not parsed, wrong_value if parsed but wrong)
def auto_label(row):
    labels = []
    if not row["parsed_ok"]:
        labels.append("format_error")
    elif not row["correct"]:
        labels.append("wrong_value")
        # off_by_one
        if row["pred_num"] is not None:
            try:
                gt = int(json.loads(row["ground_truth_raw"]).get("distance") or
                         json.loads(row["ground_truth_raw"]).get("flow_value") or
                         json.loads(row["ground_truth_raw"]).get("match_size"))
            except:
                gt = None
            if gt is not None and row["pred_num"] is not None and abs(row["pred_num"] - gt) == 1:
                labels.append("off_by_one")
    # high latency
    if row["elapsed"] is None:
        pass
    else:
        # mark high_latency later in aggregation
        pass
    return ";".join(labels)

logger.log(f"Applying auto-labels to {len(candidates)} failure cases")
candidates["auto_label"] = candidates.apply(auto_label, axis=1)
candidates = candidates.sort_values(by=["model","task","parsed_ok","correct"], ascending=[True,True,True,True])

logger.log(f"Saving to {OUT_CAND}")
candidates.to_csv(OUT_CAND, index=False, encoding="utf-8")

logger.log(f"Total candidates: {len(candidates)}")
logger.log(f"Breakdown by model: {candidates['model'].value_counts().to_dict()}")
logger.log(f"Breakdown by task: {candidates['task'].value_counts().to_dict()}")
logger.log_complete()

print("Wrote candidates to", OUT_CAND)
print("Total candidates:", len(candidates))
