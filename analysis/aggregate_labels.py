# evaluation/aggregate_labels.py
import pandas as pd
import os
import sys
sys.path.append('evaluation')
from logger import SimpleLogger

logger = SimpleLogger("aggregate_labels")
logger.log_start()

os.makedirs("analysis/figs", exist_ok=True)

logger.log("Loading annotated data")
df = pd.read_csv("analysis/error_taxonomy_annotated.csv")
logger.log(f"Loaded {len(df)} annotated error cases")

# ensure 'final_labels' exists
if "final_labels" not in df.columns:
    logger.log("ERROR: 'final_labels' column not found", "ERROR")
    raise SystemExit("Please annotate file: add 'final_labels' column")

# explode labels
logger.log("Exploding semicolon-separated labels")
df["labels_list"] = df["final_labels"].fillna("").apply(lambda s: [x.strip() for x in s.split(";") if x.strip()])
all_labels = {}
for idx, row in df.iterrows():
    for lab in row["labels_list"]:
        all_labels[lab] = all_labels.get(lab, 0) + 1

logger.log(f"Found {len(all_labels)} unique labels across all cases")
summary = pd.DataFrame.from_dict(all_labels, orient="index", columns=["count"]).sort_values("count", ascending=False)
summary["percent"] = summary["count"] / len(df) * 100
summary.to_csv("analysis/error_label_distribution.csv")
logger.log("Label distribution written to analysis/error_label_distribution.csv")
logger.log(f"Top 5 labels: {summary.head().to_dict()['count']}")
print(summary)

# example per label
logger.log("Generating representative examples for each label")
examples = []
for lab in summary.index:
    sub = df[df["labels_list"].apply(lambda L: lab in L)]
    if len(sub) == 0: continue
    ex = sub.iloc[0]
    examples.append({
        "label": lab,
        "example_id": ex["id"],
        "model": ex["model"],
        "task": ex["task"],
        "model_output": ex["model_output"],
        "ground_truth": ex["ground_truth_raw"],
        "comment": ex.get("comment","")
    })
pd.DataFrame(examples).to_csv("analysis/error_examples_summary.csv", index=False)
logger.log(f"Wrote {len(examples)} examples to analysis/error_examples_summary.csv")
print("Wrote examples to analysis/error_examples_summary.csv")

logger.log_complete()
