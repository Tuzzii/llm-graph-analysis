# evaluation/prepare_for_annotation.py
import pandas as pd
cand = pd.read_csv("analysis/error_taxonomy_candidates.csv")
# choose top N candidates by priority:
# priority rules: disagreement across models (not implemented here) => choose high elapsed or parsed errors first
cand_sorted = cand.sort_values(by=["parsed_ok","correct","elapsed"], ascending=[True, True, False])
# choose 100
sample = cand_sorted.head(120)  # pick 120 to allow human drop to 50-100
cols = ["file","id","model","task","prompt_type","elapsed","model_output","ground_truth_raw","parsed_ok","pred_num","correct","auto_label"]
sample[cols].to_csv("analysis/error_taxonomy_to_label.csv", index=False, encoding="utf-8")
print("Wrote analysis/error_taxonomy_to_label.csv (", len(sample), "rows )")
