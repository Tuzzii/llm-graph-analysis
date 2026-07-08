"""
Auto-annotate error samples with heuristic-based labels.
Manual review recommended after running this.
"""
import pandas as pd
import json
import re
import sys
import os
sys.path.append('evaluation')
from logger import SimpleLogger

logger = SimpleLogger("auto_annotate_samples")
logger.log_start()

INPUT_FILE = "analysis/error_taxonomy_candidates.csv"
OUTPUT_FILE = "analysis/error_taxonomy_annotated.csv"

# Load data
logger.log(f"Loading data from {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, encoding="utf-8")
logger.log(f"Loaded {len(df)} error cases")

def annotate_row(row):
    """Apply heuristic rules to generate annotations"""
    final_labels = []
    severity = 3  # Default medium severity
    comment = ""
    representative = "no"
    
    output = str(row.get("model_output", ""))
    task = row.get("task", "")
    parsed_ok = row.get("parsed_ok", False)
    correct = row.get("correct", False)
    auto_label = str(row.get("auto_label", ""))
    
    # Rule 1: Format errors
    if not parsed_ok or "format_error" in auto_label:
        final_labels.append("format_error")
        if output.strip() == "" or output == "None":
            final_labels.append("no_answer")
            severity = 5
            comment = "Model produced no output"
            representative = "yes"
        elif len(output) > 1000:
            final_labels.append("excessive_verbosity")
            severity = 3
            comment = "Output too verbose, no clear answer"
        else:
            severity = 4
            comment = "Invalid format, cannot parse answer"
    
    # Rule 2: Wrong value
    elif not correct and parsed_ok:
        final_labels.append("wrong_value")
        
        if "off_by_one" in auto_label:
            final_labels.append("off_by_one")
            severity = 2
            comment = "Answer off by 1, possible boundary error"
            representative = "yes"
        else:
            # Check if hallucination (random guess)
            pred_num = row.get("pred_num")
            try:
                gt_data = json.loads(row.get("ground_truth_raw", "{}"))
                gt_num = (gt_data.get("distance") or 
                         gt_data.get("flow_value") or 
                         gt_data.get("matching_size") or 
                         gt_data.get("match_size"))
                
                if pred_num and gt_num:
                    diff_ratio = abs(pred_num - gt_num) / max(gt_num, 1)
                    if diff_ratio > 2:
                        final_labels.append("hallucination")
                        severity = 4
                        comment = f"Predicted {pred_num} vs actual {gt_num}, large error"
                        representative = "yes"
                    else:
                        final_labels.append("computational_error")
                        severity = 3
                        comment = f"Predicted {pred_num} vs actual {gt_num}"
            except:
                pass
    
    # Rule 3: Task-specific patterns
    if task == "connectivity":
        if "yes" in output.lower() and "no" in output.lower():
            if "ambiguous_answer" not in final_labels:
                final_labels.append("ambiguous_answer")
            severity = 4
            comment = "Output contains both YES and NO"
    
    # Rule 4: High latency
    elapsed = row.get("elapsed")
    if elapsed and elapsed > 100:
        if "timeout" not in final_labels:
            final_labels.append("high_latency")
        severity = max(severity, 2)
        comment += f" | High latency: {elapsed:.1f}s"
    
    # Default if no labels
    if not final_labels:
        final_labels.append("unknown_error")
        severity = 3
        comment = "Needs manual review"
    
    return {
        "final_labels": ";".join(final_labels),
        "severity": severity,
        "comment": comment,
        "representative": representative
    }

# Apply annotations
print("Annotating rows...")
logger.log("Applying heuristic-based annotations")
annotations = df.apply(annotate_row, axis=1, result_type="expand")

# Merge with original data
df_annotated = pd.concat([df, annotations], axis=1)
logger.log("Annotations applied successfully")

# Select representative samples (up to 50)
logger.log("Selecting representative samples")
representatives = df_annotated[df_annotated["representative"] == "yes"]
if len(representatives) > 50:
    # Sample diverse errors
    representatives = representatives.groupby(["task", "model"]).head(2)
    if len(representatives) > 50:
        representatives = representatives.sample(n=50, random_state=42)

# Mark remaining as not representative
df_annotated.loc[~df_annotated.index.isin(representatives.index), "representative"] = "no"

# Save
logger.log(f"Saving annotated data to {OUTPUT_FILE}")
df_annotated.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

logger.log(f"Total annotated rows: {len(df_annotated)}")
logger.log(f"Representative samples: {len(representatives)}")

label_dist = df_annotated["final_labels"].value_counts().head(10).to_dict()
logger.log(f"Top 10 label distribution: {label_dist}")

severity_dist = df_annotated["severity"].value_counts().sort_index().to_dict()
logger.log(f"Severity distribution: {severity_dist}")

logger.log_complete()

print(f"\n✅ Annotated file saved: {OUTPUT_FILE}")
print(f"📊 Total rows: {len(df_annotated)}")
print(f"⭐ Representative samples: {len(representatives)}")
print("\nLabel distribution:")
print(df_annotated["final_labels"].value_counts().head(10))
print("\nSeverity distribution:")
print(df_annotated["severity"].value_counts().sort_index())
print("\n⚠️  Please review and adjust annotations manually in Excel/Google Sheets")
