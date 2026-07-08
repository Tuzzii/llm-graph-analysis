"""
NLGraph Evaluation - Universal evaluator for all 4 tasks
Handles: connectivity, shortest_path, maximum_flow, bipartite_matching
"""

import json
import glob
import re
import pandas as pd
from pathlib import Path
import sys
sys.path.append('evaluation')
from logger import SimpleLogger

logger = SimpleLogger("nlgraph_eval.log")

def extract_yes_no(text):
    """Extract yes/no answer for connectivity"""
    if text is None:
        return None
    text = text.lower()
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return None

def extract_number(text):
    """Extract numeric answer for shortest_path, maximum_flow, bipartite_matching"""
    if text is None:
        return None
    
    # Try to find numbers in the text
    # Look for patterns like "answer is 5", "The answer is: 5", etc.
    patterns = [
        r'answer is[:\s]+(\d+)',
        r'answer:[:\s]+(\d+)',
        r'is[:\s]+(\d+)',
        r'^(\d+)$',  # Just a number
        r'(\d+)\s*\.'  # Number followed by period
    ]
    
    text_lower = text.lower().strip()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                return int(match.group(1))
            except:
                continue
    
    # Last resort: find any number in the text
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[-1])  # Take the last number found
    
    return None

def normalize_ground_truth(gt_text, task):
    """Convert ground_truth string to comparable format"""
    if task == "connectivity":
        # "The answer is yes." or "The answer is no."
        if "yes" in gt_text.lower():
            return "yes"
        if "no" in gt_text.lower():
            return "no"
        return None
    else:
        # Extract number from "The answer is X."
        num = extract_number(gt_text)
        return num

def evaluate_file(path, task):
    """Evaluate single file"""
    correct = 0
    total = 0
    invalid = 0
    
    rows = []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                
                # Get ground truth
                gt_raw = item.get("ground_truth", "")
                gt = normalize_ground_truth(gt_raw, task)
                
                # Get prediction
                output = item.get("model_output", "")
                if task == "connectivity":
                    pred = extract_yes_no(output)
                else:
                    pred = extract_number(output)
                
                # Check correctness
                total += 1
                is_invalid = pred is None
                if is_invalid:
                    invalid += 1
                
                is_correct = (pred == gt) if pred is not None else False
                if is_correct:
                    correct += 1
                
                rows.append({
                    "id": item.get("id", ""),
                    "task": task,
                    "pred": pred,
                    "gt": gt,
                    "correct": is_correct,
                    "invalid": is_invalid,
                    "elapsed": item.get("elapsed", 0)
                })
    
    except Exception as e:
        logger.log(f"Error evaluating {path}: {e}")
        return 0, 0, 0, pd.DataFrame()
    
    return correct, total, invalid, pd.DataFrame(rows)

def evaluate_all_nlgraph():
    """Evaluate all NLGraph results"""
    logger.log("="*80)
    logger.log("NLGraph Evaluation - All Tasks")
    logger.log("="*80)
    
    # Find all NLGraph result files
    files = glob.glob("results/answers/*/*/nlgraph_*.jsonl")
    
    if not files:
        logger.log("No NLGraph result files found!")
        return
    
    logger.log(f"Found {len(files)} NLGraph result files")
    
    all_results = []
    
    for filepath in sorted(files):
        path = Path(filepath)
        model = path.parent.parent.name
        task = path.parent.name
        prompt_type = path.stem.replace("nlgraph_", "")
        
        # Evaluate
        correct, total, invalid, df = evaluate_file(filepath, task)
        
        if total == 0:
            continue
        
        accuracy = correct / total * 100
        invalid_rate = invalid / total * 100
        
        result = {
            "model": model,
            "task": task,
            "prompt_type": prompt_type,
            "total": total,
            "correct": correct,
            "invalid": invalid,
            "accuracy": accuracy,
            "invalid_rate": invalid_rate
        }
        
        all_results.append(result)
        
        logger.log(f"\n{model}/{task}/{prompt_type}:")
        logger.log(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
        logger.log(f"  Invalid: {invalid_rate:.1f}% ({invalid}/{total})")
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(all_results)
    
    # Save to CSV
    output_path = "results/nlgraph_summary.csv"
    df_summary.to_csv(output_path, index=False)
    logger.log(f"\n✓ Saved summary to {output_path}")
    
    # Print summary by model
    logger.log("\n" + "="*80)
    logger.log("SUMMARY BY MODEL")
    logger.log("="*80)
    
    for model in sorted(df_summary["model"].unique()):
        model_df = df_summary[df_summary["model"] == model]
        avg_acc = model_df["accuracy"].mean()
        avg_invalid = model_df["invalid_rate"].mean()
        
        logger.log(f"\n{model}:")
        logger.log(f"  Average Accuracy: {avg_acc:.1f}%")
        logger.log(f"  Average Invalid: {avg_invalid:.1f}%")
        logger.log(f"  Experiments: {len(model_df)}")
    
    # Print summary by task
    logger.log("\n" + "="*80)
    logger.log("SUMMARY BY TASK")
    logger.log("="*80)
    
    for task in sorted(df_summary["task"].unique()):
        task_df = df_summary[df_summary["task"] == task]
        avg_acc = task_df["accuracy"].mean()
        
        logger.log(f"\n{task}:")
        logger.log(f"  Average Accuracy: {avg_acc:.1f}%")
        logger.log(f"  Models: {task_df['model'].nunique()}")
    
    logger.log("\n" + "="*80)
    logger.log("EVALUATION COMPLETE")
    logger.log("="*80)

if __name__ == "__main__":
    evaluate_all_nlgraph()
