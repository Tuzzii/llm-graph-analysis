"""
Export Results to Multiple Formats
Creates Excel, CSV, and text summaries
"""

import pandas as pd
from pathlib import Path

def export_all_formats():
    """Export results to various formats"""
    
    print("="*80)
    print("Exporting NLGraph Results")
    print("="*80)
    
    # Load data
    df = pd.read_csv("results/nlgraph_summary.csv")
    
    # Create exports directory
    export_dir = Path("results/exports")
    export_dir.mkdir(exist_ok=True)
    
    # 1. Excel with multiple sheets
    print("\n[1/5] Creating Excel file...")
    excel_path = export_dir / "nlgraph_results.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # All results
        df.to_excel(writer, sheet_name='All Results', index=False)
        
        # Pivot: Model × Task
        pivot_model_task = df.pivot_table(
            values='accuracy',
            index='model',
            columns='task',
            aggfunc='mean'
        ).round(1)
        pivot_model_task.to_excel(writer, sheet_name='Model vs Task')
        
        # Pivot: Task × Prompt
        pivot_task_prompt = df.pivot_table(
            values='accuracy',
            index='task',
            columns='prompt_type',
            aggfunc='mean'
        ).round(1)
        pivot_task_prompt.to_excel(writer, sheet_name='Task vs Prompt')
        
        # Summary statistics
        stats = df.groupby('model')['accuracy'].agg(['mean', 'std', 'min', 'max']).round(1)
        stats.to_excel(writer, sheet_name='Statistics')
        
        # Top 10 best
        top10 = df.nlargest(10, 'accuracy')[['model', 'task', 'prompt_type', 'accuracy', 'total']]
        top10.to_excel(writer, sheet_name='Top 10', index=False)
        
        # Bottom 10 worst
        bottom10 = df.nsmallest(10, 'accuracy')[['model', 'task', 'prompt_type', 'accuracy', 'total']]
        bottom10.to_excel(writer, sheet_name='Bottom 10', index=False)
    
    print(f"✓ Saved: {excel_path}")
    
    # 2. Detailed CSV
    print("\n[2/5] Creating detailed CSV...")
    csv_path = export_dir / "detailed_results.csv"
    df_detailed = df.sort_values(['model', 'task', 'prompt_type'])
    df_detailed.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # 3. Summary text report
    print("\n[3/5] Creating text summary...")
    txt_path = export_dir / "summary.txt"
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("NLGraph Evaluation Summary\n")
        f.write("="*80 + "\n\n")
        
        # Overall stats
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Experiments: {len(df)}\n")
        f.write(f"Total Prompts: {df['total'].sum()}\n")
        f.write(f"Average Accuracy: {df['accuracy'].mean():.1f}%\n")
        f.write(f"Best Accuracy: {df['accuracy'].max():.1f}%\n")
        f.write(f"Worst Accuracy: {df['accuracy'].min():.1f}%\n\n")
        
        # By model
        f.write("PERFORMANCE BY MODEL\n")
        f.write("-"*80 + "\n")
        for model in sorted(df['model'].unique()):
            model_df = df[df['model'] == model]
            avg = model_df['accuracy'].mean()
            f.write(f"{model}: {avg:.1f}% average\n")
            for task in sorted(model_df['task'].unique()):
                task_acc = model_df[model_df['task'] == task]['accuracy'].mean()
                f.write(f"  {task}: {task_acc:.1f}%\n")
            f.write("\n")
        
        # By task
        f.write("PERFORMANCE BY TASK\n")
        f.write("-"*80 + "\n")
        for task in sorted(df['task'].unique()):
            task_df = df[df['task'] == task]
            avg = task_df['accuracy'].mean()
            f.write(f"{task}: {avg:.1f}% average\n")
            for model in sorted(task_df['model'].unique()):
                model_acc = task_df[task_df['model'] == model]['accuracy'].mean()
                f.write(f"  {model}: {model_acc:.1f}%\n")
            f.write("\n")
        
        # Top 5
        f.write("TOP 5 BEST COMBINATIONS\n")
        f.write("-"*80 + "\n")
        top5 = df.nlargest(5, 'accuracy')
        for idx, row in enumerate(top5.itertuples(), 1):
            f.write(f"{idx}. {row.model} / {row.task} / {row.prompt_type}: {row.accuracy:.1f}%\n")
    
    print(f"✓ Saved: {txt_path}")
    
    # 4. Markdown table
    print("\n[4/5] Creating markdown table...")
    md_path = export_dir / "results_table.md"
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# NLGraph Results Table\n\n")
        
        # Summary by model
        f.write("## Performance by Model\n\n")
        f.write("| Model | Avg Accuracy | Connectivity | Shortest Path | Max Flow | Bipartite |\n")
        f.write("|-------|--------------|--------------|---------------|----------|----------|\n")
        
        for model in sorted(df['model'].unique()):
            model_df = df[df['model'] == model]
            avg = model_df['accuracy'].mean()
            conn = model_df[model_df['task'] == 'connectivity']['accuracy'].mean()
            sp = model_df[model_df['task'] == 'shortest_path']['accuracy'].mean()
            mf = model_df[model_df['task'] == 'maximum_flow']['accuracy'].mean()
            bm = model_df[model_df['task'] == 'bipartite_matching']['accuracy'].mean()
            f.write(f"| {model} | {avg:.1f}% | {conn:.1f}% | {sp:.1f}% | {mf:.1f}% | {bm:.1f}% |\n")
        
        f.write("\n## All Results\n\n")
        f.write("| Model | Task | Prompt Type | Accuracy | Total Cases |\n")
        f.write("|-------|------|-------------|----------|-------------|\n")
        
        for row in df.sort_values('accuracy', ascending=False).itertuples():
            f.write(f"| {row.model} | {row.task} | {row.prompt_type} | {row.accuracy:.1f}% | {row.total} |\n")
    
    print(f"✓ Saved: {md_path}")
    
    # 5. JSON export
    print("\n[5/5] Creating JSON export...")
    json_path = export_dir / "results.json"
    
    results_dict = {
        "summary": {
            "total_experiments": len(df),
            "total_prompts": int(df['total'].sum()),
            "average_accuracy": float(df['accuracy'].mean()),
            "best_accuracy": float(df['accuracy'].max()),
            "worst_accuracy": float(df['accuracy'].min())
        },
        "by_model": {},
        "by_task": {},
        "all_results": df.to_dict(orient='records')
    }
    
    # Add model summaries
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        results_dict["by_model"][model] = {
            "average_accuracy": float(model_df['accuracy'].mean()),
            "experiments": len(model_df),
            "by_task": {
                task: float(model_df[model_df['task'] == task]['accuracy'].mean())
                for task in model_df['task'].unique()
            }
        }
    
    # Add task summaries
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        results_dict["by_task"][task] = {
            "average_accuracy": float(task_df['accuracy'].mean()),
            "cases": int(task_df['total'].iloc[0]),
            "by_model": {
                model: float(task_df[task_df['model'] == model]['accuracy'].mean())
                for model in task_df['model'].unique()
            }
        }
    
    import json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"✓ Saved: {json_path}")
    
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"\nAll files saved to: {export_dir}")
    print("\nExported formats:")
    print("  - Excel (multi-sheet)")
    print("  - CSV (detailed)")
    print("  - Text summary")
    print("  - Markdown table")
    print("  - JSON")

if __name__ == "__main__":
    export_all_formats()
