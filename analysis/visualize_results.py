"""
Visualize NLGraph Results
Creates summary tables and insights
"""

import pandas as pd
import sys
sys.path.append('evaluation')
from logger import SimpleLogger

logger = SimpleLogger("visualize_results.log")

def create_pivot_tables():
    """Create pivot tables for easy analysis"""
    
    logger.log("="*80)
    logger.log("NLGraph Results Visualization")
    logger.log("="*80)
    
    # Load data
    df = pd.read_csv("results/nlgraph_summary.csv")
    
    # 1. Model × Task Pivot
    logger.log("\n### Table 1: Accuracy by Model × Task")
    logger.log("-" * 80)
    pivot_model_task = df.pivot_table(
        values='accuracy',
        index='model',
        columns='task',
        aggfunc='mean'
    ).round(1)
    
    for model in pivot_model_task.index:
        logger.log(f"\n{model}:")
        for task in pivot_model_task.columns:
            acc = pivot_model_task.loc[model, task]
            stars = "⭐" * int(acc / 20)  # Star rating
            logger.log(f"  {task:20s}: {acc:5.1f}% {stars}")
    
    # 2. Task × Prompt Type Pivot
    logger.log("\n" + "="*80)
    logger.log("### Table 2: Accuracy by Task × Prompt Type")
    logger.log("-" * 80)
    pivot_task_prompt = df.pivot_table(
        values='accuracy',
        index='task',
        columns='prompt_type',
        aggfunc='mean'
    ).round(1)
    
    for task in pivot_task_prompt.index:
        logger.log(f"\n{task}:")
        for ptype in pivot_task_prompt.columns:
            acc = pivot_task_prompt.loc[task, ptype]
            logger.log(f"  {ptype:15s}: {acc:5.1f}%")
    
    # 3. Model × Prompt Type Pivot
    logger.log("\n" + "="*80)
    logger.log("### Table 3: Accuracy by Model × Prompt Type")
    logger.log("-" * 80)
    pivot_model_prompt = df.pivot_table(
        values='accuracy',
        index='model',
        columns='prompt_type',
        aggfunc='mean'
    ).round(1)
    
    for model in pivot_model_prompt.index:
        logger.log(f"\n{model}:")
        for ptype in pivot_model_prompt.columns:
            acc = pivot_model_prompt.loc[model, ptype]
            logger.log(f"  {ptype:15s}: {acc:5.1f}%")
    
    # 4. Detailed Comparison Table
    logger.log("\n" + "="*80)
    logger.log("### Table 4: All Combinations (Sorted by Accuracy)")
    logger.log("-" * 80)
    
    df_sorted = df.sort_values('accuracy', ascending=False)
    logger.log(f"\n{'Rank':<6}{'Model':<15}{'Task':<20}{'Prompt':<15}{'Accuracy':<10}{'Total':<6}")
    logger.log("-" * 80)
    
    for idx, row in enumerate(df_sorted.itertuples(), 1):
        logger.log(f"{idx:<6}{row.model:<15}{row.task:<20}{row.prompt_type:<15}{row.accuracy:>6.1f}%    {row.total:<6}")
    
    # 5. Statistical Summary
    logger.log("\n" + "="*80)
    logger.log("### Table 5: Statistical Summary")
    logger.log("-" * 80)
    
    stats = df.groupby('model')['accuracy'].agg(['mean', 'std', 'min', 'max'])
    logger.log(f"\n{'Model':<15}{'Mean':<10}{'Std':<10}{'Min':<10}{'Max':<10}")
    logger.log("-" * 80)
    for model in stats.index:
        logger.log(f"{model:<15}{stats.loc[model, 'mean']:>8.1f}%  "
                  f"{stats.loc[model, 'std']:>8.1f}%  "
                  f"{stats.loc[model, 'min']:>8.1f}%  "
                  f"{stats.loc[model, 'max']:>8.1f}%")
    
    # 6. Insights
    logger.log("\n" + "="*80)
    logger.log("### Key Insights")
    logger.log("="*80)
    
    # Best model per task
    logger.log("\n🏆 Best Model per Task:")
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        best = task_df.loc[task_df['accuracy'].idxmax()]
        logger.log(f"  {task}: {best['model']} ({best['accuracy']:.1f}%) with {best['prompt_type']}")
    
    # Prompt impact
    logger.log("\n📊 Prompt Engineering Impact:")
    prompt_stats = df.groupby('prompt_type')['accuracy'].mean().sort_values(ascending=False)
    for ptype, acc in prompt_stats.items():
        logger.log(f"  {ptype}: {acc:.1f}% average")
    
    # Model size vs performance
    logger.log("\n🔍 Model Size Analysis:")
    model_params = {
        'llama3.2-1b': '1.2B',
        'qwen2.5-3b': '3B',
        'phi3-medium': '14B'
    }
    model_perf = df.groupby('model')['accuracy'].mean().sort_values(ascending=False)
    for model, acc in model_perf.items():
        params = model_params.get(model, 'Unknown')
        logger.log(f"  {model} ({params}): {acc:.1f}%")
    
    # Task difficulty
    logger.log("\n📈 Task Difficulty Ranking:")
    task_perf = df.groupby('task')['accuracy'].mean().sort_values(ascending=False)
    for idx, (task, acc) in enumerate(task_perf.items(), 1):
        if acc > 50:
            difficulty = "EASY"
        elif acc > 20:
            difficulty = "MEDIUM"
        elif acc > 10:
            difficulty = "HARD"
        else:
            difficulty = "VERY HARD"
        logger.log(f"  {idx}. {task}: {acc:.1f}% ({difficulty})")
    
    # Variance analysis
    logger.log("\n📉 Consistency Analysis (Lower is better):")
    variance = df.groupby('model')['accuracy'].std().sort_values()
    for model, std in variance.items():
        logger.log(f"  {model}: {std:.1f}% std deviation")
    
    logger.log("\n" + "="*80)
    logger.log("VISUALIZATION COMPLETE")
    logger.log("="*80)

if __name__ == "__main__":
    create_pivot_tables()
