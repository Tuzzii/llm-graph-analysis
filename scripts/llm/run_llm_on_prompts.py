"""
run_llm_on_prompts.py
Author: Tran Le Vinh Buu (Nhóm 17)
Purpose:
    Run Zero-shot, CoT, CoT-hidden on all 200 OOD prompts (4 tasks),
    using all configured models (OpenAI + Ollama).
"""

import os, json, time
from tqdm import tqdm

from llm_utils import call_openai_model, call_ollama_model, save_jsonl
from prompt_templates import zero_shot_prompt, cot_prompt, cot_hidden_prompt
from config import OPENAI_MODELS, OLLAMA_MODELS

BASE_DIR = "results/prompts/"
TASK_MAP = {
    "connectivity": "buu_connectivity_facebook.json",
    "shortest_path": "buu_shortestpath_facebook.json",
    "maximum_flow": "buu_maxflow_facebook.json",
    "bipartite_matching": "buu_bipartite_facebook.json"
}

PROMPT_FUNCS = {
    "zero_shot": zero_shot_prompt,
    "cot": cot_prompt,
    "cot_hidden": cot_hidden_prompt
}

def load_prompts(task):
    path = os.path.join(BASE_DIR, TASK_MAP[task])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def count_existing_records(filepath):
    """Đếm số lượng bản ghi đã được xử lý trong file JSONL"""
    if not os.path.exists(filepath):
        return 0
    count = 0
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception as e:
        print(f"⚠ Warning: Error reading {filepath}: {e}")
        return 0
    return count

def get_processed_ids(filepath):
    """Lấy danh sách ID đã được xử lý"""
    if not os.path.exists(filepath):
        return set()
    processed = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    processed.add(record.get("id"))
    except Exception as e:
        print(f"⚠ Warning: Error reading {filepath}: {e}")
        return set()
    return processed

def run_single_query(model_name, task, prompt_type, prompt_text):
    if "gpt" in model_name:
        answer = call_openai_model(model_name, prompt_text)
    else:
        answer = call_ollama_model(model_name, prompt_text)
    return answer

def run_model_on_task(model_name, task):
    prompts = load_prompts(task)
    total_prompts = len(prompts)

    for prompt_type, build_prompt in PROMPT_FUNCS.items():
        # Sanitize model name for Windows file system (replace : with -)
        safe_model_name = model_name.replace(":", "-")
        outpath = f"results/answers/{safe_model_name}/{task}/{prompt_type}.jsonl"

        # Kiểm tra những prompt đã được xử lý
        processed_ids = get_processed_ids(outpath)
        already_done = len(processed_ids)
        
        print(f"\n▶ MODEL={model_name} | TASK={task} | MODE={prompt_type}")
        print(f"Saving → {outpath}")
        print(f"📊 Progress: {already_done}/{total_prompts} prompts already completed")
        
        # Nếu đã xử lý đủ, bỏ qua
        if already_done >= total_prompts:
            print(f"✓ Already completed! Skipping...")
            continue
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        
        # Chỉ xử lý những prompt chưa hoàn thành
        remaining_prompts = [p for p in prompts if p["id"] not in processed_ids]
        print(f"🔄 Processing {len(remaining_prompts)} remaining prompts...")

        for item in tqdm(remaining_prompts, desc=f"{model_name}-{task}-{prompt_type}"):
            q = item["question"]
            full_prompt = build_prompt(q)

            t0 = time.time()
            output = run_single_query(model_name, task, prompt_type, full_prompt)
            t1 = time.time()

            record = {
                "id": item["id"],
                "model": model_name,
                "task": task,
                "prompt_type": prompt_type,
                "question": q,
                "ground_truth": item["ground_truth"],
                "model_output": output,
                "elapsed": t1 - t0,
            }

            save_jsonl(outpath, record)
        
        print(f"✅ Completed: {prompt_type}")

def run_all():
    # OPENAI MODELS
    for model in OPENAI_MODELS:
        for task in TASK_MAP.keys():
            run_model_on_task(model, task)

    # LOCAL MODELS (OLLAMA)
    for model in OLLAMA_MODELS:
        for task in TASK_MAP.keys():
            run_model_on_task(model, task)

if __name__ == "__main__":
    run_all()
