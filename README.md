# Hướng dẫn chạy thí nghiệm

## 1. Cài đặt môi trường

### Cài đặt Ollama (cho local models)
- Windows: Tải về từ https://ollama.com
- macOS: `brew install ollama`
- Linux: `curl https://ollama.ai/install.sh | sh`

### Cài đặt Python environment
```bash
conda create -n flenv python=3.10
conda activate flenv
pip install -r requirements.txt
```

### Setup Cloud Models (OpenAI) - Optional
Nếu muốn sử dụng GPT-3.5-turbo hoặc GPT-4o-mini:

1. Tạo API key tại https://platform.openai.com/api-keys
2. Tạo file `.env` trong thư mục gốc:
```bash
OPENAI_API_KEY=your_api_key_here
```
3. Cài đặt thư viện:
```bash
pip install openai python-dotenv
```

**Lưu ý**: Cloud models có thể bị giới hạn quota, khuyến khích dùng local models (Ollama) để tránh lỗi.

## 2. Cách chạy thí nghiệm

### Bước 1: Tải các models
```bash
python scripts/nlgraph/prepare_models.py
```
- Tự động tải 3 models: llama3.2-1b, qwen2.5-3b, phi3-medium
- Kiểm tra RAM trước khi tải

### Bước 2: Tạo prompts
```bash
python scripts/nlgraph/make_nlgraph_prompts.py
```
- Tạo prompts cho 4 tasks: connectivity, shortest_path, maximum_flow, bipartite_matching
- 3 prompt types: zero-shot, cot, cot-hidden
- Output: `results/prompts/`

### Bước 3: Chạy thí nghiệm
```bash
python scripts/nlgraph/run_all_nlgraph.py
```
- Chạy tất cả 3 models × 4 tasks × 3 prompt types
- Output: `results/answers/{model}/`
- Tự động bỏ qua các prompt đã chạy

### Bước 4: Đánh giá kết quả
```bash
python evaluation/evaluate_nlgraph.py
```
- Output: `results/nlgraph_summary.csv`

### Bước 5: So sánh IID vs OOD (optional)
```bash
python analysis/compare_iid_ood.py
```
- So sánh kết quả NLGraph (IID) vs Facebook (OOD)

## 3. Chạy từng phần (optional)

### Chạy 1 model cụ thể
```bash
python scripts/nlgraph/run_llm_nlgraph.py --model qwen2.5-3b --task connectivity --prompt_type cot
```

### Chạy một số tasks
```bash
python scripts/nlgraph/run_all_nlgraph.py --tasks connectivity shortest_path --models llama3.2-1b qwen2.5-3b
```

### Bỏ qua kiểm tra RAM (để chạy phi3-medium trên máy 16GB)
```bash
python scripts/nlgraph/run_all_nlgraph.py --models phi3-medium --skip-ram-check
```

## 4. Output logs

### Logs trong quá trình chạy
- Console output: Hiển thị progress và errors real-time
- Tự động skip các prompts đã có kết quả

### Kết quả thí nghiệm
- `results/answers/{model}/`: Các file JSONL chứa câu trả lời của model
- `results/nlgraph_summary.csv`: Bảng tổng hợp accuracy
- `results/csv/`: Chi tiết từng test case

### Experiment diary (thống kê tổng hợp)
```
Tổng số thí nghiệm hoàn thành: 8,266 queries
- 3 local models (llama3.2-1b, qwen2.5-3b, phi3-medium)
- 4 graph tasks (connectivity, shortest_path, maximum_flow, bipartite_matching)
- 3 prompt types (zero-shot, cot, cot-hidden)
- 577 test cases từ NLGraph benchmark

Kết quả chính:
- qwen2.5-3b: 23.0% accuracy (tốt nhất)
- phi3-medium: 22.9% accuracy
- llama3.2-1b: 21.1% accuracy

Hiệu suất theo task:
- Connectivity: 59.8% (dễ nhất)
- Shortest Path: 19.3%
- Bipartite Matching: 7.4%
- Maximum Flow: 2.7% (khó nhất)

Thời gian chạy:
- llama3.2-1b: ~30-60s/query
- qwen2.5-3b: ~60-120s/query
- phi3-medium: ~120-300s/query (timeout 300s)

Vấn đề gặp phải:
- OpenAI quota limited → chuyển sang Ollama local models
- phi3-medium RAM 30GB > 16GB → tự động skip trong batch mode
- Timeout cho large models → tăng lên 300s
```
