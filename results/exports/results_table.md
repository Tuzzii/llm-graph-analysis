# NLGraph Results Table

## Performance by Model

| Model | Avg Accuracy | Connectivity | Shortest Path | Max Flow | Bipartite |
|-------|--------------|--------------|---------------|----------|----------|
| llama3.2-1b | 21.1% | 49.5% | 26.3% | 2.3% | 6.3% |
| phi3-medium | 22.9% | 67.5% | 15.6% | 1.7% | 6.7% |
| qwen2.5-3b | 23.0% | 62.5% | 16.1% | 4.0% | 9.1% |

## All Results

| Model | Task | Prompt Type | Accuracy | Total Cases |
|-------|------|-------------|----------|-------------|
| phi3-medium | connectivity | zero_shot | 69.3% | 371 |
| phi3-medium | connectivity | cot_hidden | 68.2% | 371 |
| phi3-medium | connectivity | cot | 65.0% | 371 |
| qwen2.5-3b | connectivity | cot | 64.7% | 371 |
| qwen2.5-3b | connectivity | cot_hidden | 63.3% | 371 |
| qwen2.5-3b | connectivity | zero_shot | 59.6% | 371 |
| llama3.2-1b | connectivity | cot | 51.8% | 371 |
| llama3.2-1b | connectivity | zero_shot | 48.5% | 371 |
| llama3.2-1b | connectivity | cot_hidden | 48.2% | 371 |
| llama3.2-1b | shortest_path | cot | 32.8% | 64 |
| qwen2.5-3b | shortest_path | cot_hidden | 28.1% | 64 |
| llama3.2-1b | shortest_path | cot_hidden | 23.4% | 64 |
| llama3.2-1b | shortest_path | zero_shot | 22.6% | 124 |
| phi3-medium | shortest_path | zero_shot | 18.8% | 64 |
| phi3-medium | shortest_path | cot_hidden | 17.2% | 64 |
| qwen2.5-3b | shortest_path | cot | 14.1% | 64 |
| qwen2.5-3b | bipartite_matching | cot_hidden | 13.1% | 84 |
| phi3-medium | shortest_path | cot | 10.9% | 64 |
| llama3.2-1b | bipartite_matching | zero_shot | 8.3% | 84 |
| phi3-medium | bipartite_matching | zero_shot | 8.3% | 84 |
| phi3-medium | bipartite_matching | cot_hidden | 8.3% | 84 |
| qwen2.5-3b | bipartite_matching | cot | 7.1% | 84 |
| qwen2.5-3b | bipartite_matching | zero_shot | 7.1% | 84 |
| qwen2.5-3b | shortest_path | zero_shot | 6.2% | 64 |
| llama3.2-1b | bipartite_matching | cot_hidden | 6.0% | 84 |
| qwen2.5-3b | maximum_flow | zero_shot | 5.2% | 58 |
| qwen2.5-3b | maximum_flow | cot_hidden | 5.2% | 58 |
| llama3.2-1b | bipartite_matching | cot | 4.8% | 84 |
| phi3-medium | bipartite_matching | cot | 3.6% | 84 |
| llama3.2-1b | maximum_flow | cot_hidden | 3.4% | 58 |
| llama3.2-1b | maximum_flow | zero_shot | 3.4% | 58 |
| phi3-medium | maximum_flow | cot | 3.4% | 58 |
| qwen2.5-3b | maximum_flow | cot | 1.7% | 58 |
| phi3-medium | maximum_flow | cot_hidden | 1.7% | 58 |
| llama3.2-1b | maximum_flow | cot | 0.0% | 58 |
| phi3-medium | maximum_flow | zero_shot | 0.0% | 58 |
