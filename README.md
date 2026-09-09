# InferForge

InferForge is a lightweight benchmarking tool for exploring local LLM inference performance with `llama.cpp`.

It automates GPU-layer offload sweeps, measures prompt processing and token generation throughput, compares speedups, and saves benchmark results for later analysis.

## Current benchmark

Model: `Qwen3-1.7B-GGUF:Q4_K_M`  
GPU: `NVIDIA T1200 Laptop GPU`  
Backend: `CUDA`

| GPU layers | PP tok/s | TG tok/s | Speedup |
| ---: | ---: | ---: | ---: |
| 0 | 394.00 | 16.18 | 1.00x |
| 1 | 395.29 | 20.60 | 1.27x |
| 5 | 405.12 | 22.90 | 1.42x |
| 10 | 423.26 | 29.15 | 1.80x |
| 15 | 441.14 | 36.02 | 2.23x |
| 20 | 459.06 | 49.74 | 3.07x |

## Usage

```bash
python inferforge/runner.py
