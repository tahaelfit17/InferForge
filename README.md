# InferForge

InferForge is a lightweight benchmarking and tuning tool for exploring local LLM inference performance with `llama.cpp`.

It automates GPU-layer offload benchmarking, measures inference throughput, detects performance saturation, and recommends an efficient GPU offload configuration instead of simply selecting the configuration with the highest observed throughput.

## Features

- Automatic GPU-layer (`NGL`) sweeps
- Manual GPU-layer selection
- Prompt-processing throughput measurement
- Token-generation throughput measurement
- CPU-only baseline comparison
- Automatic speedup calculation
- Saturation-aware configuration recommendation
- Configurable performance threshold
- Benchmark failure detection
- JSON result export
- Benchmark metadata collection

## How It Works

InferForge benchmarks a model across multiple GPU offload configurations.

For every tested NGL value it measures:

- **PP tok/s** — prompt-processing throughput
- **TG tok/s** — token-generation throughput
- **Speedup** — token-generation improvement relative to CPU-only inference

After the sweep, InferForge finds the maximum observed token-generation throughput and recommends the **earliest tested GPU-layer configuration that reaches the configured performance threshold**.

The default threshold is **99%**.

This avoids recommending additional GPU offload when it provides effectively no meaningful performance improvement.

## Current Benchmark

**Model:** `Qwen3-1.7B-GGUF:Q4_K_M`  
**GPU:** NVIDIA T1200 Laptop GPU — 4 GB VRAM  
**CPU:** Intel Core i7-11850H  
**Backend:** CUDA  
**llama.cpp build:** `10866 / 9113cc188`

| GPU layers | PP tok/s | TG tok/s | Speedup |
| ---: | ---: | ---: | ---: |
| 0 | 393.36 | 17.37 | 1.00x |
| 5 | 408.06 | 24.92 | 1.43x |
| 10 | 424.21 | 29.77 | 1.71x |
| 15 | 442.07 | 37.33 | 2.15x |
| 20 | 458.87 | 49.18 | 2.83x |
| 25 | 479.02 | 75.59 | 4.35x |
| 30 | 492.55 | 118.34 | 6.81x |
| 35 | 492.56 | 118.33 | 6.81x |
| 40 | 496.26 | 118.36 | 6.81x |

## Saturation Detection

The benchmark shows a clear token-generation saturation point at approximately **30 GPU layers**.

At NGL 30:

- **118.34 tok/s** token generation
- **492.55 tok/s** prompt processing
- **6.81x** generation speedup over CPU-only
- **99.99%** of the maximum observed token-generation throughput

Maximum observed token-generation throughput:

```text
118.36 tok/s
```

Increasing GPU offload from NGL 30 to 35 and 40 produced effectively no additional token-generation performance.

InferForge therefore recommends:

```text
Recommended GPU layers: 30
```

rather than blindly selecting the configuration with the numerically highest benchmark result.

## Usage

Run an automatic GPU-layer sweep:

```bash
python inferforge/runner.py --auto --step 5 --limit 40
```

Example output:

```text
NGL sweep summary
-------------------------------------------------------
  NGL     PP tok/s     TG tok/s    Speedup
-------------------------------------------------------
    0       393.36        17.37      1.00x
    5       408.06        24.92      1.43x
   10       424.21        29.77      1.71x
   15       442.07        37.33      2.15x
   20       458.87        49.18      2.83x
   25       479.02        75.59      4.35x
   30       492.55       118.34      6.81x
   35       492.56       118.33      6.81x
   40       496.26       118.36      6.81x

Recommended configuration
------------------------------------
GPU layers       : 30
PP tok/s         : 492.55
TG tok/s         : 118.34
Maximum observed : 118.36
Max performance  : 99.99%
Threshold        : 99.00%
Speedup          : 6.81x
```

## Results

Benchmark results are automatically saved as timestamped JSON files under:

```text
results/
```

The exported data can be used for later analysis, hardware comparisons, regression testing, or visualization.

## Project Status

InferForge is currently an early-stage experimental project focused on local LLM inference benchmarking and hardware-aware performance tuning.

Current development is centered around improving automatic configuration discovery, benchmarking reliability, and performance analysis.

## License

No license has been selected yet.
