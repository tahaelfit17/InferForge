import subprocess
import json
from pathlib import Path


LLAMA_BENCH = Path.home() / "Projects/llama.cpp/build-t1200/bin/llama-bench"

MODEL = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"


def run_benchmark(gpu_layers: int):
    command = [
        str(LLAMA_BENCH),
        "-hf", MODEL,
        "-ngl", str(gpu_layers),
        "-p", "512",
        "-n", "128",
        "-r", "3",
        "-o", "json",
    ]

    print(f"Benchmarking NGL={gpu_layers}...")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)


if __name__ == "__main__":
    data = run_benchmark(10)
    print(json.dumps(data, indent=2))
