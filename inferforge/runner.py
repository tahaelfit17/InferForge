import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


LLAMA_BENCH = Path.home() / "Projects/llama.cpp/build-t1200/bin/llama-bench"
MODEL = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"
DEFAULT_LAYERS = [0, 1, 5, 10]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_benchmark(gpu_layers: int) -> list[dict]:
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


def summarize_run(gpu_layers: int, rows: list[dict]) -> dict:
    prompt = next((row for row in rows if row.get("n_prompt", 0) > 0), None)
    generation = next((row for row in rows if row.get("n_gen", 0) > 0), None)

    if prompt is None or generation is None:
        raise ValueError(f"Incomplete benchmark output for NGL={gpu_layers}")

    return {
        "gpu_layers": gpu_layers,
        "prompt_tps": prompt["avg_ts"],
        "generation_tps": generation["avg_ts"],
        "backend": generation["backends"],
        "gpu": generation["gpu_info"],
        "cpu": generation["cpu_info"],
        "model": generation["model_type"],
    }


def run_sweep(layers: list[int]) -> list[dict]:
    results = []

    for gpu_layers in layers:
        rows = run_benchmark(gpu_layers)
        results.append(summarize_run(gpu_layers, rows))

    baseline = results[0]["generation_tps"]

    for result in results:
        result["generation_speedup"] = result["generation_tps"] / baseline

    return results


def save_results(results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"ngl-sweep-{timestamp}.json"

    path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    return path


def print_summary(results: list[dict]) -> None:
    print("\nNGL sweep summary")
    print("-" * 55)
    print(f"{'NGL':>5} {'PP tok/s':>12} {'TG tok/s':>12} {'Speedup':>10}")
    print("-" * 55)

    for result in results:
        print(
            f"{result['gpu_layers']:>5} "
            f"{result['prompt_tps']:>12.2f} "
            f"{result['generation_tps']:>12.2f} "
            f"{result['generation_speedup']:>9.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers",
        default="0,1,5,10",
        help="comma-separated GPU layer counts",
    )

    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    results = run_sweep(layers)

    print_summary(results)

    output_path = save_results(results)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
