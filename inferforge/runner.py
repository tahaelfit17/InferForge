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
    prompt = next(
        (row for row in rows if row.get("n_prompt", 0) > 0),
        None,
    )

    generation = next(
        (row for row in rows if row.get("n_gen", 0) > 0),
        None,
    )

    if prompt is None or generation is None:
        raise ValueError(
            f"Incomplete benchmark output for NGL={gpu_layers}"
        )

    return {
        "gpu_layers": gpu_layers,
        "prompt_tps": prompt["avg_ts"],
        "generation_tps": generation["avg_ts"],
        "backend": generation["backends"],
        "gpu": generation["gpu_info"],
        "cpu": generation["cpu_info"],
        "model": generation["model_type"],
    }


def generate_auto_layers(
    start: int = 0,
    step: int = 5,
    limit: int = 40,
) -> list[int]:
    return list(range(start, limit + 1, step))


def run_sweep(layers: list[int]) -> list[dict]:
    results = []

    for gpu_layers in layers:
        try:
            rows = run_benchmark(gpu_layers)
            results.append(
                summarize_run(gpu_layers, rows)
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"\nBenchmark failed at NGL={gpu_layers} "
                f"(exit code {exc.returncode})."
            )

            if exc.stderr:
                print(exc.stderr.strip())

            print(
                "Stopping automatic sweep at the first failing configuration."
            )
            break

    if not results:
        raise RuntimeError("No benchmark completed successfully.")

    baseline = results[0]["generation_tps"]

    for result in results:
        result["generation_speedup"] = (
            result["generation_tps"] / baseline
        )

    return results


def find_best_result(results: list[dict]) -> dict:
    return max(
        results,
        key=lambda item: item["generation_tps"],
    )


def save_results(results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    path = RESULTS_DIR / f"ngl-sweep-{timestamp}.json"

    path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    return path


def print_summary(results: list[dict]) -> None:
    print("\nNGL sweep summary")
    print("-" * 55)
    print(
        f"{'NGL':>5} "
        f"{'PP tok/s':>12} "
        f"{'TG tok/s':>12} "
        f"{'Speedup':>10}"
    )
    print("-" * 55)

    for result in results:
        print(
            f"{result['gpu_layers']:>5} "
            f"{result['prompt_tps']:>12.2f} "
            f"{result['generation_tps']:>12.2f} "
            f"{result['generation_speedup']:>9.2f}x"
        )


def print_best_result(result: dict) -> None:
    print("\nBest configuration")
    print("-" * 30)
    print(f"GPU layers : {result['gpu_layers']}")
    print(f"PP tok/s   : {result['prompt_tps']:.2f}")
    print(f"TG tok/s   : {result['generation_tps']:.2f}")
    print(f"Speedup    : {result['generation_speedup']:.2f}x")


def parse_layers(value: str) -> list[int]:
    try:
        layers = [
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layers must be comma-separated integers"
        ) from exc

    if not layers:
        raise argparse.ArgumentTypeError(
            "at least one layer value is required"
        )

    if any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError(
            "layer values cannot be negative"
        )

    return layers


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark llama.cpp GPU layer offload "
            "configurations."
        )
    )

    parser.add_argument(
        "--layers",
        type=parse_layers,
        help="comma-separated GPU layer counts",
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="automatically sweep GPU layers",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="GPU-layer step used with --auto",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="maximum GPU-layer value used with --auto",
    )

    args = parser.parse_args()

    if args.step <= 0:
        parser.error("--step must be greater than zero")

    if args.limit < 0:
        parser.error("--limit cannot be negative")

    if args.auto and args.layers:
        parser.error(
            "use either --auto or --layers, not both"
        )

    if args.auto:
        layers = generate_auto_layers(
            start=0,
            step=args.step,
            limit=args.limit,
        )
    elif args.layers:
        layers = args.layers
    else:
        layers = DEFAULT_LAYERS

    results = run_sweep(layers)

    print_summary(results)

    best = find_best_result(results)
    print_best_result(best)

    output_path = save_results(results)

    print(
        f"\nSaved results to {output_path}"
    )


if __name__ == "__main__":
    main()
