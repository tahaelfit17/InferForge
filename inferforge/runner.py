import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


LLAMA_BENCH = Path.home() / "Projects/llama.cpp/build-t1200/bin/llama-bench"
MODEL = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"

DEFAULT_LAYERS = [0, 1, 5, 10]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

SATURATION_THRESHOLD = 0.99


def run_benchmark(gpu_layers: int) -> list[dict]:
    command = [
        str(LLAMA_BENCH),
        "-hf",
        MODEL,
        "-ngl",
        str(gpu_layers),
        "-p",
        "512",
        "-n",
        "128",
        "-r",
        "3",
        "-o",
        "json",
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

            summary = summarize_run(
                gpu_layers,
                rows,
            )

            results.append(summary)

        except subprocess.CalledProcessError as exc:
            print(
                f"\nBenchmark failed at NGL={gpu_layers} "
                f"(exit code {exc.returncode})."
            )

            if exc.stderr:
                print(exc.stderr.strip())

            print(
                "Stopping sweep at the first failing configuration."
            )
            break

        except (json.JSONDecodeError, ValueError) as exc:
            print(
                f"\nInvalid benchmark output at NGL={gpu_layers}: "
                f"{exc}"
            )

            print(
                "Stopping sweep because the result could not be parsed."
            )
            break

    if not results:
        raise RuntimeError(
            "No benchmark completed successfully."
        )

    baseline = results[0]["generation_tps"]

    if baseline <= 0:
        raise RuntimeError(
            "Invalid baseline generation throughput."
        )

    for result in results:
        result["generation_speedup"] = (
            result["generation_tps"] / baseline
        )

    return results


def find_recommended_result(
    results: list[dict],
    threshold: float = SATURATION_THRESHOLD,
) -> dict:
    """
    Return the first configuration that reaches the requested
    fraction of the maximum observed generation throughput.

    Example:
        threshold=0.99 means the smallest tested NGL that
        achieves at least 99% of maximum observed TG throughput.
    """

    max_tps = max(
        result["generation_tps"]
        for result in results
    )

    target_tps = max_tps * threshold

    for result in results:
        if result["generation_tps"] >= target_tps:
            recommended = result.copy()

            recommended["max_generation_tps"] = max_tps

            recommended["max_performance_percent"] = (
                result["generation_tps"] / max_tps
            ) * 100

            recommended["saturation_threshold_percent"] = (
                threshold * 100
            )

            return recommended

    raise RuntimeError(
        "Could not determine a recommended configuration."
    )


def save_results(
    results: list[dict],
    recommended: dict,
) -> Path:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    path = RESULTS_DIR / f"ngl-sweep-{timestamp}.json"

    payload = {
        "timestamp_utc": timestamp,
        "model": MODEL,
        "saturation_threshold": SATURATION_THRESHOLD,
        "recommended": recommended,
        "results": results,
    }

    path.write_text(
        json.dumps(payload, indent=2) + "\n",
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


def print_recommended_result(result: dict) -> None:
    print("\nRecommended configuration")
    print("-" * 36)

    print(
        f"GPU layers       : "
        f"{result['gpu_layers']}"
    )

    print(
        f"PP tok/s         : "
        f"{result['prompt_tps']:.2f}"
    )

    print(
        f"TG tok/s         : "
        f"{result['generation_tps']:.2f}"
    )

    print(
        f"Maximum observed : "
        f"{result['max_generation_tps']:.2f}"
    )

    print(
        f"Max performance  : "
        f"{result['max_performance_percent']:.2f}%"
    )

    print(
        f"Threshold        : "
        f"{result['saturation_threshold_percent']:.2f}%"
    )

    print(
        f"Speedup          : "
        f"{result['generation_speedup']:.2f}x"
    )


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
            "Benchmark and tune llama.cpp "
            "GPU-layer offload configurations."
        )
    )

    parser.add_argument(
        "--layers",
        type=parse_layers,
        help=(
            "comma-separated GPU layer counts "
            "(example: 0,5,10,15)"
        ),
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="automatically generate GPU-layer configurations",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="GPU-layer step used with --auto (default: 5)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="maximum GPU-layer value used with --auto (default: 40)",
    )

    args = parser.parse_args()

    if args.step <= 0:
        parser.error(
            "--step must be greater than zero"
        )

    if args.limit < 0:
        parser.error(
            "--limit cannot be negative"
        )

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

    recommended = find_recommended_result(
        results
    )

    print_recommended_result(
        recommended
    )

    output_path = save_results(
        results,
        recommended,
    )

    print(
        f"\nSaved results to {output_path}"
    )


if __name__ == "__main__":
    main()
