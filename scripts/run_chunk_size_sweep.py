"""Run a CUDA chunk-size sweep for the analytical torch lane."""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends import cpu_existing
from backends.torch_portable import (
    build_torch_dataset,
    export_perceptron_to_torch,
    get_torch_runtime_metadata,
    perceptron_predict_torch,
    synchronize_torch_device,
    torch,
)
from benchmarks.cases import parse_cases
from benchmarks.schema import validate_report_text


def _parse_int_list(text: str) -> list[int]:
    values = [int(token.strip()) for token in str(text).split(",") if token.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    for value in values:
        if value <= 0:
            raise ValueError("chunk sizes must be positive")
    return values


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[index]


def _format_env_lines(env: dict[str, object]) -> list[str]:
    return [
        f"- Python: `{env.get('python_version', 'unknown')}`",
        f"- Platform: `{env.get('platform', 'unknown')}`",
        f"- Torch: `{env.get('torch_version', 'unavailable')}`",
        f"- CUDA version: `{env.get('cuda_version') or 'none'}`",
        f"- Accelerator runtime: `{env.get('runtime_kind', 'unknown')}`",
        f"- Accelerator device: `{env.get('device_display_name', 'unknown')}`",
    ]


def build_report(
    *,
    case_id: str,
    chunk_sizes: list[int],
    rows: list[dict[str, object]],
    env: dict[str, object],
    warmup_runs: int,
    repeat_runs: int,
) -> str:
    lines = [
        "# RTX 4060 Ti Chunk-Size Sweep",
        "",
        "This experiment varies the `build_torch_dataset()` chunk size for the canonical `torch_accelerated` analytical lane.",
        "",
        "It is a structured performance experiment for the existing PyTorch CUDA path, not a hand-written CUDA kernel microbenchmark.",
        "",
        "## Environment",
        "",
    ]
    lines.extend(_format_env_lines(env))
    lines.extend(
        [
            "",
            "## Method",
            "",
            f"- Case: `{case_id}`",
            f"- Chunk sizes: `{', '.join(str(size) for size in chunk_sizes)}`",
            f"- Warmup / repeats: `{warmup_runs}` / `{repeat_runs}`",
            "- Timing scope matches `docs/benchmark_methodology.md`: accelerator-side synthetic dataset generation, analytical SNM / BER tensor computation, perceptron inference, and explicit synchronization before timer stop.",
            "- Compile overhead, PCIe transfer benchmarking, SPICE runtime, and custom CUDA kernel performance are not measured here.",
            "",
            "## Results",
            "",
            "| Chunk Size | Median Wall Clock (s) | Mean Wall Clock (s) | p95 Wall Clock (s) | Throughput (samples/s) | Speedup vs 1024 Chunk | Mean Prediction |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{int(row['chunk_size'])} | "
            f"{float(row['median_wall_clock_sec']):.6f} | "
            f"{float(row['mean_wall_clock_sec']):.6f} | "
            f"{float(row['p95_wall_clock_sec']):.6f} | "
            f"{float(row['throughput_samples_per_sec']):.3f} | "
            f"{float(row['speedup_vs_1024']):.2f}x | "
            f"{float(row['mean_prediction']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- The 1024 chunk is the current default in `build_torch_dataset()` and is used as the relative baseline.",
            "- Results are local RTX 4060 Ti evidence, not a universal GPU tuning rule.",
            "- Use this as a performance-engineering note beside the main full-suite CUDA validation report.",
        ]
    )
    text = "\n".join(lines) + "\n"
    validate_report_text(text)
    return text


def run_sweep(
    *,
    case_text: str,
    chunk_sizes: list[int],
    warmup_runs: int,
    repeat_runs: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object], str]:
    if torch is None:
        raise RuntimeError("PyTorch is unavailable")
    runtime = get_torch_runtime_metadata()
    if not runtime.accelerator_available:
        raise RuntimeError(f"accelerator unavailable: {runtime.reason}")

    cases = parse_cases(case_text)
    if len(cases) != 1:
        raise ValueError("chunk-size sweep expects exactly one case")
    case = cases[0]

    model = cpu_existing.fit_reference_perceptron(
        n_samples=4096,
        variability_samples=256,
        seed=int(seed),
        max_iter=4000,
    )
    exported = export_perceptron_to_torch(model, device="gpu")

    rows: list[dict[str, object]] = []
    for chunk_size in chunk_sizes:
        for warmup_index in range(max(int(warmup_runs), 0)):
            dataset = build_torch_dataset(
                n_samples=case.n_samples,
                variability_samples=case.variability_samples,
                seed=int(seed) + warmup_index,
                device=exported.device,
                chunk_size=int(chunk_size),
            )
            _ = perceptron_predict_torch(exported, dataset["X"])
            synchronize_torch_device(exported.device, exported.backend_kind)

        elapsed_values: list[float] = []
        mean_predictions: list[float] = []
        for repeat_index in range(max(int(repeat_runs), 1)):
            started = time.perf_counter()
            dataset = build_torch_dataset(
                n_samples=case.n_samples,
                variability_samples=case.variability_samples,
                seed=int(seed) + 100 + repeat_index,
                device=exported.device,
                chunk_size=int(chunk_size),
            )
            predictions = perceptron_predict_torch(exported, dataset["X"])
            synchronize_torch_device(exported.device, exported.backend_kind)
            elapsed_values.append(time.perf_counter() - started)
            mean_predictions.append(float(predictions.detach().mean().cpu().item()))

        median_wall_clock = statistics.median(elapsed_values)
        rows.append(
            {
                "case_id": case.case_id,
                "chunk_size": int(chunk_size),
                "median_wall_clock_sec": median_wall_clock,
                "mean_wall_clock_sec": statistics.mean(elapsed_values),
                "p95_wall_clock_sec": _p95(elapsed_values),
                "throughput_samples_per_sec": float(case.n_samples) / median_wall_clock if median_wall_clock > 0.0 else 0.0,
                "mean_prediction": statistics.mean(mean_predictions),
                "repeat_runs": int(repeat_runs),
                "warmup_runs": int(warmup_runs),
            }
        )

    baseline = next(
        (float(row["throughput_samples_per_sec"]) for row in rows if int(row["chunk_size"]) == 1024),
        float(rows[0]["throughput_samples_per_sec"]) if rows else 1.0,
    )
    for row in rows:
        row["speedup_vs_1024"] = (
            float(row["throughput_samples_per_sec"]) / baseline if baseline > 0.0 else 0.0
        )

    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": runtime.torch_version,
        "cuda_version": runtime.cuda_version,
        "runtime_kind": runtime.runtime_kind,
        "device_display_name": runtime.device_display_name,
    }
    return rows, env, case.case_id


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an RTX 4060 Ti chunk-size sweep")
    parser.add_argument("--case", default="20000x512")
    parser.add_argument("--chunk-sizes", default="256,512,1024,2048")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260310)
    parser.add_argument("--out-report", type=Path, default=REPO_ROOT / "reports" / "portability" / "chunk_size_sweep_4060ti.md")
    parser.add_argument("--out-csv", type=Path, default=REPO_ROOT / "reports" / "portability" / "chunk_size_sweep_4060ti.csv")
    args = parser.parse_args()

    chunk_sizes = _parse_int_list(str(args.chunk_sizes))
    rows, env, case_id = run_sweep(
        case_text=str(args.case),
        chunk_sizes=chunk_sizes,
        warmup_runs=int(args.warmup_runs),
        repeat_runs=int(args.repeat_runs),
        seed=int(args.seed),
    )
    report_text = build_report(
        case_id=case_id,
        chunk_sizes=chunk_sizes,
        rows=rows,
        env=env,
        warmup_runs=int(args.warmup_runs),
        repeat_runs=int(args.repeat_runs),
    )
    write_csv(Path(args.out_csv), rows)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(report_text, encoding="utf-8")
    print(f"[ok] wrote chunk-size sweep: {args.out_report}")
    print(f"[ok] wrote chunk-size sweep csv: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
