"""Build a markdown dashboard from portability benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.schema import normalize_lane_name, validate_report_text


def _load_rows(results_path: Path) -> list[dict[str, str]]:
    with results_path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _latest_artifacts(artifact_root: Path, limit: int) -> list[Path]:
    candidates = [path for path in artifact_root.iterdir() if path.is_dir()]
    return sorted(candidates)[-max(int(limit), 1) :]


def _case_summary(metadata: dict[str, object]) -> str:
    cases = metadata.get("cases", [])
    if not isinstance(cases, list) or not cases:
        return "unknown"
    case_ids = []
    for case in cases:
        if isinstance(case, dict) and case.get("case_id"):
            case_ids.append(str(case["case_id"]))
    return ", ".join(case_ids) if case_ids else "unknown"


def _accelerator_runtime(metadata: dict[str, object]) -> str:
    env = metadata.get("env", {})
    if not isinstance(env, dict):
        return "unknown"
    runtime = env.get("accelerator_runtime_kind")
    backend = env.get("accelerator_backend_kind")
    if runtime:
        return str(runtime)
    if backend:
        return str(backend)
    if env.get("cuda_available"):
        return "cuda"
    return "unknown"


def _throughput_summary(rows: list[dict[str, str]], lane_name: str) -> str:
    values = [
        float(row["throughput_samples_per_sec"])
        for row in rows
        if normalize_lane_name(row["lane"]) == lane_name and row.get("status") == "pass"
    ]
    if not values:
        return "0.000"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{min(values):.3f}-{max(values):.3f}"


def _torch_speedup_summary(rows: list[dict[str, str]]) -> str:
    cpu_existing_by_case = {
        row["case_id"]: float(row["throughput_samples_per_sec"])
        for row in rows
        if normalize_lane_name(row["lane"]) == "cpu_existing"
        and row.get("status") == "pass"
        and float(row["throughput_samples_per_sec"]) > 0.0
    }
    speedups = []
    for row in rows:
        if normalize_lane_name(row["lane"]) != "torch_accelerated" or row.get("status") != "pass":
            continue
        baseline = cpu_existing_by_case.get(row["case_id"], 0.0)
        throughput = float(row["throughput_samples_per_sec"])
        if baseline > 0.0 and throughput > 0.0:
            speedups.append(throughput / baseline)
    if not speedups:
        return "n/a"
    if len(speedups) == 1:
        return f"{speedups[0]:.2f}x"
    return f"{min(speedups):.2f}x-{max(speedups):.2f}x"


def build_dashboard(artifact_dirs: list[Path]) -> str:
    lines = [
        "# Portability Benchmark Dashboard",
        "",
        "This dashboard summarizes representative portability benchmark artifacts produced by the standardized analytical benchmark pipeline.",
        "",
        "Interpret smoke rows as correctness/fidelity checks. Treat measured throughput rows as environment-specific snapshots, not universal speedup claims.",
        "",
        "| Artifact | Suite | Cases | Device Mode | Validation Scope | Claim | Accelerator Runtime | CPU Existing Throughput | CPU NumPy Throughput | Torch Accelerated Throughput | Torch Speedup vs CPU Existing |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for artifact_dir in artifact_dirs:
        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        rows = _load_rows(artifact_dir / "results.csv")
        lines.append(
            "| "
            f"{artifact_dir.name} | {metadata['suite']} | {_case_summary(metadata)} | {metadata['device_mode']} | "
            f"{metadata.get('validation_scope', 'legacy_snapshot')} | "
            f"{metadata.get('claim_level', 'legacy')} | "
            f"{_accelerator_runtime(metadata)} | "
            f"{_throughput_summary(rows, 'cpu_existing')} | "
            f"{_throughput_summary(rows, 'cpu_numpy')} | "
            f"{_throughput_summary(rows, 'torch_accelerated')} | "
            f"{_torch_speedup_summary(rows)} |"
        )
    text = "\n".join(lines) + "\n"
    validate_report_text(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Build portability benchmark dashboard")
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "artifacts" / "benchmarks")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="artifact directory name or path to include; can be repeated",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out-report", type=Path, default=REPO_ROOT / "reports" / "portability" / "dashboard.md")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    if args.artifact:
        artifact_dirs = []
        for artifact in args.artifact:
            artifact_path = Path(artifact)
            if not artifact_path.is_absolute():
                artifact_path = artifact_root / artifact_path
            artifact_dirs.append(artifact_path)
    else:
        artifact_dirs = _latest_artifacts(artifact_root, args.limit)
    if not artifact_dirs:
        raise FileNotFoundError(f"no benchmark artifacts found under {args.artifact_root}")

    report_text = build_dashboard(artifact_dirs)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(report_text, encoding="utf-8")
    print(f"[ok] wrote dashboard: {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
