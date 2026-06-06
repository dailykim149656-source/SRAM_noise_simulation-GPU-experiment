# SRAM GPU Portability And Benchmarking

[![CPU Smoke](https://github.com/dailykim149656-source/SRAM_noise_simulation-GPU-experiment/actions/workflows/cpu-smoke.yml/badge.svg)](https://github.com/dailykim149656-source/SRAM_noise_simulation-GPU-experiment/actions/workflows/cpu-smoke.yml)

This repository is an SRAM surrogate and simulation codebase with a portability-focused analytical benchmark path.

It is positioned as a reproducible GPU validation and benchmarking portfolio asset, not as a hand-tuned CUDA kernel library. The core value is the separation of CPU, NumPy, and accelerator lanes with benchmark artifacts, environment metadata, and CPU-vs-accelerator fidelity checks.

## Portfolio Snapshot

| Area | Current Evidence |
|---|---|
| Domain workload | SRAM analytical surrogate and simulation paths |
| Execution lanes | `cpu_existing`, `cpu_numpy`, canonical `torch_accelerated` |
| CUDA validation | Checked-in RTX 4060 Ti snapshots under `reports/portability/` |
| Fidelity validation | CPU reference vs accelerator checks with max/mean absolute-delta thresholds |
| Portability boundary | CUDA-measured evidence is kept separate from ROCm/HIP future work |

## Representative Results

Representative checked-in evidence:

| Evidence | Environment | Workload | Fidelity | Performance Signal |
|---|---|---|---|---|
| `reports/portability/cuda_smoke_report.md` | Windows, Python 3.11.9, Torch `2.6.0+cu124`, RTX 4060 Ti | `1024x64` smoke | CPU-vs-accelerator passed | correctness-oriented smoke snapshot |
| `reports/portability/cuda_full_report.md` | Windows, Python 3.11.9, Torch `2.12.0+cu126`, CUDA `12.6`, RTX 4060 Ti | `10000x512`, `5000x1024`, `20000x512` full suite | max abs delta `2.958160e-08` | `185k-1.23M` samples/s, `18.64x-138.25x` vs `cpu_existing` across measured cases |
| `reports/portability/dashboard.md` | RTX 4060 Ti snapshot series | smoke plus measured full rows | validation scope and claim level shown per artifact | throughput and speedup ranges summarized beside case size |

Read the benchmark numbers with two separate questions in mind:

- smoke rows prove artifact generation and numerical fidelity on a small case
- measured throughput rows are environment-specific performance snapshots, not a universal speedup claim
- the `2.12.0+cu126` Torch value in the full snapshot was rechecked from the local CUDA benchmark environment via `torch.__version__`, `torch.version.cuda`, `torch.cuda.get_device_name(0)`, and `pip show torch`; see `docs/reproduce_cuda_4060ti.md`

Today it provides:

- reproducible CPU benchmark artifacts
- a canonical `torch_accelerated` lane that is currently CUDA-validated when a compatible PyTorch build is available
- fidelity checks between CPU inference paths and the canonical accelerator lane
- isolation of accelerator-specific logic to reduce future ROCm/HIP porting cost

## Why This Matters For Semiconductor Simulation

Semiconductor simulation and SRAM reliability work often mix domain assumptions, generated collateral, CPU reference paths, and accelerator experiments. This repository keeps those claims separated: CPU baselines remain reproducible, accelerator paths emit comparable artifacts, and portability plans stay distinct from measured CUDA evidence.

## What Is Validated

- CPU analytical benchmark smoke runs through `python -m benchmarks.cli --suite smoke --device cpu`
- the compatibility wrapper `python scripts/run_gpu_analytical_benchmark.py` still works
- analytical benchmark runs emit standard artifacts:
  - `metadata.json`
  - `results.csv`
  - `report.md`
  - `fidelity.md`
- fresh artifacts record `validation_scope`, `claim_level`, and accelerator backend/runtime metadata
- CPU existing vs CPU NumPy inference parity is checked automatically
- accelerator lanes degrade to `skipped` or `unsupported` instead of crashing when an accelerator runtime is unavailable

## What Is Not Claimed

- No AMD GPU or ROCm benchmark result is included
- No HIP port is implemented in this batch
- ROCm validation is pending AMD hardware access
- `native_backend.py` simulate/lifetime/optimize flows are not fully migrated into the new backend package yet
- The checked-in CUDA snapshots do not claim universal GPU speedup across all workload sizes

Use conservative wording:

> CPU benchmark artifacts are reproducible today, the `torch_accelerated` lane is currently CUDA-validated when a compatible PyTorch build is installed, and ROCm validation remains pending AMD hardware access.

## Linux-First Quickstart

CPU-only benchmark setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt -r requirements-benchmark.txt
python -m benchmarks.cli --suite smoke --device cpu
```

Optional CUDA benchmark setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt -r requirements-benchmark.txt
# install the correct PyTorch build for your CUDA/runtime combination
python -m benchmarks.cli --suite smoke --device auto
```

## Quick Start

Create an environment and install the benchmark stack:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-base.txt -r requirements-benchmark.txt
```

If you want the older umbrella install, `requirements.txt` still installs the base, benchmark, and UI dependency sets together.

### CPU-only benchmark smoke

```powershell
python -m benchmarks.cli --suite smoke --device cpu
```

CPU-only emulation with auto selection:

```powershell
$env:SRAM_FORCE_CPU='1'
python -m benchmarks.cli --suite smoke
Remove-Item Env:SRAM_FORCE_CPU
```

### Optional CUDA benchmark smoke

Install a CUDA-capable PyTorch build for your platform first, then run:

```powershell
python -m benchmarks.cli --suite smoke --device auto
```

The canonical accelerator lane recorded in fresh artifacts is `torch_accelerated`. Historical artifacts may still show the legacy alias `gpu_pytorch`.

### Compatibility wrapper

```powershell
python scripts/run_gpu_analytical_benchmark.py
```

Alternative module entrypoint:

```powershell
python -m benchmarks.run_suite --suite smoke --device auto
```

This keeps the legacy CLI flags while also writing a standard artifact directory under `artifacts/benchmarks/`.

## Dependency Layout

- `requirements-base.txt`
  - NumPy and SciPy
- `requirements-benchmark.txt`
  - scikit-learn
  - PyTorch is documented as an optional manual install because the correct package depends on platform and accelerator runtime
- `requirements-ui.txt`
  - Matplotlib, Streamlit, PySide6
- `requirements-dev.txt`
  - base + benchmark development/test stack

## Benchmark Architecture

- `backends/`
  - `cpu_existing.py`
  - `cpu_numpy.py`
  - `torch_portable.py`
  - `accelerator_lane.py`
  - `cuda_lane.py`
  - `registry.py`
- `benchmarks/`
  - suite cases, environment capture, metrics, report writers, CLI
- `gpu_analytical_adapter.py`
  - compatibility facade for earlier analytical helper imports

The main simulation and UI entry points remain in place:

- `main.py`
- `main_advanced.py`
- `native_backend.py`
- `streamlit_app*.py`
- `pyside_sram_app_advanced.py`

## Standard Benchmark Artifacts

Each run writes a timestamped directory under `artifacts/benchmarks/` containing:

- `metadata.json`
- `results.csv`
- `report.md`
- `fidelity.md`

New portability benchmark artifacts avoid absolute local filesystem paths.
Fresh artifacts use the canonical lane name `torch_accelerated`, while readers and dashboards still normalize the legacy `gpu_pytorch` alias from older snapshots.

## Representative Portability Snapshots

Checked-in sanitized snapshots are available under `reports/portability/`:

- `reports/portability/cpu_smoke_report.md`
- `reports/portability/cpu_smoke_fidelity.md`
- `reports/portability/cuda_smoke_report.md`
- `reports/portability/cuda_smoke_fidelity.md`
- `reports/portability/cuda_full_report.md`
- `reports/portability/cuda_full_fidelity.md`
- `reports/portability/dashboard.md`

Some generated benchmark artifacts may also include optional plots under `plots/`.

Minimal packaging metadata and console-script entrypoints are also defined in `pyproject.toml`.

Release-oriented portability automation is defined in `.github/workflows/portability-release.yml`.

## Other Entry Points

Core simulation:

```powershell
python main.py
python main_advanced.py
python hybrid_perceptron_sram.py
python adaptive_perceptron_sram.py
python reliability_model.py
python workload_model.py
```

UI:

```powershell
pip install -r requirements-ui.txt
streamlit run streamlit_app.py
streamlit run streamlit_app_advanced.py
streamlit run streamlit_app_unified.py
python pyside_sram_app_advanced.py
```

Validation and report generation:

```powershell
python spice_validation/run_spice_validation.py --spice-source placeholder
python scripts/run_pdk_matrix.py
python scripts/run_model_selection.py
python scripts/run_node_scaling.py
python scripts/build_research_evidence_pack.py
python scripts/export_research_bundle.py --tag public_snapshot --skip-zip
```

## Key Docs

- `docs/benchmark_baseline_inventory.md`
- `docs/benchmark_methodology.md`
- `docs/backend_portability.md`
- `docs/hip_porting_plan.md`
- `docs/rocm_validation_matrix.md`
- `docs/instinct_target_profile.md`
- `docs/hipify_preflight_inventory.md`
- `docs/rocm_manual_checklist.md`
- `docs/limitations_and_claims.md`
- `docs/reproduce_cuda_4060ti.md`
- `docs/results_interpretation_guide.md`
- `docs/portability_issue_backlog.md`
- `docs/portability_release_checklist.md`
- `docs/prd_completion_matrix.md`
- `docs/native_backend_portability_inventory.md`
- `docs/native_backend_rocm_migration_plan.md`
- `docs/ci_future_rocm_runner_note.md`
- `docs/pdk_validation_criteria.md`
- `docs/open_source_reliability_roadmap_2026-03-09.md`
- `docker/README.md`
- `reports/portability/changelog.md`
