# Reproduce The RTX 4060 Ti CUDA Snapshot

This note documents how the checked-in CUDA full-suite snapshot was verified and how to refresh it without overstating portability claims.

## Captured Snapshot

- Report: `reports/portability/cuda_full_report.md`
- Fidelity report: `reports/portability/cuda_full_fidelity.md`
- Artifact source: `artifacts/benchmarks/20260606T075234Z_full`
- Host OS in metadata: Windows
- Python in metadata: `3.11.9`
- Accelerator: `NVIDIA GeForce RTX 4060 Ti`
- CUDA runtime reported by PyTorch: `12.6`
- Torch reported by the local CUDA benchmark environment: `2.12.0+cu126`

The Torch version was not hand-written into the public report. It was captured from benchmark metadata and rechecked locally with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
python -m pip show torch
```

Observed values for the checked-in full snapshot:

```text
2.12.0+cu126
12.6
NVIDIA GeForce RTX 4060 Ti
```

`pip show torch` reported `Version: 2.12.0+cu126`. The local wheel metadata did not include a `direct_url.json`, so this repository treats the value as captured environment metadata rather than a portable dependency pin.

## Environment Setup

Install the base benchmark dependencies, then install a CUDA-capable PyTorch wheel that matches the target host and driver/runtime. Use the PyTorch install selector or the package index appropriate for the target CUDA runtime.

```powershell
python -m venv .venv-cuda-benchmark
.\.venv-cuda-benchmark\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-base.txt -r requirements-benchmark.txt

# Choose the correct PyTorch wheel for the local CUDA runtime.
# Example shape only; verify the current selector output before using it.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Before trusting any benchmark result, record the runtime identity:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Run And Validate

Run the smoke suite first to verify artifact generation and CPU-vs-accelerator fidelity on a small case:

```powershell
python -m benchmarks.cli --suite smoke --device auto
python -m benchmarks.validate --artifact-dir artifacts\benchmarks\<timestamp>_smoke
```

Then run the full suite:

```powershell
python -m benchmarks.cli --suite full --device auto
python -m benchmarks.validate --artifact-dir artifacts\benchmarks\<timestamp>_full
```

The full snapshot is useful only when:

- `validation_scope` is `cuda_validated`
- `claim_level` is `measured`
- `torch_accelerated` rows have `status` set to `pass`
- `cpu_existing_vs_torch_accelerated` fidelity passes
- the report environment records the Torch version, build tag, CUDA runtime, and accelerator device

## Refresh Public Snapshots

After a validated run, promote only sanitized public evidence into `reports/portability/`:

```powershell
Copy-Item artifacts\benchmarks\<timestamp>_full\report.md reports\portability\cuda_full_report.md
Copy-Item artifacts\benchmarks\<timestamp>_full\fidelity.md reports\portability\cuda_full_fidelity.md
python scripts\build_portability_dashboard.py --artifact 20260417T070914Z_smoke --artifact 20260417T070915Z_smoke --artifact 20260417T105911Z_full --artifact <timestamp>_full --out-report reports\portability\dashboard.md
```

Do not describe this repository as a hand-tuned CUDA kernel implementation. The supported external claim is narrower: a reproducible CPU/CUDA benchmarking and fidelity-validation framework for an SRAM surrogate workload.
