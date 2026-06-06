# Portfolio Review Guide

Use this path for a 5-minute technical review of the NVIDIA-facing GPU validation work.

## Read In This Order

1. `README.md`
2. `reports/portability/cuda_full_report.md`
3. `reports/portability/cuda_full_fidelity.md`
4. `reports/portability/cuda_full_environment.txt`
5. `reports/portability/cuda_full_metadata.json`
6. `reports/portability/cuda_full_results.csv`
7. `docs/benchmark_methodology.md`
8. `backends/accelerator_lane.py`
9. `backends/torch_portable.py`
10. `benchmarks/runner.py`

## What This Demonstrates

- Semiconductor-domain workload framing around an SRAM analytical surrogate
- CPU, NumPy, and CUDA-backed PyTorch lane separation
- Standardized benchmark artifact design
- CPU-vs-accelerator fidelity validation
- Runtime/environment metadata capture
- Conservative claim boundaries for public portfolio use

## What This Does Not Demonstrate

- Custom CUDA kernel optimization
- SPICE signoff validation
- Silicon correlation
- ROCm/HIP validation on AMD hardware
- Universal GPU speedup across all SRAM workloads

## Fast Local Verification

CPU-only smoke:

```powershell
python -m unittest discover -s tests -p "test_*.py"
$env:SRAM_FORCE_CPU="1"
python -m benchmarks.cli --suite smoke
Remove-Item Env:SRAM_FORCE_CPU
```

Validate a generated artifact:

```powershell
python -m benchmarks.validate --artifact-dir artifacts\benchmarks\<timestamp>_smoke
```

## Refresh CUDA Evidence

Use `docs/reproduce_cuda_4060ti.md` for the full RTX 4060 Ti reproduction flow. The checked-in CUDA evidence is a measured local snapshot with explicit runtime metadata, not a portable dependency pin and not a hand-written CUDA kernel benchmark.
