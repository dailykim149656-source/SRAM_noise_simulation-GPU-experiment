# SRAM Analytical Benchmark Report

- Suite: `full`
- Device mode: `auto`
- Seed: `20260310`
- Warmup / repeats: `1` / `3`
- Validation scope: `cuda_validated`
- Claim level: `measured`
- Selected artifact files: `metadata.json, results.csv, report.md, fidelity.md`

## Environment

- Python: `3.11.9`
- Platform: `Windows-10-10.0.26200-SP0`
- Torch: `2.12.0+cu126`
- Torch build tag: `cu126`
- Accelerator available: `True`
- Accelerator runtime: `cuda`
- Accelerator device: `NVIDIA GeForce RTX 4060 Ti`
- CUDA version: `12.6`
- HIP version: `none`

## Results

| Case | Lane | Status | Engine | Backend | Runtime | Device | Median Wall Clock (s) | Throughput (samples/s) | Speedup vs CPU Existing | Mean Prediction |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| 10000x512 | cpu_existing | pass | cpu | cpu | cpu | cpu | 1.004453 | 9955.670 | 1.00x | 0.060547 |
| 10000x512 | cpu_numpy | pass | cpu | cpu | cpu | cpu | 1.056001 | 9469.692 | 0.95x | 0.060547 |
| 10000x512 | torch_accelerated | pass | gpu | cuda | cuda | NVIDIA GeForce RTX 4060 Ti | 0.053900 | 185527.036 | 18.64x | 0.060239 |
| 5000x1024 | cpu_existing | pass | cpu | cpu | cpu | cpu | 1.033600 | 4837.459 | 1.00x | 0.059223 |
| 5000x1024 | cpu_numpy | pass | cpu | cpu | cpu | cpu | 1.108249 | 4511.620 | 0.93x | 0.059223 |
| 5000x1024 | torch_accelerated | pass | gpu | cuda | cuda | NVIDIA GeForce RTX 4060 Ti | 0.007476 | 668788.957 | 138.25x | 0.062002 |
| 20000x512 | cpu_existing | pass | cpu | cpu | cpu | cpu | 2.160713 | 9256.202 | 1.00x | 0.060831 |
| 20000x512 | cpu_numpy | pass | cpu | cpu | cpu | cpu | 2.134848 | 9368.350 | 1.01x | 0.060831 |
| 20000x512 | torch_accelerated | pass | gpu | cuda | cuda | NVIDIA GeForce RTX 4060 Ti | 0.016212 | 1233646.474 | 133.28x | 0.060532 |

## Fidelity Summary

| Pair | Status | Max Abs Delta | Mean Abs Delta | Threshold Max | Threshold Mean |
|---|---|---:|---:|---:|---:|
| cpu_existing_vs_cpu_numpy | pass | 0.000000e+00 | 0.000000e+00 | 1.000000e-06 | 1.000000e-07 |
| cpu_existing_vs_torch_accelerated | pass | 2.958160e-08 | 7.233361e-09 | 1.000000e-03 | 1.000000e-04 |

## Notes

- `cpu_existing` uses `AnalyticalSRAMModel.generate_dataset()` with the fitted perceptron `.predict()` path.
- `cpu_numpy` uses chunked analytical generation plus explicit NumPy forward.
- `torch_accelerated` is the canonical accelerator lane and is currently CUDA-validated when a compatible PyTorch build is installed.
- The `2.12.0+cu126` Torch value was rechecked in the local CUDA benchmark environment with `torch.__version__`, `torch.version.cuda`, `torch.cuda.get_device_name(0)`, and `pip show torch`; see `docs/reproduce_cuda_4060ti.md`.
- ROCm validation remains pending real AMD hardware access; a skipped accelerator row is not evidence of ROCm support.
