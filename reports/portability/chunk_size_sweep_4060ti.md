# RTX 4060 Ti Chunk-Size Sweep

This experiment varies the `build_torch_dataset()` chunk size for the canonical `torch_accelerated` analytical lane.

It is a structured performance experiment for the existing PyTorch CUDA path, not a hand-written CUDA kernel microbenchmark.

## Environment

- Python: `3.11.9`
- Platform: `Windows-10-10.0.26200-SP0`
- Torch: `2.12.0+cu126`
- CUDA version: `12.6`
- Accelerator runtime: `cuda`
- Accelerator device: `NVIDIA GeForce RTX 4060 Ti`

## Method

- Case: `20000x512`
- Chunk sizes: `256, 512, 1024, 2048`
- Warmup / repeats: `1` / `3`
- Timing scope matches `docs/benchmark_methodology.md`: accelerator-side synthetic dataset generation, analytical SNM / BER tensor computation, perceptron inference, and explicit synchronization before timer stop.
- Compile overhead, PCIe transfer benchmarking, SPICE runtime, and custom CUDA kernel performance are not measured here.

## Results

| Chunk Size | Median Wall Clock (s) | Mean Wall Clock (s) | p95 Wall Clock (s) | Throughput (samples/s) | Speedup vs 1024 Chunk | Mean Prediction |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.124000 | 0.119143 | 0.129232 | 161290.583 | 0.25x | 0.059903 |
| 512 | 0.059026 | 0.060706 | 0.064253 | 338833.160 | 0.52x | 0.059903 |
| 1024 | 0.030715 | 0.032102 | 0.035120 | 651156.128 | 1.00x | 0.059903 |
| 2048 | 0.022613 | 0.021596 | 0.022725 | 884435.266 | 1.36x | 0.059903 |

## Reading Notes

- The 1024 chunk is the current default in `build_torch_dataset()` and is used as the relative baseline.
- Results are local RTX 4060 Ti evidence, not a universal GPU tuning rule.
- Use this as a performance-engineering note beside the main full-suite CUDA validation report.
