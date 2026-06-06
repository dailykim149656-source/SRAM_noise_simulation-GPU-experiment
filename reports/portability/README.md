# Portability Snapshot Reports

This directory contains representative sanitized snapshots from the portability benchmark workflow.

Use these files as public evidence snapshots, not as a complete benchmark archive. The smoke reports emphasize correctness, artifact shape, and fidelity. The dashboard includes measured throughput rows, but those rows are tied to the recorded environment and case sizes.

Included:

- `cpu_smoke_report.md`
- `cpu_smoke_fidelity.md`
- `cuda_smoke_report.md`
- `cuda_smoke_fidelity.md`
- `cuda_full_report.md`
- `cuda_full_fidelity.md`
- `dashboard.md`
- `changelog.md`
- `prd_verify.json`

These are representative benchmark artifacts captured from the standardized portability pipeline. They are not a complete experimental archive.

The RTX 4060 Ti full-suite snapshot is paired with `docs/reproduce_cuda_4060ti.md`, which records the local Torch/CUDA verification commands used to defend the `2.12.0+cu126` environment metadata.

Important reading notes:

- historical snapshots may use the legacy `gpu_pytorch` lane name; current code normalizes it to `torch_accelerated`
- smoke rows are small-case validation checks and may show accelerator overhead
- throughput and speedup rows should be read alongside `metadata.json` fields such as validation scope, claim level, runtime, and case size
- ROCm/HIP validation is not claimed by these snapshots
