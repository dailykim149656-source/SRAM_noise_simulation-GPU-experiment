# Portability Snapshot Reports

This directory contains representative sanitized snapshots from the portability benchmark workflow.

Use these files as public evidence snapshots, not as a complete benchmark archive. The smoke reports emphasize correctness, artifact shape, and fidelity. The dashboard includes measured throughput rows, but those rows are tied to the recorded environment and case sizes.

Included:

- `cpu_smoke_report.md`
- `cpu_smoke_fidelity.md`
- `cuda_smoke_report.md`
- `cuda_smoke_fidelity.md`
- `dashboard.md`
- `changelog.md`
- `prd_verify.json`

These are representative benchmark artifacts captured from the standardized portability pipeline. They are not a complete experimental archive.

Important reading notes:

- historical snapshots may use the legacy `gpu_pytorch` lane name; current code normalizes it to `torch_accelerated`
- smoke rows are small-case validation checks and may show accelerator overhead
- throughput rows should be read alongside `metadata.json` fields such as validation scope, claim level, runtime, and case size
- ROCm/HIP validation is not claimed by these snapshots
