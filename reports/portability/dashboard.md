# Portability Benchmark Dashboard

This dashboard summarizes representative portability benchmark artifacts produced by the standardized analytical benchmark pipeline.

Interpret smoke rows as correctness/fidelity checks. Treat measured throughput rows as environment-specific snapshots, not universal speedup claims.

| Artifact | Suite | Cases | Device Mode | Validation Scope | Claim | Accelerator Runtime | CPU Existing Throughput | CPU NumPy Throughput | Torch Accelerated Throughput |
|---|---|---|---|---|---|---|---|---|---|
| 20260417T070914Z_smoke | smoke | 1024x64 | cpu | legacy_snapshot | legacy | cuda | 50387.006 | 50807.008 | 0.000 |
| 20260417T070915Z_smoke | smoke | 1024x64 | auto | legacy_snapshot | legacy | cuda | 40478.948 | 49986.332 | 2109.712 |
| 20260417T105911Z_full | full | 1024x64 | auto | cuda_validated | measured | cuda | 75128.393 | 70129.781 | 465772.117 |
| 20260606T075234Z_full | full | 10000x512, 5000x1024, 20000x512 | auto | cuda_validated | measured | cuda | 4837.459-9955.670 | 4511.620-9469.692 | 185527.036-1233646.474 |
