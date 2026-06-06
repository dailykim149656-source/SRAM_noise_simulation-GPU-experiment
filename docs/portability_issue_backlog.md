# Portability Issue Backlog

This backlog breaks the portability PRD into trackable follow-up work after the initial benchmark refactor.

## Completed

1. Added representative CPU benchmark snapshots under `reports/portability/`.
2. Added representative CUDA benchmark snapshots under `reports/portability/`.
3. Added machine-readable CUDA full-suite metadata and results artifacts:
   - `reports/portability/cuda_full_metadata.json`
   - `reports/portability/cuda_full_results.csv`
4. Added CUDA environment provenance notes:
   - `reports/portability/cuda_full_environment.txt`
   - `docs/reproduce_cuda_4060ti.md`
5. Added public claim and README-number guard tests in `tests/test_portfolio_public_evidence.py`.
6. Added `PORTFOLIO_REVIEW.md` as the 5-minute NVIDIA-facing review path.

## Open Follow-ups

1. Add an Nsight Systems profiling summary for the RTX 4060 Ti run.
2. Add one structured performance experiment:
   - `torch.compile` comparison, or
   - chunk-size sweep on RTX 4060 Ti.
3. Refactor `native_backend.py` GPU torch fallbacks onto the new backend abstraction surface.
4. Add ROCm-capability detection to the backend registry once AMD hardware is available.
5. Add a manual ROCm validation checklist only after real AMD hardware access.
6. Revisit whether CuPy detection should remain in `execution_policy.py`.
