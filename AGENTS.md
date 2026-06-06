# Project Instructions

## Repository Shape

- This repository is an SRAM surrogate simulation and GPU portability benchmark project.
- Keep benchmark lanes separated: CPU reference paths live under `backends/cpu_*.py`, accelerator-neutral PyTorch logic lives under `backends/torch_portable.py` and `backends/accelerator_lane.py`, and native/runtime orchestration lives in `native_backend.py`.
- Treat PyTorch CUDA as the validated NVIDIA path and ROCm/HIP as prepared but unmeasured unless a real AMD run has been performed.

## Verification

- Default unit test command: `python -m unittest discover -s tests -p "test_*.py"`.
- Python source compile check: `python -m compileall backends benchmarks tests gpu_analytical_adapter.py execution_policy.py reliability_model.py lifetime_service.py scripts/run_gpu_analytical_benchmark.py`.
- CPU benchmark smoke: `python -m benchmarks.cli --suite smoke --device cpu`.
- Optional accelerator smoke: `python -m benchmarks.cli --suite smoke --device auto`.
- Validate fresh benchmark artifacts with `python -m benchmarks.validate --artifact-dir <artifact-dir>`.

## Artifact And Claim Boundaries

- Generated benchmark artifacts normally belong under `artifacts/benchmarks/` or task-local `.omx/` scratch directories.
- Do not promote new CUDA/accelerator results into `reports/portability/` unless the run has passed artifact validation and the public claim wording remains conservative.
- Do not describe this project as a hand-written CUDA kernel library, a completed HIP port, or silicon/PDK signoff validation.
