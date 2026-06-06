# RTX 4060 Ti torch.compile Preflight

This preflight checked whether a `torch.compile` comparison should be published beside the CUDA full-suite and chunk-size sweep evidence.

## Environment

- Python: `3.11.9`
- Platform: `Windows-10-10.0.26200-SP0`
- Torch: `2.12.0+cu126`
- CUDA version: `12.6`
- Accelerator device: `NVIDIA GeForce RTX 4060 Ti`

## Probe

A small CUDA tensor function was wrapped with `torch.compile` and executed after moving inputs to CUDA.

Observed facts:

- `hasattr(torch, "compile")`: `True`
- `torch.cuda.is_available()`: `True`
- compile execution failed before a usable timing comparison was produced

Failure summary:

```text
torch._inductor.exc.TritonMissing: Cannot find a working triton installation.
```

## Decision

Do not publish a `torch.compile` speedup table from this environment yet.

The checked-in chunk-size sweep remains the structured performance experiment for this portfolio pass. A future `torch.compile` comparison should first install and document a compatible Triton stack, then report compile warmup handling separately from steady-state timing.

## Claim Boundary

This preflight is an environment-readiness note. It is not evidence that `torch.compile` is slow or unsuitable for the workload, and it is not a hand-written CUDA kernel benchmark.
