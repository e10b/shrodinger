# H-AMR Closeness Map

This project is not the H-AMR codebase, but the HARM engine now exposes the same major solver surfaces reviewers expect to see in an H-AMR-adjacent GRMHD project.

## Current Readiness

| Area | Status |
|---|---|
| Horizon-penetrating Kerr-Schild coordinates | Present |
| Fishbone-Moncrief torus setup | Present |
| Conservative primitive/conserved state path | Present |
| HLL Riemann fluxes | Present |
| MC-limited reconstruction | Present through `--high-order` |
| CFL-limited stepping | Present in the CPU reference path |
| Primitive recovery with fallback | Present |
| Floors and polar/radial boundary policy | Present |
| Magnetic divergence diagnostics/control | Present |
| MRI quality diagnostics | Present |
| Porth-style benchmark report | Present |
| AMR-style refinement candidate scoring | Present |
| AMR block ghost/restrict/prolong helpers | Present |
| Adaptive block-local subcycling | Present |
| State norms for convergence/parity tracking | Present |
| GPU/CPU upload parity | Present, exact zero-frame gate |
| GPU/CPU evolution parity | Covariant GPU path implemented; executable parity awaiting a WebGPU-capable validation host |
| Scientific replacement score | Present, target 9/10 |
| Method validation suite | Present |

## How To Generate The Score

```bash
cmake --build build --target HarmBench
./build/HarmBench --frames 3 --high-order --out hamr_readiness.md
```

GPU/CPU parity evidence:

```bash
cmake --build build --target HarmGpuParity
./build/HarmGpuParity --frames 0 --grid 32 --high-order --out gpu_upload_parity.md
./build/HarmGpuParity --frames 1 --grid 32 --out gpu_cpu_parity.md
```

The zero-frame gate verifies upload/readback exactly. The GPU equations have since been ported to the covariant coordinate-basis formulation, midpoint RK2, MC reconstruction, and edge-EMF CT. The one-frame evolution gate must now be rerun on a WebGPU-capable host; prior parity numbers describe the removed legacy shader and are no longer evidence for the current implementation.

The report includes:

| Section | Purpose |
|---|---|
| Checks | Fishbone setup, divB, failures, CFL, and short-run drift gates |
| Evolution Norms | Relative/absolute state change norms for convergence and parity tracking |
| AMR Refinement Candidates | Blocks selected by density contrast, magnetization, and floor/funnel criteria |
| Adaptive Block Evolution | Region-local subcycled block evolution and uniform/adaptive parity norms |
| H-AMR Readiness | A 10-point architecture/evidence score |
| Scientific Replacement Score | A 9-point evidence score covering high-order mode, divB, failures, drift, adaptive parity, MRI quality, coarse/medium/fine stability, longer coarse evolution, and GPU compute parity audit |
| Method Validation Suite | Primitive recovery round-trip, HLL equal-state consistency, and Fishbone initial divB checks |
| AMR Block Method Checks | Ghost-cell block load/inject and restrict/prolong sanity checks |

## Honest Gap List

The engine now has an H-AMR-shaped adaptive benchmark path, but it is still not the production H-AMR code. The next scientific jumps are executable GPU/CPU parity and recovery tuning, published-code comparison tables, longer production Fishbone windows, and stronger Riemann/reconstruction options.
