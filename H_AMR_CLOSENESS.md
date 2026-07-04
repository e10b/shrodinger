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
| Adaptive block-local subcycling | Present |
| State norms for convergence/parity tracking | Present |

## How To Generate The Score

```bash
cmake --build build --target HarmBench
./build/HarmBench --frames 3 --high-order --out hamr_readiness.md
```

The report includes:

| Section | Purpose |
|---|---|
| Checks | Fishbone setup, divB, failures, CFL, and short-run drift gates |
| Evolution Norms | Relative/absolute state change norms for convergence and parity tracking |
| AMR Refinement Candidates | Blocks selected by density contrast, magnetization, and floor/funnel criteria |
| Adaptive Block Evolution | Region-local subcycled block evolution and uniform/adaptive parity norms |
| H-AMR Readiness | A 10-point architecture/evidence score |

## Honest Gap List

The engine now has an H-AMR-shaped adaptive benchmark path, but it is still not the production H-AMR code. The next scientific jumps are GPU/CPU state readback parity in the benchmark runner, longer Fishbone evolution windows, resolution convergence, stronger Riemann/reconstruction options, and promoting the adaptive block-local path into the interactive GPU hot loop.
