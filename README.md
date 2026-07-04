# shrodinger

HARM GRMHD black-hole accretion renderer built on **[wgfx](https://github.com/Vyscosity/wgfx)**, WebGPU, SDL3, and Dear ImGui.

The app boots directly into a 3D HARM-inspired GRMHD scene. It evolves a packed primitive field on the GPU, renders density/magnetization/beta/velocity/shadow views from WGSL, and exposes camera, grid, initial-data, and diagnostics controls through ImGui.

## Clone

```bash
git clone --recurse-submodules git@github.com:vyscosity/shrodinger.git
cd shrodinger
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Build

```bash
cmake -S . -B out -DCMAKE_BUILD_TYPE=Release
cmake --build out -j
./out/App
```

Headless render:

```bash
./out/App --headless --frames 600 --resolution 1920x1080 --video output.mp4
```

Fishbone-Moncrief benchmark:

```bash
cmake --build out --target HarmBench -j
./out/HarmBench --frames 3 --out fishbone_benchmark.md
```

The benchmark writes a Markdown report with Porth-style setup checks plus short-run stability gates: density peak radius, torus inner edge, weak-loop beta, magnetic divergence, primitive-recovery failure fraction, CFL, and mass/internal/magnetic-energy drift.

Benchmark-specific flags:

| Flag | Role |
|------|------|
| `--grid N` | Override benchmark grid size |
| `--dt value` | Override benchmark CFL time step ceiling |
| `--substeps N` | Override CPU reference substeps per frame |
| `--high-order` | Use MC-limited reconstructed interface states |

Useful runtime flags:

| Flag | Role |
|------|------|
| `--headless` | Render frames to an mp4 through ffmpeg |
| `--frames N` | Number of headless frames to render |
| `--resolution WxH` | Headless render resolution |
| `--video path.mp4` | Output path for headless video |
| `--grid N` | Override the maximum HARM grid size |
| `--play` | Start camera-keyframe playback |

## Layout

| Path | Role |
|------|------|
| `example/main.cpp` | App entry point and render loop |
| `example/quad.h` | Thin compatibility facade used by the app loop |
| `example/harm_*.h` | Reusable HARM components: config, grid, initial data, diagnostics, GPU compute, renderer, camera, controller |
| `example/harm_kerr_schild.h` | Kerr-Schild metric utilities used by diagnostics and the reference solver |
| `example/harm_geometry.h` | Shared logarithmic Kerr-Schild grid geometry, cell sizes, volumes, and metric lookup |
| `example/harm_state.h` | Primitive/conserved state conversion scaffolding |
| `example/harm_flux.h` | HLL flux and limited reconstruction helpers |
| `example/harm_timestep.h` | CFL-limited time-step estimator used by the CPU reference solver |
| `example/harm_boundaries.h` | Inner/outer radial outflow and polar boundary policy |
| `example/harm_amr.h` | Block-refinement candidate scoring for H-AMR-style adaptive mesh planning |
| `example/harm_amr_evolution.h` | Adaptive block-local subcycling benchmark path |
| `example/harm_scientific_score.h` | Scientific replacement scoring for stability, parity, coarse/fine sanity, and GPU compute parity |
| `example/harm_state_norms.h` | State-difference norms for convergence and CPU/GPU parity reports |
| `example/harm_primitive_recovery.h` | Conservative-to-primitive recovery path with entropy-style fallback |
| `example/harm_constrained_transport.h` | Magnetic divergence and ideal-MHD electric-field helpers |
| `example/harm_cpu_solver.h` | Readable CPU reference evolution path for solver development |
| `example/harm_benchmark.cpp` | Fishbone-Moncrief benchmark runner |
| `example/harm_validation.h` | Porth-style benchmark report checks |
| `H_AMR_CLOSENESS.md` | Current H-AMR compatibility/readiness map |
| `res/harm_grmhd.wgsl` | HARM render shader |
| `res/harm_grmhd_compute.wgsl` | HARM GPU evolution shader |
| `deps/wgfx` | WebGPU/SDL graphics layer |
| `deps/imgui` | Dear ImGui |

## Notes

This is currently a HARM-only engine. Earlier alternate simulation paths have been removed from the runtime and source facade so the published surface matches the black-hole accretion focus.

The solver is being moved toward a HARM/H-AMR-style architecture: shared Kerr-Schild grid geometry, conservative variables, HLL fluxes, MC-limited reconstruction, adaptive CFL stepping, scalar primitive recovery, metric-derivative source terms, outflow/polar boundary policy, magnetic-divergence controls, MRI quality diagnostics, and a readable CPU reference path before the same math is fully promoted into the GPU hot path.
