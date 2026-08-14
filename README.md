# shrodinger

HARM-inspired black-hole accretion solver-development renderer built on **[wgfx](https://github.com/Vyscosity/wgfx)**, WebGPU, SDL3, and Dear ImGui.

The app boots directly into a 3D HARM-inspired GRMHD scene. It evolves a packed primitive field on the GPU, renders density/magnetization/beta/velocity/shadow views from WGSL, and exposes camera, grid, initial-data, and diagnostics controls through ImGui.

For a quick presentation, click **Fast chaos demo** in the control panel or launch
with `./build/App --chaos-demo`. This selects a `32^3`, 32-substep GPU movie
preset so orbital structure visibly moves without changing the normal `96^3`
Porth setup. The underlying field still evolves with the CFL-safe solver; the
raytraced view adds clearly labeled velocity-advected flow tracers at 60x
playback so a hundreds-of-`M` orbit is perceptible in a short demonstration.
It is a coarse visualization mode, not validation evidence.

For a more aggressive presentation, click **MAD CHAOS** or run
`./build/App --mad-chaos`. This uses strongly magnetized MAD-seeded initial
data with asymmetric fluid perturbations and accelerated magnetic
knots/plunging-stream tracers. It is intentionally cinematic; a physically
developed MAD state still requires a long, stable evolution.

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

GPU/CPU parity harness:

```bash
cmake --build out --target HarmGpuParity -j
./out/HarmGpuParity --frames 0 --grid 32 --high-order --out gpu_upload_parity.md
./out/HarmGpuParity --frames 1 --grid 32 --out gpu_cpu_parity.md
```

The zero-frame mode is an exact upload/readback gate. The one-frame mode is a strict executable evolution gate that compares CPU and GPU state norms plus mass, internal energy, magnetic energy, mdot, divB, recovery failures, floor mass, and MRI quality factors. It must be rerun after shader changes on a WebGPU-capable host; source inspection never counts as parity evidence.

Resumable long GPU run by simulated time:

```bash
cmake --build out --target HarmGpuBench -j
./out/HarmGpuBench --grid 32 --target-time 10000 --substeps 200 --dt 0.03 \
  --high-order --checkpoint-every 100 --checkpoint harm_32_t10000.chk \
  --out harm_32_t10000.md
```

If interrupted, add `--resume harm_32_t10000.chk`. The requested `dt` remains a ceiling; the device-side state-dependent CFL reduction chooses the actual stable step. Each checkpoint batch prints simulated time, accepted step, recovery-failure fraction, and magnetic-divergence L1.

Useful runtime flags:

| Flag | Role |
|------|------|
| `--headless` | Render frames to an mp4 through ffmpeg |
| `--frames N` | Number of headless frames to render |
| `--resolution WxH` | Headless render resolution |
| `--video path.mp4` | Output path for headless video |
| `--grid N` | Override the cubic HARM grid size (`N x N x N`; default `96^3`) |
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
| `example/harm_amr_blocks.h` | AMR block storage, ghost loading, restriction, prolongation, and injection helpers |
| `example/harm_amr_evolution.h` | Adaptive block-local subcycling benchmark path |
| `example/harm_scientific_score.h` | Scientific replacement scoring for stability, parity, coarse/fine sanity, and GPU compute parity |
| `example/harm_method_suite.h` | Primitive recovery, flux consistency, and divB method checks |
| `example/harm_state_norms.h` | State-difference norms for convergence and CPU/GPU parity reports |
| `example/harm_primitive_recovery.h` | Conservative-to-primitive recovery path with entropy-style fallback |
| `example/harm_constrained_transport.h` | Magnetic divergence and ideal-MHD electric-field helpers |
| `example/harm_cpu_solver.h` | Readable CPU reference evolution path for solver development |
| `example/harm_benchmark.cpp` | Fishbone-Moncrief benchmark runner |
| `example/harm_gpu_parity.cpp` | WebGPU readback parity runner for CPU/GPU HARM evolution checks |
| `example/harm_validation.h` | Porth-style benchmark report checks |
| `H_AMR_CLOSENESS.md` | Current H-AMR compatibility/readiness map |
| `res/harm_grmhd.wgsl` | HARM render shader |
| `res/harm_grmhd_compute.wgsl` | HARM GPU evolution shader |
| `deps/wgfx` | WebGPU/SDL graphics layer |
| `deps/imgui` | Dear ImGui |

## Notes

This is currently a HARM-only engine. Earlier alternate simulation paths have been removed from the runtime and source facade so the published surface matches the black-hole accretion focus.

The CPU reference uses covariant stress-energy conserved variables and fluxes, metric-aware primitive recovery, RK2 time integration, an excision surface inside the horizon, and an edge-EMF constrained-transport magnetic update. The GPU shader now follows the same coordinate-basis equations with midpoint RK2 and edge-EMF CT, but it remains unvalidated until the executable `HarmGpuParity` readback test can run on a WebGPU-capable host and pass. Long-duration Porth-style convergence has not yet been demonstrated.
