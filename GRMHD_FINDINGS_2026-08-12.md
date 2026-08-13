# GRMHD Development Findings — 2026-08-12

## Executive assessment

The project is now a credible **HARM-inspired GRMHD solver-development code**, not yet a research-validated replacement for the codes compared by Porth et al. (2019). The current branch contains real conservative GRMHD evolution machinery, GPU midpoint RK2 evolution, HLL fluxes, metric-aware primitive recovery, radial floors, horizon-penetrating Kerr-Schild coordinates, and a constrained-transport magnetic update. It also has executable GPU/CPU parity and GPU benchmark harnesses.

The strongest current evidence is short-duration correctness and stability. The missing evidence is a converged, long-duration turbulent evolution through the Porth analysis interval, with resolution-ladder statistics and independent-code agreement. The rendered image is useful for debugging and presentation, but it is not itself proof that the fluid solution or radiation model is scientifically correct.

## What was fixed in this development pass

- Corrected WGSL compilation after `target` was used as an identifier even though it is reserved by the shader language.
- Reworked GPU storage allocation so changing from `96^3` to `128^3` reallocates buffers instead of uploading approximately 50 MB into buffers allocated for approximately 28 MB.
- Made the runtime grid genuinely cubic. The Porth comparison tiers are `96^3`, `128^3`, and `192^3`, not `N x N/2 x N`.
- Made the Porth-style `96^3` SANE configuration the normal application default.
- Moved the inner excision radius inside the Kerr horizon, consistent with horizon-penetrating Kerr-Schild coordinates.
- Brought CPU and GPU conservative-variable, flux, recovery, floor, and constrained-transport paths closer together.
- Added state-dependent CFL estimation and exposed the actual accepted time step. The UI's configured `dt` is a ceiling; it is not permission to violate the stability condition.
- Added executable readback diagnostics and benchmark reporting so non-finite state, recovery failures, magnetic divergence, and global drift are measurable.
- Improved the fast-light shadow renderer and camera behavior. This renderer remains a visualization model rather than validated general-relativistic radiative transfer.

## Current measured results

### GPU/CPU parity

The current short parity run uses a cubic `32 x 32 x 32` grid, three frames, high-order reconstruction, and a `0.0005` time-step ceiling. The recorded report passes and gives:

| Quantity | Result |
|---|---:|
| density L1 relative difference | `6.31e-6` |
| density L2 relative difference | `6.76e-5` |
| density Linf relative difference | `1.38e-3` |
| internal-energy L1 relative difference | `8.58e-5` |
| velocity L1 absolute difference | `4.20e-5` |
| magnetic-magnitude L1 relative difference | `4.31e-3` |
| mass diagnostic difference | `0` at report precision |
| internal-energy diagnostic difference | `1.45e-7` |
| magnetic-energy diagnostic difference | `0` at report precision |

This is good short-run implementation evidence. It is not a convergence study and does not establish long-time equivalence.

### `96^3` GPU pilot

The recorded cubic pilot evolved 1,000 steps:

| Quantity | Result |
|---|---:|
| grid | `96 x 96 x 96` (`884,736` cells) |
| wall time | `38.561 s` |
| throughput | `2.294e7` cell-updates/s |
| simulated time reached | `0.5 M` |
| mass drift | `-2.69e-5` |
| internal-energy drift | `-2.60e-5` |
| magnetic-energy drift | `+6.41e-3` |
| divB L1 | `2.84e-3` |
| recovery-failure fraction | `0` |

This is encouraging as a smoke/pilot test. It covers only `0.5 M`, whereas the Porth comparison analyzes turbulent statistics over `5,000 M` to `10,000 M`. Extrapolating the pilot linearly would also be misleading because the adaptive CFL step is state-dependent and may contract as the flow evolves.

### Apparent late-time “blow-up”

The bright, spatially fragmented state produced with a requested `dt = 0.03` was numerical, not established physical turbulence. A fixed time step that exceeds the cell- and state-dependent signal-crossing limit violates the explicit solver's CFL condition. Increasing substeps per rendered frame does not make an unsafe individual step safe; it simply requests more evolution steps between images.

The state-dependent limiter reduces the accepted step from the requested ceiling using local metric geometry and GRMHD signal speeds. Seeing an accepted value near `0.00371` therefore means the current state and mesh require that smaller step. The ceiling can remain `0.03`, but the solver must accept the smaller stable value.

## What “Porth level” requires

Porth et al. compared independent GRMHD codes statistically because turbulent disks do not match point-for-point. Reaching comparable validation quality requires all of the following:

1. Reproduce the common Fishbone-Moncrief SANE setup and document every code-dependent choice: coordinates, polar treatment, reconstruction, Riemann solver, floors, inversion fallbacks, and magnetic update.
2. Run the literal `96^3`, `128^3`, and `192^3` tiers. `96^3` is a low-resolution comparison run, not the final convergence claim.
3. Evolve through at least `10,000 M` and compute statistics over the quasi-steady `5,000 M..10,000 M` interval.
4. Record the same horizon and disk diagnostics, including mass accretion, magnetic flux, energy and angular-momentum fluxes, disk profiles, scale height, magnetization, plasma beta, and MRI quality factors.
5. Demonstrate finite-volume convergence on smooth tests and pass standard relativistic MHD shock, wave, magnetized-torus, and magnetic-divergence tests.
6. Show that atmosphere intervention and primitive-recovery fallback remain controlled, especially in the funnel and near the horizon.
7. Compare time-averaged quantities and their resolution trend with published independent-code scatter rather than judging resemblance from rendered frames.
8. Freeze a reproducible configuration, compiler/backend information, checkpoints, and machine-readable diagnostics for every production run.

Until that campaign succeeds, the accurate description remains **HARM-inspired** or **under validation against the Porth benchmark**.

## Resolution and storage conclusions

The default `96^3` mesh contains about 2.67 times as many cells as the former `96 x 48 x 96` layout. `128^3` contains about 2.37 times as many cells as `96^3`, and `192^3` contains eight times as many. Runtime will scale at least approximately with cell count and can scale worse when the CFL-limited step shrinks with resolution.

The earlier `128^3` crash was a buffer-lifecycle bug, not evidence that the GPU lacked memory. The destination had been allocated for the old `96^3` state (`28,311,552` bytes), while the new upload required `50,331,648` bytes. Reallocation on dimension change is therefore mandatory.

## Hardware outlook

The present solver is WGSL/WebGPU code dominated by ordinary FP32 arithmetic, memory traffic, transcendental functions, branches, and primitive recovery. Tensor-core marketing figures do not directly predict its speed.

| GPU | Likely role for this code |
|---|---|
| Apple M-series development GPU | convenient development and short validation runs |
| RTX 5090 | likely the fastest single consumer GPU for the current ordinary-FP32 WebGPU path |
| NVIDIA A100 | strong memory capacity, ECC, sustained server operation, and a better platform for a future CUDA implementation; not automatically faster than a 5090 for unmodified WGSL |

An A100's 40 or 80 GB of ECC HBM is ample for the Porth grids. Its value is reliability and capacity. A CUDA port could exploit the platform much better, but that is a separate optimization project and must preserve the validated numerical method.

Production-duration estimates are not trustworthy yet. A defensible estimate needs a healthy `96^3` run long enough to observe the steady adaptive time-step distribution, plus measurements from the actual target GPU. The immediate milestone should be sustained `96^3` evolution with periodic checkpoints and diagnostic gates, followed by `128^3`; starting `192^3` before those pass would generate expensive but weak evidence.

## Recommended validation sequence

1. Keep upload/readback parity and short GPU/CPU evolution parity as required shader-change gates.
2. Add automated non-finite detection, CFL history, recovery/floor maps, conservation histories, and restart checkpoints.
3. Run `32^3` and `96^3` endurance tests, increasing simulated duration geometrically and stopping on explicit diagnostic thresholds.
4. Complete a `96^3` run to `10,000 M` and inspect statistics over `5,000 M..10,000 M`.
5. Repeat at `128^3`, then attempt `192^3` only after the lower tiers remain stable and scientifically interpretable.
6. Validate the image independently with a recognized GRRT pipeline before calling the rendered emission Porth-level or observationally predictive.

## Bottom line

The code has advanced substantially: it now has the right low-resolution cubic target, meaningful GPU/CPU checks, adaptive CFL protection, buffer resizing, and a more faithful conservative GRMHD structure. The results justify continued validation. They do **not** yet justify a research-grade convergence or Porth-level confirmation claim. The decisive work remaining is long-duration evolution, resolution convergence, standard test problems, published diagnostic comparison, and validated radiative transfer.
