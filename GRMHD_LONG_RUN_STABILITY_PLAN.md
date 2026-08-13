# GRMHD Long-Run Magnetic Stability Plan

## Observed failure

The verified NVIDIA L4/Vulkan run used `32^3`, high-order reconstruction, a requested `dt` ceiling of `0.03`, 200 substeps per host batch, and state-dependent CFL reduction. It remained finite and reported no primitive-recovery failures, but its magnetic divergence accelerated:

| Simulated time | Accepted dt | `divB L1` | Growth from initial |
|---:|---:|---:|---:|
| initial | - | `2.72251e-5` | `1.0x` |
| `24.90784M` | `3.90696e-4` | `2.80581e-5` | `1.03x` |
| `40.73392M` | `2.43560e-4` | `4.08489e-4` | `15.0x` |
| `48.10112M` | `2.48700e-4` | `5.01672e-3` | `184x` |

This is a numerical failure, not developed MRI turbulence. A run that survives computationally while violating the solenoidal magnetic constraint is not a valid GRMHD solution.

## Implemented first stability slice and L4 result

The first implementation slice completed the following items:

- corrected the scalar `E_r`, `E_theta`, and `E_phi` edge-family mapping in the CPU and WGSL curls;
- removed post-CT polar `B^theta = 0` writes and GPU magnetic component clipping;
- removed the unsafe `5e-5` accepted-step floor and refreshed CFL every RK step;
- added a periodic non-cubic `div(curl E) = 0` method test, which passed with a maximum FP32 change of `9.53674e-7`;
- added a real GPU recovery-failure counter and automatic non-finite, divergence-growth, recovery-fraction, and timestep-collapse gates that preserve the previous good checkpoint.

The corrected code passed local Metal CPU/GPU parity. A real NVIDIA L4/Vulkan replay then gave:

| Simulated time | Accepted dt | Recovery fail fraction | `divB L1` |
|---:|---:|---:|---:|
| initial | - | `0` | `2.72251e-5` |
| `17.13525M` | `3.92286e-4` | `2.13623e-4` | `2.73292e-5` |
| `22.98525M` | `3.91187e-4` | `3.66211e-4` | `2.85788e-5` |
| `24.93525M` | `3.90736e-4` | `3.78418e-3` | `3.27892e-5` |

The gate stopped at `24.93525M` and preserved the `22.98525M` checkpoint. Unlike the old run, magnetic divergence did not run away: at the stop it was only `1.20x` its initial value rather than heading toward the previous `184x` failure. The newly visible limiter is primitive inversion.

Two controlled replays isolated that limiter:

- reducing the ceiling to `dt=1e-4` failed at essentially the same physical time, `24.98525M`, so it is not a CFL-overstep failure;
- raising the heuristic inversion from 10 to 32 iterations did not help, and a stronger damped momentum correction improved a one-step test but still failed the long replay at `24.93525M`.

Therefore the next required implementation is a production primitive recovery method with a safeguarded Newton solve and evolved-entropy fallback. More heuristic iterations, a smaller timestep ceiling, or a looser failure gate are explicitly rejected as fixes.

## What Porth-level codes do

The Porth et al. (2019) comparison does not rely on a small CFL number to control magnetic divergence. Every participating production method uses a magnetic representation whose discrete topology preserves the constraint:

| Code | Magnetic method in the comparison |
|---|---|
| Athena++ | staggered constrained transport, Gardiner & Stone (2005) |
| BHAC | staggered upwind constrained transport, Del Zanna et al. (2007) |
| Cosmos++ | staggered constrained transport |
| ECHO | staggered UCT, Del Zanna et al. (2007) |
| H-AMR | staggered magnetic fields with velocity-upwinded edge electric fields, Gardiner & Stone (2005) |
| HARM-Noble | PPM edge electric fields plus Tóth (2000) FluxCT |
| iharm3D | zone-centered variables plus Tóth (2000) FluxCT; reported divergence-free to machine precision |
| IllinoisGRMHD | evolves the vector potential; reduces to staggered FluxCT on a uniform grid |
| KORAL | Tóth (2000) FluxCT |

H-AMR combines this topological divergence control with PPM reconstruction, HLL fluxes, second-order time integration, magnetization-aware drift-frame floors, entropy evolution as a recovery fallback, pole derefinement, and locally adaptive time stepping. Those other techniques matter for a `10,000M` flow, but they cannot compensate for an induction update that does not satisfy the discrete divergence identity.

## Immediate defect in the present curl

The current CPU and WGSL implementations calculate three edge locations:

- `edgeR`: the r-directed edge carrying scalar `E_r`;
- `edgeT`: the theta-directed edge carrying scalar `E_theta`;
- `edgeP`: the phi-directed edge carrying scalar `E_phi`.

For densitized magnetic field `mathcal{B}^i = sqrt(-g) B^i`, Faraday's law requires:

```text
d mathcal{B}^r / dt     = -(delta_theta E_phi - delta_phi E_theta)
d mathcal{B}^theta / dt = -(delta_phi E_r     - delta_r E_phi)
d mathcal{B}^phi / dt   = -(delta_r E_theta   - delta_theta E_r)
```

The current update uses:

```text
B^r:     edgeP.z and edgeR.y       # second family is wrong
B^theta: edgeR.x and edgeP.z       # family mapping is consistent
B^phi:   edgeP.y and edgeP.x       # both families are wrong
```

`edgeT` is computed but not used in the magnetic update. The required component/location mapping is:

```text
B^r:     edgeP.z and edgeT.y
B^theta: edgeR.x and edgeP.z
B^phi:   edgeT.y and edgeR.x
```

This defect alone destroys the discrete `div(curl(E)) = 0` cancellation. It must be fixed in both CPU and GPU implementations before any further endurance test.

However, correcting those indices is only the first gate. The current representation remains cell-centered and applies non-topological magnetic edits in recovery and boundary handling, so it cannot yet make the same machine-precision claim as Porth's production codes.

## Other present mechanisms that can create divergence

### Cell-centered magnetic storage

The code stores all three magnetic components at cell centers. A true finite-volume CT scheme stores the normal magnetic flux on each cell face. Neighboring cells then share the same face value, and edge EMF differences cancel exactly when taking the finite-volume divergence.

The current cell-centered central-difference diagnostic is not the exact algebraic divergence paired with the update. Even a correctly indexed cell-centered curl can lose cancellation at nonuniform radial spacing and boundaries.

### Arithmetic, non-upwind edge EMFs

The present edge EMF is a simple average of four cell-centered `-sqrt(-g) v cross B` values. H-AMR instead velocity-upwinds the edge fields, while Athena++ and modern UCT methods derive edge EMFs consistently from reconstructed face states and Riemann fluxes. The arithmetic average lacks the multidimensional dissipation needed to suppress grid-scale/odd-even magnetic modes.

### Primitive recovery can edit the magnetic field

Primitive recovery copies `B` from the conserved target, but the shared `timelike` sanitizer clamps each magnetic component to `[-10, 10]`. Any such componentwise edit violates CT. In a CT code, face magnetic flux is authoritative; primitive recovery solves the fluid variables while treating the interpolated cell-centered magnetic field as fixed input.

### Polar boundary modifies `B^theta` after CT

`applyBoundary`/`applyOutflow` directly sets `B^theta = 0` in the first and last polar cell. This happens after the curl update and is not itself a curl, so it can inject magnetic monopoles. The clamped theta neighbor lookup also copies the terminal cell instead of applying a geometrically correct across-axis parity map.

Porth participants handle the axis deliberately: Athena++ communicates data across the pole and uses pole magnetic fluxes in edge EMFs; BHAC uses diagonal/π-periodic ghost mapping and zeros selected axis flux/EMF components; HARM-Noble and ECHO avoid the singular axis with a polar cutout. Any of these is more defensible than editing a cell-centered magnetic component after evolution.

### Initialization and evolution do not share one discrete operator

The vector-potential initializer produces cell-centered `B` through central differences. Production CT should initialize face fluxes from a discrete curl of edge/vertex vector potential using exactly the same mesh incidence relations used during evolution. Its compatible finite-volume divergence should be roundoff-level at time zero.

### CFL has an absolute lower clamp

The GPU CFL function clamps the accepted step to at least `5e-5`. If the physical stability limit falls below that during a long run, the solver will knowingly overstep it. Production behavior should allow a smaller positive step, detect a stalled/collapsing timestep explicitly, checkpoint, and abort with a diagnostic rather than violate CFL.

The CFL reduction is also refreshed only every eight substeps. It should be refreshed at least once per full RK step during validation, and preferably for each stage until stability is established.

## Target architecture: staggered, densitized CT

### Authoritative state

Add three GPU face buffers, separate from cell-centered primitives:

```text
Br_face: (Nr + 1) x Ntheta x Nphi
Bt_face: Nr x (Ntheta + 1) x Nphi
Bp_face: Nr x Ntheta x (Nphi + 1)
```

Store the integrated/densitized normal magnetic flux, not an independently recoverable cell-centered field. Periodic phi can alias the final face to the first or store one redundant plane with an enforced copy.

Cell-centered `B^i` used by stress-energy, reconstruction, and primitive recovery is derived from the two adjacent face fluxes with metric/area-aware interpolation. It is never evolved or clipped independently.

### Edge EMFs

Store or compute the three scalar line-integrated edge EMFs:

```text
Er_edge: Nr x (Ntheta + 1) x (Nphi + 1)
Et_edge: (Nr + 1) x Ntheta x (Nphi + 1)
Ep_edge: (Nr + 1) x (Ntheta + 1) x Nphi
```

Construct them from the induction components of the same face Riemann fluxes used by the finite-volume fluid update. First implement a known FluxCT mapping as the lower-risk HARM-family baseline. Then upgrade to Gardiner-Stone/UCT edge upwinding once the exact divergence identity passes.

### Face update

Update each face flux by the oriented circulation of edge EMFs around that face. The geometry enters through stored face areas/edge lengths or through densitized coordinate fluxes, but the incidence signs must be purely topological and shared by adjacent faces.

With one common edge value, taking the divergence of the updated face fluxes cancels every interior edge contribution pairwise. This is the property the present implementation lacks.

### RK staging

Each RK stage needs its own consistent face field:

1. derive cell-centered `B` from the stage's face fluxes;
2. reconstruct face states while preserving the normal face field;
3. solve face Riemann problems;
4. build edge EMFs from those face fluxes;
5. advance face magnetic flux and cell-centered hydrodynamic conserved variables with the same stage time;
6. recover only hydrodynamic primitives with `B` held fixed from faces.

Do not update `B` through the ordinary conservative flux divergence and CT simultaneously.

## Implementation sequence and hard gates

### Phase 0: make failure automatic

Before changing the method:

- report both absolute and normalized divergence, e.g. `h |div B| / (|B| + epsilon)`;
- record `divB L1`, `divB max`, minimum dt, recovery failures, floor injection, mass/energy drift, and non-finite counts at every checkpoint;
- abort and retain the previous good checkpoint if divergence grows by more than a configured factor or any non-finite value appears;
- write progress/report rows incrementally rather than only at normal termination.

Gate: the existing L4 failure must be detected and stopped automatically near its onset, without manual log inspection.

### Phase 1: repair and prove the discrete curl indexing

- correct the edge-family/component mapping in CPU and WGSL;
- remove unused/misleading vector-valued edge variables in favor of scalar `Er`, `Etheta`, and `Ephi` accessors;
- add a unit test that fills a periodic uniform mesh with arbitrary edge EMFs, applies one curl update, and verifies that the paired discrete divergence is unchanged to FP32 roundoff;
- add CPU/GPU parity for the magnetic update alone.

Gate: `div_after - div_before` is roundoff-level for random edge fields on periodic meshes, including non-cubic dimensions.

This phase may extend the current run, but it is not the final production architecture.

### Phase 2: prohibit non-CT magnetic edits

- remove magnetic component clamps from primitive sanitization;
- treat face-derived `B` as immutable input to primitive recovery;
- replace polar-cell `B^theta = 0` edits with ghost/face parity rules;
- audit every floor, recovery fallback, boundary, and synchronization path for writes to magnetic state;
- add a counter that trips if a non-CT kernel changes a face magnetic buffer.

Gate: recovery/floor/boundary tests preserve the compatible divergence exactly.

### Phase 3: implement face-centered densitized CT

- introduce authoritative face buffers and stage buffers;
- initialize them from a discrete curl of `A_phi`;
- derive cell-centered magnetic fields only for the fluid calculation and rendering;
- update face fields using common edge scalar EMFs;
- replace the diagnostic with finite-volume face-flux divergence.

Gate: the initial torus and 10,000 random-EMF update cycles retain divergence at roundoff without fluid evolution.

### Phase 4: implement upwind FluxCT/UCT

- first reproduce Tóth FluxCT using the induction components of face Riemann fluxes, matching iharm3D/KORAL/HARM-Noble lineage;
- add Gardiner-Stone velocity upwinding or Del Zanna UCT for the H-AMR-level target;
- upgrade the Riemann wave bounds from one symmetric maximum speed to directional left/right bounds where required by the selected UCT formula;
- retain one arithmetic-EMF mode only as a negative/control test.

Gate: multidimensional field-loop advection does not develop checkerboard fields and shows convergent magnetic-energy loss without secular divergence.

### Phase 5: pole and boundary policy

Choose and document one strategy before production:

1. full-axis parity mapping with shared pole fluxes/EMFs, closest to Athena++;
2. diagonal π-periodic ghost mapping plus explicit axis EMF rules, similar to BHAC;
3. a small polar cutout with reflecting/outflow parity, similar to HARM-Noble/ECHO and simplest for the first reliable spherical run.

For the immediate `32^3` endurance milestone, a documented small polar cutout is the lowest-risk choice. Later full-domain Porth comparisons can add a verified across-axis treatment.

Gate: a stationary analytic divergence-free field remains divergence-free with radial and polar boundary kernels active.

### Phase 6: CFL and recovery robustness

- remove the unsafe absolute minimum accepted dt;
- recompute CFL every full step during production validation;
- checkpoint and abort on timestep collapse instead of overstepping;
- add conserved entropy and use it as a primitive-recovery fallback in highly magnetized cells, following H-AMR/HARM-Noble practice;
- add magnetization-aware floors without altering face magnetic flux;
- track injected mass, internal energy, and momentum by region and frame choice.

Gate: standard relativistic MHD waves/shocks and magnetized atmosphere tests remain finite without divergence growth or unexplained conserved-variable edits.

## Validation ladder

Do not jump directly back to `10,000M`. Each rung must start from the same frozen configuration and pass all previous gates:

| Rung | Grid/time | Required evidence |
|---|---|---|
| A | algebraic operator tests | exact compatible `div(curl)=0`; CPU/GPU CT parity |
| B | `32^3` to `50M` | no secular normalized-divB growth; no non-finite state |
| C | `32^3` to `100M` | divergence remains bounded; recovery/floor intervention explained |
| D | `32^3` to `500M` | stable conservation histories and checkpoint/resume parity |
| E | `32^3` to `1,000M` | MRI/turbulence onset remains numerically controlled |
| F | `32^3` to `10,000M` | valid endurance engineering run; not a converged Porth science run |
| G | `96^3`, `128^3`, `192^3` | Porth statistical window `5,000M..10,000M`, resolution trends, published diagnostics |

Suggested magnetic acceptance criterion after face CT is implemented:

- compatible divergence changes only at FP32 roundoff for pure CT operator tests;
- volume-averaged normalized divergence shows no secular growth in coupled runs;
- maximum normalized divergence excludes explicitly documented boundary singular/cutout cells and remains below a predeclared threshold;
- lowering the CFL ceiling changes truncation error but does not qualitatively change divergence stability.

The threshold should be defined in normalized finite-volume form before looking at production results, not tuned afterward to make a run pass.

## Effort estimate

| Work | Approximate effort |
|---|---:|
| fix EMF mapping, scalarize accessors, add exact operator test | 1-2 focused days |
| remove non-CT magnetic edits and repair boundary parity | 1-3 days |
| face-centered GPU buffers, initialization, RK staging, readback | 4-8 days |
| FluxCT from face Riemann fluxes plus CPU/GPU tests | 3-6 days |
| pole policy, standard MHD tests, automated abort/checkpoint gates | 3-7 days |
| endurance ladder through `1,000M` with debugging | several days of engineering plus runtime |

A credible first `32^3 -> 10,000M` attempt is therefore likely one to three weeks of focused numerical implementation and validation, not a parameter adjustment. A Porth-level `96^3/128^3/192^3` campaign comes afterward.

## Recommended immediate next action

Implement only Phase 0 and Phase 1 first, then rerun the verified L4 procedure to `50M`. If the corrected mapping eliminates the explosive divergence growth, proceed to Phase 2 and face-centered CT. If it does not, the new operator test and incremental diagnostics will localize whether the remaining source is initialization, polar boundaries, recovery edits, or incompatible divergence measurement.
