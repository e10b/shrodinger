# Porth et al. 2019 GRMHD Comparison Notes

Source: `Porth_2019_ApJS_243_26.pdf`, *The Event Horizon General Relativistic Magnetohydrodynamic Code Comparison Project*.

This paper compares nine GRMHD codes on the same black-hole accretion benchmark. The important thing: they do not prove accuracy by matching an analytic turbulent disk solution. There is no such solution. They prove reliability by standardizing the physical problem, applying common diagnostics, increasing resolution, and showing that independent codes move toward the same statistical answer.

## Benchmark Problem

The shared test is a 3D, non-radiative, ideal-GRMHD SANE accretion flow around a Kerr black hole.

| Setting | Value |
|---|---|
| Spacetime | Kerr black hole |
| Spin | `a = 0.9375` |
| Units | Geometric units, `G = c = 1`; lengths/times in `M` |
| Initial disk | Fishbone-Moncrief equilibrium torus |
| Torus inner radius | `r_in = 6M` |
| Density maximum | `r_max = 12M` |
| Equation of state | Ideal gas, `gamma = 4/3` |
| Magnetic field | Single weak poloidal loop |
| Vector potential | `A_phi proportional to max(rho / rho_max - 0.2, 0)` |
| Field strength | Weak-field torus, roughly `p_max / B_max^2 = 100` |
| MRI trigger | Thermal pressure white-noise perturbation of 4 percent total amplitude |
| Target regime | SANE, not MAD |
| Suggested floors | Density and pressure power-law floors, with code-specific fail-safes |
| Domain of interest | `r` from horizon neighborhood to `50M`, `theta in [0, pi]`, `phi in [0, 2pi]` |
| Resolution rounds | Low `96^3`, medium `128^3`, high `192^3` in the domain of interest |
| Analysis window | Usually `t = 5000M` to `10000M` |

The authors intentionally allowed each code to use its normal grid, boundary, floor, coordinate, and inversion practices. That makes the comparison closer to real community usage, but it also means the fair resolution variable is not only cell count. They often compare by proper midplane grid spacing near `r = 12M`.

## What They Measure

The paper uses a consistent diagnostic set in standard Kerr-Schild coordinates.

| Diagnostic | What it tests |
|---|---|
| Horizon mass accretion rate, `Mdot` | Whether the saturated turbulent disk transports mass consistently |
| Horizon magnetic flux, `Phi_BH`, and normalized flux `phi = Phi_BH / sqrt(Mdot)` | Whether the simulation lands in SANE/MAD-like magnetic-flux regime |
| Horizon angular momentum flux, `Ldot` | Whether the disk/black-hole angular momentum exchange agrees |
| Horizon energy flux, `Edot` | Whether jet/disk energetics agree |
| Disk-averaged shell profiles | Radial structure of `rho`, pressure/internal energy, magnetic field, beta, and magnetization |
| Emission proxy light curve | Synthetic optically thin synchrotron-like variability without full radiation transport |
| Time and azimuthal averages | Global disk/funnel morphology, especially density, inverse beta, and magnetization |
| MRI quality factors, `Q_r`, `Q_theta`, `Q_phi` | Whether the grid resolves the fastest-growing MRI wavelengths |
| Maxwell stress / alpha | Whether turbulent angular momentum transport is comparable |
| Jet/funnel contours | Shape of the `sigma = 1` boundary and funnel-wall behavior |
| Disk spreading / barycentric radius | Whether turbulent transport causes the torus to spread similarly |
| Run-to-run variation | Separates chaos/roundoff sensitivity from algorithmic differences |

## Accuracy Argument

The accuracy story is statistical convergence, not pointwise equality.

The key evidence is Table 2 and Figures 2-5. They report peak and time-averaged horizon fluxes for each code and resolution. Then they compare the scatter across codes. As resolution rises, the max/min spread drops strongly.

| Quantity | Low `96^3` max/min | High `192^3` max/min | Meaning |
|---|---:|---:|---|
| Peak `Mdot` | `2.406` | `1.649` | Mass inflow peaks become more consistent |
| Mean `Mdot` | `3.903` | `1.772` | Saturated accretion rates converge statistically |
| Peak `Phi_BH / sqrt(Mdot)` | `12.374` | `2.116` | Magnetic-flux disagreement shrinks a lot |
| Mean `Phi_BH / sqrt(Mdot)` | `17.408` | `2.777` | SANE flux level becomes much more code-independent |
| Peak `Ldot / Mdot` | `1.163` | `1.077` | Angular momentum flux is robust even at low resolution |
| Mean `Ldot / Mdot` | `1.823` | `1.121` | High-res angular momentum transport agrees well |
| Peak `-Edot / Mdot` | `1.448` | `1.250` | Energy-flux peaks tighten |
| Mean `-Edot / Mdot` | `7.096` | `1.405` | Mean energy flux becomes much less scattered |

Their interpretation is that agreement improves once the disk MRI is sufficiently resolved. The MRI quality factors are the physical resolution check. The commonly cited target is roughly `Q_z >= 10` and `Q_phi >= 20-25`, or a product criterion around `Q_z * Q_phi >= 200-250`. In their `96^3` and `128^3` cases, poloidal MRI resolution is often too low; the `192^3` cases are near the useful edge; higher BHAC/H-AMR runs show clearer convergence.

The paper also checks that algorithmic differences are larger than normal chaotic run-to-run variation by rerunning KORAL on different machines with different perturbations. Most repeated-run differences are low-percent except some peak and magnetic-flux values.

## Methods Compared

| Code/method | Core scheme | What distinguishes it |
|---|---|---|
| Athena++ | Second-order time integration, HLL fluxes, PPM reconstruction, staggered constrained transport | General finite-volume framework with static mesh derefinement near the poles; strong pole treatment; Kerr-Schild spherical grid |
| BHAC | Second-order predictor-corrector, LLF fluxes, PPM, upwind constrained transport | AMR-oriented GRMHD code; modified Kerr-Schild coordinates; also tested a Cartesian Kerr-Schild AMR variant |
| Cosmos++ | Third-order SSPRK, HLL, PPM, staggered constrained transport | Object-oriented C++ code with unstructured/adaptive mesh heritage; used polar cutouts and outflow polar boundaries, which affected magnetic flux |
| ECHO | Third-order IMEX/RK-like time stepping, HLL, PPM, upwind constrained transport | High-order conservative scheme lineage; excludes small polar regions to avoid tiny time steps; robust multi-stage primitive recovery |
| H-AMR | Second-order time, HLL, PPM, staggered/upwind magnetic update | HARM-derived GPU/AMR code with local adaptive time stepping; drift-frame floor injection; very high-resolution reference run |
| HARM-Noble | RK2, local Lax-Friedrichs, PPM, FluxCT | Close to original HARM lineage; robust primitive recovery via 2D/1DW inversion and entropy fix; used larger radial domain, so lower effective inner radial resolution |
| iharm3D | Second-order predictor-corrector, LLF, PLM, FluxCT | HARM-family conservative code; only main comparison code using PLM instead of PPM; funky modified Kerr-Schild coordinates with pole derefinement |
| IllinoisGRMHD | RK4, HLL, PPM, vector-potential magnetic evolution | Cartesian AMR code from numerical relativity; evolves vector potential so magnetic divergence control survives AMR interpolation |
| KORAL | Second-order, LLF, PPM, FluxCT | Closely follows iharm3D-style MHD but also supports radiation/two-temperature/nonthermal physics; modified Kerr-Schild grid |

## Method Lessons

PPM matters. Appendix A compares PLM and PPM runs in Athena++, H-AMR, and KORAL. Even at `192^3`, PLM can underpredict accretion by factors of a few compared with PPM. The authors argue PPM reduces effective numerical dissipation for these HLL/LLF schemes.

Magnetic divergence control is non-negotiable. Every production method preserves `divB` to machine precision or evolves a vector potential that enforces the constraint through the magnetic-field reconstruction.

Primitive recovery and floors are part of the method, not implementation trivia. Each code has specific handling for failed inversions, magnetized funnels, Lorentz-factor ceilings, entropy fallback, and density/internal-energy floors. These choices show up most strongly in funnel and jet diagnostics.

Coordinates and boundary choices affect measured accuracy. Cosmos++ had lower magnetic flux because its polar cutout/outflow boundaries allowed flux leakage. Cartesian grids avoid polar singularities but have different near-horizon resolution behavior.

Cell count is not enough. Proper grid spacing at the disk midplane and MRI quality factors explain convergence better than nominal `N^3`.

## What This Means For `shrodinger`

To claim HARM-level accuracy, our validation target should be a small version of this comparison:

1. Implement the exact Fishbone-Moncrief torus with `a = 0.9375`, `r_in = 6M`, `r_max = 12M`, `gamma = 4/3`.
2. Add the single-loop vector potential initialization and normalize to the paper's weak-field target.
3. Evolve conservative GRMHD variables with finite-volume fluxes, primitive recovery, floors, and constrained transport.
4. Output the Table 2 diagnostics: `Mdot`, `Phi_BH / sqrt(Mdot)`, `Ldot / Mdot`, `-Edot / Mdot`, peak and mean over `5000M-10000M`.
5. Add MRI quality-factor diagnostics and report `Q_r`, `Q_theta`, `Q_phi`.
6. Run at at least two resolutions first, then show spread/trend against a reference implementation or published benchmark values.
7. Treat image quality as secondary. For accuracy, the solver must match flux statistics, disk profiles, MRI quality, and magnetic divergence behavior.

The publishable language before that validation exists should be "HARM-style" or "HARM-inspired." The language after these diagnostics pass can move toward "validated against the EHT GRMHD code comparison benchmark."
