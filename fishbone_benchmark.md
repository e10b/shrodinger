# Fishbone-Moncrief Benchmark Report

This is a quick Porth-style initial-data and short-evolution check for the HARM GRMHD engine.

## Setup

| Setting | Value |
|---|---:|
| spin `a` | 0.9375 |
| grid | 64 x 32 x 64 |
| radial domain | 1.34933 to 50 |
| frames evolved | 60 |

## Checks

| Check | Value | Target | Tolerance | Result |
|---|---:|---:|---:|---|
| spin a | 0.9375 | 0.9375 | 0.0001 | PASS |
| density peak radius | 11.8544 | 12 | 1.25 | PASS |
| torus inner edge radius | 6.02173 | 6 | 1.25 | PASS |
| weak-loop beta pmax/B2max | 100 | 100 | 35 | PASS |
| initial divB L1 | 2.01073e-10 | 0 | 0.005 | PASS |
| initial fail fraction | 0 | 0 | 1e-05 | PASS |
| evolved fail fraction | 0 | 0 | 0.05 | PASS |
| evolved CFL | 0.0304969 | 0 | 1.25 | PASS |

## Diagnostics


### initial

| Metric | Value |
|---|---:|
| mass | 1.21352e+08 |
| mdot | 0 |
| phiBH | 0 |
| Ldot / Mdot | 0 |
| Edot / Mdot | 0 |
| beta min | 0.000962118 |
| beta mean | 208622 |
| sigma max | 28.9198 |
| divB L1 | 2.01073e-10 |
| divB max | 9.68575e-08 |
| floor mass fraction | 3.052e-08 |
| fail fraction | 0 |
| max Lorentz | 31.623 |
| CFL | 0.0304897 |
| Qtheta | 4.4477 |
| Qphi | 4.0804 |
| Q product | 18.1484 |

### evolved

| Metric | Value |
|---|---:|
| mass | 6.70421e+07 |
| mdot | 1.46976e-05 |
| phiBH | 0 |
| Ldot / Mdot | 0.879994 |
| Edot / Mdot | 0.442984 |
| beta min | 0.000940428 |
| beta mean | 444567 |
| sigma max | 0.0350198 |
| divB L1 | 2.21267e-06 |
| divB max | 0.00182113 |
| floor mass fraction | 2.99432e-08 |
| fail fraction | 0 |
| max Lorentz | 31.623 |
| CFL | 0.0304969 |
| Qtheta | 1.63807 |
| Qphi | 1.54356 |
| Q product | 2.52847 |

## Table-2-Style Time Series

| Quantity | Peak | Mean | Stddev |
|---|---:|---:|---:|
| Mdot | 1.46976e-05 | 1.46976e-05 | 7.27596e-12 |
| PhiBH/sqrt(Mdot) | 0 | 0 | 0 |
| Ldot/Mdot | 0.879994 | 0.879994 | 4.76837e-07 |
| Edot/Mdot | 0.442984 | 0.442984 | 3.27826e-07 |
| divB L1 | 2.21267e-06 | 1.05311e-06 | 6.67922e-07 |
| fail fraction | 0 | 0 | 0 |
| Qtheta | 4.41306 | 2.64672 | 0.864325 |
| Qphi | 4.04787 | 2.42958 | 0.777522 |

Overall: **PASS**
