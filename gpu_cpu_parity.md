# GPU/CPU HARM Parity Report

| Setting | Value |
|---|---:|
| grid | 32 x 16 x 32 |
| frames | 1 |
| high-order | enabled |
| dt ceiling | 0.0005 |

## CPU/GPU State Norms

| Norm | Value |
|---|---:|
| rho L1 relative | 5.03434e-05 |
| rho L2 relative | 0.000608004 |
| rho Linf relative | 0.0135116 |
| internal energy L1 relative | 4.70277e-05 |
| velocity L1 absolute | 1.89003e-05 |
| magnetic magnitude L1 relative | 0.00634902 |

## CPU/GPU Diagnostic Deltas

| Metric | Relative/absolute delta |
|---|---:|
| mass relative | 9.31287e-08 |
| internal energy relative | 2.12783e-07 |
| magnetic energy relative | 0 |
| mdot relative | 0.000149188 |
| divB L1 relative | 0 |
| fail fraction absolute | 0.00152588 |
| floor mass fraction absolute | 2.87292e-09 |
| Qtheta relative | 2.03553e-07 |
| Qphi relative | 3.91481e-07 |

### initial

| Metric | Value |
|---|---:|
| mass | 84000.6 |
| internal energy | 1721.05 |
| magnetic energy | 0.0680775 |
| mdot | 0 |
| beta min | 100 |
| beta mean | 1.08853e+08 |
| divB L1 | 2.49658e-05 |
| divB max | 0.00289546 |
| fail fraction | 0 |
| floor mass fraction | 6.22716e-08 |
| Qtheta | 0.219616 |
| Qphi | 0.228381 |

### CPU

| Metric | Value |
|---|---:|
| mass | 83889.3 |
| internal energy | 1721.05 |
| magnetic energy | 0.0680775 |
| mdot | 0.000692148 |
| beta min | 99.9991 |
| beta mean | 1.08853e+08 |
| divB L1 | 2.49658e-05 |
| divB max | 0.00289546 |
| fail fraction | 0.00335693 |
| floor mass fraction | 6.23573e-08 |
| Qtheta | 0.219616 |
| Qphi | 0.228381 |

### GPU

| Metric | Value |
|---|---:|
| mass | 83889.3 |
| internal energy | 1721.05 |
| magnetic energy | 0.0680775 |
| mdot | 0.000692045 |
| beta min | 99.9997 |
| beta mean | 1.08853e+08 |
| divB L1 | 2.49658e-05 |
| divB max | 0.00289546 |
| fail fraction | 0.00488281 |
| floor mass fraction | 5.94844e-08 |
| Qtheta | 0.219616 |
| Qphi | 0.228381 |

Overall: **PASS**
