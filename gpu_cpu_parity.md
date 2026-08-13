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
| rho L1 relative | 0.000283118 |
| rho L2 relative | 0.00408415 |
| rho Linf relative | 0.0652866 |
| internal energy L1 relative | 8.69756e-06 |
| velocity L1 absolute | 3.74842e-06 |
| magnetic magnitude L1 relative | 0.0139705 |

## CPU/GPU Diagnostic Deltas

| Metric | Relative/absolute delta |
|---|---:|
| mass relative | 0 |
| internal energy relative | 9.22062e-07 |
| magnetic energy relative | 1.39448e-07 |
| mdot relative | 1.93423e-06 |
| divB L1 relative | 0.999999 |
| fail fraction absolute | 0 |
| floor mass fraction absolute | 2.80387e-09 |
| Qtheta relative | 2.58935e-07 |
| Qphi relative | 1.21829e-07 |

### initial

| Metric | Value |
|---|---:|
| mass | 84000.6 |
| internal energy | 1721.05 |
| magnetic energy | 0.000834829 |
| mdot | 0 |
| beta min | 1100.14 |
| beta mean | 1.21818e+08 |
| divB L1 | 4.59658e-07 |
| divB max | 9.7967e-05 |
| fail fraction | 0 |
| entropy fallback fraction | 0 |
| resolved entropy fallback fraction | 0 |
| floor mass fraction | 6.22716e-08 |
| Qtheta | 0.014387 |
| Qphi | 0.015289 |

### CPU

| Metric | Value |
|---|---:|
| mass | 83889.3 |
| internal energy | 1721.05 |
| magnetic energy | 0.000834829 |
| mdot | 0.000692148 |
| beta min | 1100.11 |
| beta mean | 1.21819e+08 |
| divB L1 | 4.59658e-07 |
| divB max | 9.7967e-05 |
| fail fraction | 0 |
| entropy fallback fraction | 0.414307 |
| resolved entropy fallback fraction | 0.010617 |
| floor mass fraction | 6.22886e-08 |
| Qtheta | 0.014387 |
| Qphi | 0.015289 |

### GPU

| Metric | Value |
|---|---:|
| mass | 83889.3 |
| internal energy | 1721.05 |
| magnetic energy | 0.000834829 |
| mdot | 0.00069215 |
| beta min | 1100.14 |
| beta mean | 1.21819e+08 |
| divB L1 | 2.58587e-13 |
| divB max | 6.03998e-11 |
| fail fraction | 0 |
| entropy fallback fraction | 0.218628 |
| resolved entropy fallback fraction | 0.0252404 |
| floor mass fraction | 5.94848e-08 |
| Qtheta | 0.014387 |
| Qphi | 0.015289 |

Overall: **PASS**
