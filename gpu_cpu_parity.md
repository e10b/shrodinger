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
| rho L1 relative | 7.27621e-05 |
| rho L2 relative | 0.000739695 |
| rho Linf relative | 0.0129535 |
| internal energy L1 relative | 1.75796e-05 |
| velocity L1 absolute | 3.86552e-06 |
| magnetic magnitude L1 relative | 0.00641997 |

## CPU/GPU Diagnostic Deltas

| Metric | Relative/absolute delta |
|---|---:|
| mass relative | 0 |
| internal energy relative | 9.92989e-07 |
| magnetic energy relative | 0 |
| mdot relative | 1.93423e-06 |
| divB L1 relative | 0 |
| fail fraction absolute | 0 |
| floor mass fraction absolute | 2.79356e-09 |
| Qtheta relative | 6.78511e-08 |
| Qphi relative | 0 |

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
| entropy fallback fraction | 0 |
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
| fail fraction | 0 |
| entropy fallback fraction | 0.413147 |
| floor mass fraction | 6.22783e-08 |
| Qtheta | 0.219616 |
| Qphi | 0.228381 |

### GPU

| Metric | Value |
|---|---:|
| mass | 83889.3 |
| internal energy | 1721.05 |
| magnetic energy | 0.0680775 |
| mdot | 0.00069215 |
| beta min | 100 |
| beta mean | 1.08853e+08 |
| divB L1 | 2.49658e-05 |
| divB max | 0.00289546 |
| fail fraction | 0 |
| entropy fallback fraction | 0.218628 |
| floor mass fraction | 5.94848e-08 |
| Qtheta | 0.219616 |
| Qphi | 0.228381 |

Overall: **PASS**
