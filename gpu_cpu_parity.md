# GPU/CPU HARM Parity Report

| Setting | Value |
|---|---:|
| grid | 32 x 32 x 32 |
| frames | 3 |
| high-order | enabled |
| dt ceiling | 0.0005 |

## CPU/GPU State Norms

| Norm | Value |
|---|---:|
| rho L1 relative | 6.30692e-06 |
| rho L2 relative | 6.76209e-05 |
| rho Linf relative | 0.00137779 |
| internal energy L1 relative | 8.5787e-05 |
| velocity L1 absolute | 4.20277e-05 |
| magnetic magnitude L1 relative | 0.00431342 |

## CPU/GPU Diagnostic Deltas

| Metric | Relative/absolute delta |
|---|---:|
| mass relative | 0 |
| internal energy relative | 1.44837e-07 |
| magnetic energy relative | 0 |
| mdot relative | 0.000249952 |
| divB L1 relative | 0 |
| fail fraction absolute | 0.00012207 |
| floor mass fraction absolute | 1.35003e-13 |
| Qtheta relative | 1.96145e-07 |
| Qphi relative | 1.2564e-07 |

### initial

| Metric | Value |
|---|---:|
| mass | 82207.6 |
| internal energy | 1685.63 |
| magnetic energy | 0.0657259 |
| mdot | 0 |
| beta min | 100 |
| beta mean | 1.37667e+08 |
| divB L1 | 2.72251e-05 |
| divB max | 0.00432307 |
| fail fraction | 0 |
| floor mass fraction | 6.47193e-08 |
| Qtheta | 0.455821 |
| Qphi | 0.237204 |

### CPU

| Metric | Value |
|---|---:|
| mass | 82106.2 |
| internal energy | 1685.63 |
| magnetic energy | 0.0657259 |
| mdot | 0.000691873 |
| beta min | 99.9999 |
| beta mean | 1.37668e+08 |
| divB L1 | 2.72251e-05 |
| divB max | 0.00432307 |
| fail fraction | 0.00012207 |
| floor mass fraction | 6.47994e-08 |
| Qtheta | 0.455821 |
| Qphi | 0.237204 |

### GPU

| Metric | Value |
|---|---:|
| mass | 82106.2 |
| internal energy | 1685.63 |
| magnetic energy | 0.0657259 |
| mdot | 0.0006917 |
| beta min | 99.9996 |
| beta mean | 1.37667e+08 |
| divB L1 | 2.72251e-05 |
| divB max | 0.00432307 |
| fail fraction | 0 |
| floor mass fraction | 6.47996e-08 |
| Qtheta | 0.455821 |
| Qphi | 0.237204 |

Overall: **PASS**
