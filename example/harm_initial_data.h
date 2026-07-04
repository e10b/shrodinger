#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "harm_config.h"
#include "harm_diagnostics.h"
#include "harm_grid.h"
#include "harm_types.h"

namespace harm {

class InitialDataBuilder {
public:
    static Diagnostics build(const Config& cfg, Grid& grid) {
        grid.resize(cfg);
        const bool mad = (cfg.initialData == 1);
        const float rin = std::max(cfg.rin, 1.05f);
        const float rout = std::max(cfg.rout, rin + 4.0f);
        const float logRange = std::max(std::log(rout) - std::log(rin), 1e-6f);
        const float r0 = 0.34f * rout;
        const float sigmaR = 0.12f * rout;
        const float magneticScale = std::clamp(cfg.magneticLoop / 0.055f, 0.15f, 4.0f);

        std::vector<float> vectorPotential(cfg.cellCount(), 0.0f);
        float pressureMax = cfg.uFloor / 3.0f;

        for (int ip = 0; ip < cfg.phiN; ++ip) {
            const float phi = (static_cast<float>(ip) + 0.5f) * (2.0f * kPi / static_cast<float>(cfg.phiN));
            for (int it = 0; it < cfg.thetaN; ++it) {
                const float y = (static_cast<float>(it) + 0.5f) / static_cast<float>(cfg.thetaN);
                const float theta = 0.08f * kPi + y * 0.84f * kPi;
                const float sinTh = std::max(std::sin(theta), 0.08f);
                const float vertical = std::exp(-std::pow((theta - 0.5f * kPi) / 0.34f, 2.0f));
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const float x = (static_cast<float>(ir) + 0.5f) / static_cast<float>(cfg.radialN);
                    const float r = std::exp(std::log(rin) + x * logRange);
                    const float torus = std::exp(-((r - r0) * (r - r0)) / std::max(2.0f * sigmaR * sigmaR, 1e-6f));
                    const float arm = std::sin(2.0f * phi - 3.6f * std::log(std::max(r, 1.0f)));
                    const float perturb = 1.0f + 0.045f * arm + 0.025f * std::sin(5.0f * phi + 1.7f * x);
                    const float atmosphere = 1e-4f * std::pow(std::max(r / rin, 1.0f), -1.5f);
                    const float rho = std::max((mad ? 1.20f : 1.0f) * 0.24f * torus * vertical * perturb + atmosphere, cfg.rhoFloor);
                    const float pressure = 0.035f * std::pow(std::max(rho - atmosphere, 0.0f), 4.0f / 3.0f) + cfg.uFloor / 3.0f;
                    const float omegaK = 1.0f / (std::pow(std::max(r, 1.0f), 1.5f) + cfg.spin);
                    const float vphi = std::clamp(0.82f * r * sinTh * omegaK, -0.74f, 0.74f);
                    const float vr = -0.012f * std::exp(-std::pow((r - rin) / std::max(0.2f * rout, 1.0f), 2.0f));
                    const size_t idx = grid.index(cfg, ir, it, ip);
                    float* c = grid.cell(idx);
                    c[0] = rho;
                    c[1] = std::max(pressure / (1.0f / 3.0f), cfg.uFloor);
                    c[2] = vr;
                    c[3] = 0.015f * std::sin(theta * 2.0f) * torus;
                    c[4] = vphi / std::max(r * sinTh, 1e-3f);
                    c[8] = 0.0f;
                    vectorPotential[idx] = magneticScale * std::max(rho - 1.4f * atmosphere, 0.0f) * r * sinTh;
                    pressureMax = std::max(pressureMax, pressure);
                }
            }
        }

        const float dtheta = 0.84f * kPi / static_cast<float>(std::max(cfg.thetaN, 1));
        const float dphi = 2.0f * kPi / static_cast<float>(std::max(cfg.phiN, 1));
        for (int ip = 0; ip < cfg.phiN; ++ip) {
            const float phi = (static_cast<float>(ip) + 0.5f) * dphi;
            for (int it = 0; it < cfg.thetaN; ++it) {
                const float y = (static_cast<float>(it) + 0.5f) / static_cast<float>(cfg.thetaN);
                const float theta = 0.08f * kPi + y * 0.84f * kPi;
                const float sinTh = std::max(std::sin(theta), 0.08f);
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const float x = (static_cast<float>(ir) + 0.5f) / static_cast<float>(cfg.radialN);
                    const float r = std::exp(std::log(rin) + x * logRange);
                    const int irm = std::max(ir - 1, 0);
                    const int irp = std::min(ir + 1, cfg.radialN - 1);
                    const int itm = std::max(it - 1, 0);
                    const int itp = std::min(it + 1, cfg.thetaN - 1);
                    const float rm = std::exp(std::log(rin) + ((static_cast<float>(irm) + 0.5f) / static_cast<float>(cfg.radialN)) * logRange);
                    const float rp = std::exp(std::log(rin) + ((static_cast<float>(irp) + 0.5f) / static_cast<float>(cfg.radialN)) * logRange);
                    const size_t idx = grid.index(cfg, ir, it, ip);
                    const float dA_dtheta = (vectorPotential[grid.index(cfg, ir, itp, ip)] - vectorPotential[grid.index(cfg, ir, itm, ip)])
                        / std::max((itp - itm) * dtheta, 1e-4f);
                    const float dA_dr = (vectorPotential[grid.index(cfg, irp, it, ip)] - vectorPotential[grid.index(cfg, irm, it, ip)])
                        / std::max(rp - rm, 1e-4f);
                    float* c = grid.cell(idx);
                    c[5] = dA_dtheta / std::max(r * r * sinTh, 1e-4f);
                    c[6] = -dA_dr / std::max(r * sinTh, 1e-4f);
                    const float pressure = std::max(c[1] / 3.0f, cfg.uFloor / 3.0f);
                    const float arm = std::sin(2.0f * phi - 3.6f * std::log(std::max(r, 1.0f)));
                    c[7] = magneticScale * (mad ? 0.42f : 0.18f) * std::sqrt(std::max(pressure, cfg.uFloor)) * (1.0f + 0.12f * arm);
                }
            }
        }

        return DiagnosticsSampler::compute(cfg, grid.packed, false);
    }
};

} // namespace harm
