#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "harm_config.h"
#include "harm_diagnostics.h"
#include "harm_grid.h"
#include "harm_kerr_schild.h"
#include "harm_state.h"
#include "harm_types.h"

namespace harm {

class InitialDataBuilder {
public:
    static Diagnostics build(const Config& cfg, Grid& grid) {
        if (cfg.initialData == 2) {
            return buildFishboneMoncrief(cfg, grid);
        }
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
                const float theta = HarmGeometry::thetaAt(cfg, it);
                const float sinTh = std::max(std::sin(theta), 1.0e-4f);
                const float vertical = std::exp(-std::pow((theta - 0.5f * kPi) / 0.34f, 2.0f));
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const float x = (static_cast<float>(ir) + 0.5f) / static_cast<float>(cfg.radialN);
                    const float r = std::exp(std::log(rin) + x * logRange);
                    const float torus = std::exp(-((r - r0) * (r - r0)) / std::max(2.0f * sigmaR * sigmaR, 1e-6f));
                    const float arm = std::sin(2.0f * phi - 3.6f * std::log(std::max(r, 1.0f)));
                    const float cellNoise = deterministicNoise(ir, it, ip);
                    const float knot = std::sin(7.0f * phi + 9.0f * x + 2.3f * std::sin(3.0f * phi));
                    const float perturb = mad
                        ? std::clamp(1.0f + 0.18f * arm + 0.13f * knot + 0.16f * cellNoise, 0.52f, 1.52f)
                        : 1.0f + 0.045f * arm + 0.025f * std::sin(5.0f * phi + 1.7f * x);
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
                    c[2] = vr + (mad ? 0.028f * torus * cellNoise : 0.0f);
                    c[3] = 0.015f * std::sin(theta * 2.0f) * torus
                        + (mad ? 0.012f * torus * std::sin(5.0f * phi + 11.0f * x) : 0.0f);
                    c[4] = omegaK;
                    c[9] = 0.0f;
                    vectorPotential[idx] = magneticScale * std::max(rho - 1.4f * atmosphere, 0.0f) * r * sinTh;
                    pressureMax = std::max(pressureMax, pressure);
                }
            }
        }

        const float dtheta = HarmGeometry::dtheta(cfg);
        const float dphi = 2.0f * kPi / static_cast<float>(std::max(cfg.phiN, 1));
        for (int ip = 0; ip < cfg.phiN; ++ip) {
            const float phi = (static_cast<float>(ip) + 0.5f) * dphi;
            for (int it = 0; it < cfg.thetaN; ++it) {
                const float theta = HarmGeometry::thetaAt(cfg, it);
                const float sinTh = std::max(std::sin(theta), 1.0e-4f);
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

        initializeEntropy(cfg, grid);
        grid.initializeFaceFlux(cfg);
        return DiagnosticsSampler::compute(cfg, grid.packed, false);
    }

private:
    static Diagnostics buildFishboneMoncrief(const Config& cfg, Grid& grid) {
        grid.resize(cfg);
        const float rinGrid = HarmGeometry::rin(cfg);
        const float rout = std::max(cfg.rout, 50.0f);
        const float logRange = std::max(std::log(rout) - std::log(rinGrid), 1e-6f);
        constexpr float torusInner = 6.0f;
        constexpr float densityMaxRadius = 12.0f;
        constexpr float kappa = 1.0e-3f;
        constexpr float perturbAmp = 0.04f;
        constexpr float betaTarget = 100.0f;

        const float l = keplerianAngularMomentum(densityMaxRadius, cfg.spin);
        const float wIn = potential(torusInner, 0.5f * kPi, cfg.spin, l);
        std::vector<float> vectorPotential(cfg.cellCount(), 0.0f);
        float rhoMax = 0.0f;
        float pressureMax = 0.0f;

        for (int ip = 0; ip < cfg.phiN; ++ip) {
            const float phi = (static_cast<float>(ip) + 0.5f) * (2.0f * kPi / static_cast<float>(cfg.phiN));
            for (int it = 0; it < cfg.thetaN; ++it) {
                const float theta = thetaAt(cfg, it);
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const float r = radiusAt(cfg, ir, rinGrid, logRange);
                    const float w = potential(r, theta, cfg.spin, l);
                    const float rhoFloor = cfg.rhoFloorAt(r);
                    const float uFloor = cfg.uFloorAt(r);
                    float rho = rhoFloor;
                    float pressure = uFloor * (kAdiabaticGamma - 1.0f);
                    if (std::isfinite(w) && w < wIn && r >= torusInner) {
                        const float h = std::exp(wIn - w);
                        const float eps = std::max(h - 1.0f, 0.0f);
                        rho = std::pow(eps * (kAdiabaticGamma - 1.0f) / (kappa * kAdiabaticGamma), 1.0f / (kAdiabaticGamma - 1.0f));
                        pressure = kappa * std::pow(rho, kAdiabaticGamma);
                    }

                    const float omega = angularVelocity(r, theta, cfg.spin, l);
                    const float sinTh = std::max(std::sin(theta), 1.0e-4f);
                    const float vphi = std::clamp(r * sinTh * omega, -0.88f, 0.88f);
                    const size_t idx = grid.index(cfg, ir, it, ip);
                    float* c = grid.cell(idx);
                    c[0] = rho;
                    c[1] = std::max(pressure / (kAdiabaticGamma - 1.0f), uFloor);
                    c[2] = 0.0f;
                    c[3] = 0.0f;
                    c[4] = omega;
                    c[9] = 0.0f;
                    vectorPotential[idx] = rho;
                    rhoMax = std::max(rhoMax, rho);
                    (void)phi;
                }
            }
        }

        // The comparison normalizes rho_max to unity, then perturbs pressure
        // (not density) by four per cent to seed the MRI.
        const float densityScale = 1.0f / std::max(rhoMax, 1.0e-12f);
        for (int ip = 0; ip < cfg.phiN; ++ip) {
            for (int it = 0; it < cfg.thetaN; ++it) {
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const float r = radiusAt(cfg, ir, rinGrid, logRange);
                    const size_t idx = grid.index(cfg, ir, it, ip);
                    float* c = grid.cell(idx);
                    const bool inTorus = vectorPotential[idx] > 8.0f * cfg.rhoFloorAt(r);
                    c[0] = std::max(c[0] * densityScale, cfg.rhoFloorAt(r));
                    c[1] = std::max(c[1] * densityScale, cfg.uFloorAt(r));
                    if (inTorus) {
                        c[1] *= 1.0f + perturbAmp * deterministicNoise(ir, it, ip);
                    }
                    pressureMax = std::max(pressureMax, (kAdiabaticGamma - 1.0f) * c[1]);
                    vectorPotential[idx] = std::max(c[0] - 0.2f, 0.0f);
                }
            }
        }
        curlVectorPotential(cfg, grid, vectorPotential, rinGrid, logRange);

        float betaMinUnscaled = std::numeric_limits<float>::max();
        for (int ip = 0; ip < cfg.phiN; ++ip) {
            for (int it = 0; it < cfg.thetaN; ++it) {
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const CellGeometry geom = HarmGeometry::cell(cfg, ir, it);
                    const float* c = grid.cell(grid.index(cfg, ir, it, ip));
                    const Primitive p = HarmState::fromPacked(c, geom.r, geom.theta);
                    const FourVector b = HarmState::magneticFourVector(p, geom.metric);
                    const float b2 = std::max(KerrSchild::dot(geom.metric, b, b), 0.0f);
                    if (b2 > 1.0e-20f && p.rho > 0.2f) {
                        const float pressure = (kAdiabaticGamma - 1.0f) * p.u;
                        betaMinUnscaled = std::min(betaMinUnscaled, 2.0f * pressure / b2);
                    }
                }
            }
        }
        // beta = p_gas / p_mag = 2 p_gas / b^2.  Normalize using the
        // cellwise minimum, as specified by the comparison setup.
        const float scale = std::sqrt(std::max(betaMinUnscaled, 1.0e-20f) / betaTarget);
        for (size_t i = 0; i < cfg.cellCount(); ++i) {
            float* c = grid.cell(i);
            c[5] *= scale;
            c[6] *= scale;
            c[7] *= scale;
        }

        // Preserve the vector-potential topology on the production staggered
        // faces instead of averaging the differentiated cell field. `scale`
        // is the beta=100 magnetic normalization applied above.
        grid.initializeFaceFluxFromAphi(cfg, vectorPotential, scale);
        initializeEntropy(cfg, grid);
        return DiagnosticsSampler::compute(cfg, grid.packed, false);
    }

    static void initializeEntropy(const Config& cfg, Grid& grid) {
        for (int ip = 0; ip < cfg.phiN; ++ip) {
            for (int it = 0; it < cfg.thetaN; ++it) {
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const CellGeometry geom = HarmGeometry::cell(cfg, ir, it);
                    float* c = grid.cell(grid.index(cfg, ir, it, ip));
                    const Primitive p = HarmState::fromPacked(c, geom.r, geom.theta);
                    c[8] = HarmState::conservedEntropy(p, geom.metric);
                    c[9] = 0.0f;
                }
            }
        }
    }

    static float radiusAt(const Config& cfg, int ir, float rinGrid, float logRange) {
        const float x = (static_cast<float>(ir) + 0.5f) / static_cast<float>(std::max(cfg.radialN, 1));
        return std::exp(std::log(rinGrid) + x * logRange);
    }

    static float thetaAt(const Config& cfg, int it) {
        const float y = (static_cast<float>(it) + 0.5f) / static_cast<float>(std::max(cfg.thetaN, 1));
        return y * kPi;
    }

    static float deterministicNoise(int ir, int it, int ip) {
        const float x = std::sin(12.9898f * static_cast<float>(ir + 1) +
                                 78.233f * static_cast<float>(it + 3) +
                                 37.719f * static_cast<float>(ip + 7)) * 43758.5453f;
        return 2.0f * (x - std::floor(x)) - 1.0f;
    }

    static float keplerianAngularMomentum(float r, float spin) {
        const Metric m = KerrSchild::metric(r, 0.5f * kPi, spin);
        const float omega = 1.0f / (std::pow(std::max(r, 1.0f), 1.5f) + spin);
        const float denom = std::sqrt(std::max(-(m.gcov[0][0] + 2.0f * omega * m.gcov[0][3] + omega * omega * m.gcov[3][3]), 1.0e-10f));
        const float ut = 1.0f / denom;
        const float uLowerT = ut * (m.gcov[0][0] + omega * m.gcov[0][3]);
        const float uLowerPhi = ut * (m.gcov[0][3] + omega * m.gcov[3][3]);
        return -uLowerPhi / std::min(uLowerT, -1.0e-8f);
    }

    static float angularVelocity(float r, float theta, float spin, float l) {
        const Metric m = KerrSchild::metric(r, theta, spin);
        return -(m.gcov[0][3] + l * m.gcov[0][0]) / std::max(m.gcov[3][3] + l * m.gcov[0][3], 1.0e-8f);
    }

    static float potential(float r, float theta, float spin, float l) {
        const Metric m = KerrSchild::metric(r, theta, spin);
        const float denom = m.gcov[3][3] + 2.0f * l * m.gcov[0][3] + l * l * m.gcov[0][0];
        const float numer = m.gcov[0][3] * m.gcov[0][3] - m.gcov[0][0] * m.gcov[3][3];
        if (denom <= 0.0f || numer <= 0.0f) {
            return std::numeric_limits<float>::infinity();
        }
        return std::log(std::sqrt(numer / denom));
    }

    static void curlVectorPotential(const Config& cfg, Grid& grid, const std::vector<float>& aphi, float rinGrid, float logRange) {
        const float dtheta = kPi / static_cast<float>(std::max(cfg.thetaN, 1));
        for (int ip = 0; ip < cfg.phiN; ++ip) {
            for (int it = 0; it < cfg.thetaN; ++it) {
                const float theta = thetaAt(cfg, it);
                const float sinTh = std::max(std::sin(theta), 1.0e-4f);
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const float r = radiusAt(cfg, ir, rinGrid, logRange);
                    const int irm = std::max(ir - 1, 0);
                    const int irp = std::min(ir + 1, cfg.radialN - 1);
                    const int itm = std::max(it - 1, 0);
                    const int itp = std::min(it + 1, cfg.thetaN - 1);
                    const float rm = radiusAt(cfg, irm, rinGrid, logRange);
                    const float rp = radiusAt(cfg, irp, rinGrid, logRange);
                    const float dA_dtheta = (aphi[grid.index(cfg, ir, itp, ip)] - aphi[grid.index(cfg, ir, itm, ip)])
                        / std::max((itp - itm) * dtheta, 1.0e-4f);
                    const float dA_dr = (aphi[grid.index(cfg, irp, it, ip)] - aphi[grid.index(cfg, irm, it, ip)])
                        / std::max(rp - rm, 1.0e-4f);
                    float* c = grid.cell(grid.index(cfg, ir, it, ip));
                    c[5] = dA_dtheta / std::max(r * r * sinTh, 1.0e-4f);
                    c[6] = -dA_dr / std::max(r * sinTh, 1.0e-4f);
                    c[7] = 0.0f;
                }
            }
        }
    }
};

} // namespace harm
