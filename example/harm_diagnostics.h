#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "harm_config.h"
#include "harm_grid.h"
#include "harm_kerr_schild.h"
#include "harm_types.h"

namespace harm {

class DiagnosticsSampler {
public:
    static Diagnostics compute(const Config& cfg, const std::vector<float>& packed, bool gpuLive) {
        Diagnostics out{};
        out.gpuLive = gpuLive;
        if (packed.size() < packedFloatCount(cfg.cellCount())) {
            return out;
        }

        const float logRange = std::max(std::log(cfg.rout) - std::log(cfg.rin), 1e-6f);
        const float dtheta = 0.84f * kPi / static_cast<float>(std::max(cfg.thetaN, 1));
        const float dphi = 2.0f * kPi / static_cast<float>(std::max(cfg.phiN, 1));
        const int fluxIr = std::min(2, cfg.radialN - 1);
        float betaSum = 0.0f;
        float betaMin = std::numeric_limits<float>::max();
        int betaCount = 0;
        float fluxBH = 0.0f;
        float floorMass = 0.0f;
        float failCells = 0.0f;
        float divBVolume = 0.0f;
        float qThetaSum = 0.0f;
        float qPhiSum = 0.0f;
        float qWeight = 0.0f;

        auto at = [&](int ir, int it, int ip, int comp) -> float {
            int p = ip % cfg.phiN;
            if (p < 0) p += cfg.phiN;
            const int r = std::clamp(ir, 0, cfg.radialN - 1);
            const int t = std::clamp(it, 0, cfg.thetaN - 1);
            const size_t idx = (static_cast<size_t>(p) * static_cast<size_t>(cfg.thetaN) + static_cast<size_t>(t))
                * static_cast<size_t>(cfg.radialN) + static_cast<size_t>(r);
            return packed[idx * 12 + static_cast<size_t>(comp)];
        };

        for (int ip = 0; ip < cfg.phiN; ++ip) {
            for (int it = 0; it < cfg.thetaN; ++it) {
                const float y = (static_cast<float>(it) + 0.5f) / static_cast<float>(cfg.thetaN);
                const float theta = 0.08f * kPi + y * 0.84f * kPi;
                const float sinTh = std::max(std::sin(theta), 0.08f);
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const float x = (static_cast<float>(ir) + 0.5f) / static_cast<float>(cfg.radialN);
                    const float r = std::exp(std::log(cfg.rin) + x * logRange);
                    const float dr = std::max(r * logRange / static_cast<float>(std::max(cfg.radialN, 1)), 1e-4f);
                    const Metric metric = KerrSchild::metric(r, theta, cfg.spin);
                    const float volume = metric.sqrtMinusG * dr * dtheta * dphi;
                    const size_t base = ((static_cast<size_t>(ip) * static_cast<size_t>(cfg.thetaN) + static_cast<size_t>(it))
                        * static_cast<size_t>(cfg.radialN) + static_cast<size_t>(ir)) * 12;

                    const float rho = std::max(packed[base + 0], cfg.rhoFloor);
                    const float uu = std::max(packed[base + 1], cfg.uFloor);
                    const float vr = packed[base + 2];
                    const float vth = packed[base + 3];
                    const float vph = packed[base + 4];
                    const float br = packed[base + 5];
                    const float bth = packed[base + 6];
                    const float bph = packed[base + 7];
                    const float b2 = br * br + bth * bth + bph * bph;
                    const float pressure = uu / 3.0f;
                    const float beta = pressure / std::max(0.5f * b2, 1e-12f);
                    const float v2 = vr * vr + vth * vth + vph * vph;
                    const float gamma = 1.0f / std::sqrt(std::max(1.0f - std::min(v2, 0.999f), 1e-4f));
                    const float cf = std::sqrt(std::clamp((4.0f / 3.0f * pressure + b2) /
                        std::max(rho + 4.0f * uu / 3.0f + b2, 1e-8f), 0.0f, 0.92f));
                    const float cell = std::min(dr, std::min(r * dtheta, r * sinTh * dphi));

                    out.mass += rho * volume;
                    out.internalEnergy += uu * volume;
                    out.magneticEnergy += 0.5f * b2 * volume;
                    out.angularMomentum += rho * r * sinTh * vph * volume;
                    out.sigmaMax = std::max(out.sigmaMax, b2 / std::max(rho, cfg.rhoFloor));
                    out.maxLorentz = std::max(out.maxLorentz, gamma);
                    out.cfl = std::max(out.cfl, cfg.dt * (std::sqrt(std::max(v2, 0.0f)) + cf) / std::max(cell, 1e-5f));
                    const float omega = std::abs(vph / std::max(r * sinTh, 1.0e-5f));
                    if (rho > 8.0f * cfg.rhoFloor && omega > 1.0e-5f && b2 > 1.0e-12f) {
                        const float vA = std::sqrt(b2 / std::max(rho + 4.0f * uu / 3.0f + b2, 1.0e-8f));
                        const float lambdaMri = 2.0f * kPi * vA / omega;
                        const float qTheta = lambdaMri / std::max(r * dtheta, 1.0e-6f);
                        const float qPhi = lambdaMri / std::max(r * sinTh * dphi, 1.0e-6f);
                        const float weight = rho * volume;
                        qThetaSum += qTheta * weight;
                        qPhiSum += qPhi * weight;
                        qWeight += weight;
                    }
                    if (rho <= 1.01f * cfg.rhoFloor || uu <= 1.01f * cfg.uFloor) {
                        floorMass += rho * volume;
                    }
                    if (packed[base + 8] > 0.5f) {
                        failCells += 1.0f;
                    }
                    if (rho > 8.0f * cfg.rhoFloor) {
                        betaMin = std::min(betaMin, beta);
                        betaSum += beta;
                        ++betaCount;
                    }
                    if (ir == fluxIr) {
                        const float area = r * r * sinTh * dtheta * dphi;
                        const float inflow = std::max(-rho * vr, 0.0f);
                        const float specificL = r * sinTh * vph;
                        const float b2Flux = 0.5f * b2;
                        const float specificE = uu / std::max(rho, cfg.rhoFloor) + 0.5f * v2 + b2Flux / std::max(rho, cfg.rhoFloor);
                        fluxBH += std::abs(br) * area;
                        out.mdot += inflow * area;
                        out.ldot += inflow * specificL * area;
                        out.edot += inflow * specificE * area;
                    }
                }
            }
        }

        for (int ip = 0; ip < cfg.phiN; ++ip) {
            for (int it = 0; it < cfg.thetaN; ++it) {
                const float y = (static_cast<float>(it) + 0.5f) / static_cast<float>(cfg.thetaN);
                const float theta = 0.08f * kPi + y * 0.84f * kPi;
                const float sinTh = std::max(std::sin(theta), 0.08f);
                const float thm = std::max(theta - dtheta, 0.02f);
                const float thp = std::min(theta + dtheta, kPi - 0.02f);
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const float x = (static_cast<float>(ir) + 0.5f) / static_cast<float>(cfg.radialN);
                    const float r = std::exp(std::log(cfg.rin) + x * logRange);
                    const float dr = std::max(r * logRange / static_cast<float>(std::max(cfg.radialN, 1)), 1e-4f);
                    const Metric metric = KerrSchild::metric(r, theta, cfg.spin);
                    const float volume = metric.sqrtMinusG * dr * dtheta * dphi;
                    const int irm = std::max(ir - 1, 0);
                    const int irp = std::min(ir + 1, cfg.radialN - 1);
                    const int itm = std::max(it - 1, 0);
                    const int itp = std::min(it + 1, cfg.thetaN - 1);
                    const int ipm = (ip + cfg.phiN - 1) % cfg.phiN;
                    const int ipp = (ip + 1) % cfg.phiN;
                    const float rm = std::exp(std::log(cfg.rin) + ((static_cast<float>(irm) + 0.5f) / static_cast<float>(cfg.radialN)) * logRange);
                    const float rp = std::exp(std::log(cfg.rin) + ((static_cast<float>(irp) + 0.5f) / static_cast<float>(cfg.radialN)) * logRange);
                    const float radial = (rp * rp * at(irp, it, ip, 5) - rm * rm * at(irm, it, ip, 5)) / std::max(rp - rm, 1e-4f);
                    const float polar = (std::sin(thp) * at(ir, itp, ip, 6) - std::sin(thm) * at(ir, itm, ip, 6)) /
                        std::max((itp - itm) * dtheta, 1e-4f);
                    const float azimuth = (at(ir, it, ipp, 7) - at(ir, it, ipm, 7)) / std::max(2.0f * dphi, 1e-4f);
                    const float divB = radial / std::max(r * r, 1e-4f) + polar / std::max(r * sinTh, 1e-4f) + azimuth / std::max(r * sinTh, 1e-4f);
                    out.divBL1 += std::abs(divB) * volume;
                    out.divBMax = std::max(out.divBMax, std::abs(divB));
                    divBVolume += volume;
                }
            }
        }

        out.betaMin = (betaCount > 0) ? betaMin : 0.0f;
        out.betaMean = (betaCount > 0) ? betaSum / static_cast<float>(betaCount) : 0.0f;
        out.phiBH = 0.5f * fluxBH / std::sqrt(std::max(out.mdot, 1e-10f));
        out.divBL1 = out.divBL1 / std::max(divBVolume, 1e-10f);
        out.floorMassFrac = floorMass / std::max(out.mass, 1e-10f);
        out.failFrac = failCells / std::max(static_cast<float>(cfg.cellCount()), 1.0f);
        if (qWeight > 0.0f) {
            out.qTheta = qThetaSum / qWeight;
            out.qPhi = qPhiSum / qWeight;
            out.qProduct = out.qTheta * out.qPhi;
        }
        return out;
    }
};

} // namespace harm
