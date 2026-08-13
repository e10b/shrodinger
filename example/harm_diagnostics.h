#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "harm_config.h"
#include "harm_flux.h"
#include "harm_geometry.h"
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

        const float dtheta = HarmGeometry::dtheta(cfg);
        const float dphi = HarmGeometry::dphi(cfg);
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
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const CellGeometry geom = HarmGeometry::cell(cfg, ir, it);
                    const size_t base = ((static_cast<size_t>(ip) * static_cast<size_t>(cfg.thetaN) + static_cast<size_t>(it))
                        * static_cast<size_t>(cfg.radialN) + static_cast<size_t>(ir)) * 12;

                    const Primitive primitive = HarmState::fromPacked(&packed[base], geom.r, geom.theta);
                    const float rhoFloor = cfg.rhoFloorAt(geom.r);
                    const float uFloor = cfg.uFloorAt(geom.r);
                    const float rho = std::max(primitive.rho, rhoFloor);
                    const float uu = std::max(primitive.u, uFloor);
                    const FourVector ucon = HarmState::fourVelocity(primitive, geom.metric);
                    const FourVector bcon = HarmState::magneticFourVector(primitive, geom.metric);
                    const StressEnergy stress = HarmState::stressEnergyContravariant(primitive, geom.metric);
                    const float b2 = std::max(KerrSchild::dot(geom.metric, bcon, bcon), 0.0f);
                    const float pressure = uu / 3.0f;
                    const float beta = pressure / std::max(0.5f * b2, 1e-12f);
                    const float gamma = std::max(geom.metric.alpha * ucon[0], 1.0f);
                    const float cf = std::sqrt(std::clamp((4.0f / 3.0f * pressure + b2) /
                        std::max(rho + 4.0f * uu / 3.0f + b2, 1e-8f), 0.0f, 0.92f));

                    out.mass += rho * ucon[0] * geom.volume;
                    out.internalEnergy += uu * geom.volume;
                    out.magneticEnergy += 0.5f * b2 * geom.volume;
                    out.angularMomentum += stress.T[0][3] * geom.metric.sqrtMinusG * geom.dr * geom.dtheta * geom.dphi;
                    out.sigmaMax = std::max(out.sigmaMax, b2 / std::max(rho, rhoFloor));
                    out.maxLorentz = std::max(out.maxLorentz, gamma);
                    const float coordinateSpeed = std::max({
                        HarmFlux::maxSignalSpeed(primitive, geom.metric, 0) / std::max(geom.dr, 1e-5f),
                        HarmFlux::maxSignalSpeed(primitive, geom.metric, 1) / std::max(geom.dtheta, 1e-5f),
                        HarmFlux::maxSignalSpeed(primitive, geom.metric, 2) / std::max(geom.dphi, 1e-5f)});
                    out.cfl = std::max(out.cfl, cfg.dt * coordinateSpeed);
                    const float omega = std::abs(primitive.v.z);
                    if (rho > 8.0f * rhoFloor && omega > 1.0e-5f && b2 > 1.0e-12f) {
                        const float vA = std::sqrt(b2 / std::max(rho + 4.0f * uu / 3.0f + b2, 1.0e-8f));
                        const float lambdaMri = 2.0f * kPi * vA / omega;
                        const float qTheta = lambdaMri / geom.thetaLength();
                        const float qPhi = lambdaMri / geom.phiLength();
                        const float weight = rho * geom.volume;
                        qThetaSum += qTheta * weight;
                        qPhiSum += qPhi * weight;
                        qWeight += weight;
                    }
                    if (rho <= 1.01f * rhoFloor || uu <= 1.01f * uFloor) {
                        floorMass += rho * geom.volume;
                    }
                    if (packed[base + 8] > 0.5f) {
                        failCells += 1.0f;
                    }
                    if (rho > 8.0f * rhoFloor) {
                        betaMin = std::min(betaMin, beta);
                        betaSum += beta;
                        ++betaCount;
                    }
                    if (ir == fluxIr) {
                        const float area = geom.r * geom.r * geom.sinTheta * dtheta * dphi;
                        float trt = 0.0f;
                        float trp = 0.0f;
                        for (int nu = 0; nu < 4; ++nu) {
                            trt += stress.T[1][nu] * geom.metric.gcov[nu][0];
                            trp += stress.T[1][nu] * geom.metric.gcov[nu][3];
                        }
                        fluxBH += std::abs(primitive.B.x) * area;
                        out.mdot += std::max(-rho * ucon[1], 0.0f) * area;
                        out.ldot += std::max(trp, 0.0f) * area;
                        out.edot += std::max(-trt, 0.0f) * area;
                    }
                }
            }
        }

        for (int ip = 0; ip < cfg.phiN; ++ip) {
            for (int it = 0; it < cfg.thetaN; ++it) {
                const float theta = HarmGeometry::thetaAt(cfg, it);
                const float thm = std::max(theta - dtheta, 0.02f);
                const float thp = std::min(theta + dtheta, kPi - 0.02f);
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const CellGeometry geom = HarmGeometry::cell(cfg, ir, it);
                    const int irm = std::max(ir - 1, 0);
                    const int irp = std::min(ir + 1, cfg.radialN - 1);
                    const int itm = std::max(it - 1, 0);
                    const int itp = std::min(it + 1, cfg.thetaN - 1);
                    const int ipm = (ip + cfg.phiN - 1) % cfg.phiN;
                    const int ipp = (ip + 1) % cfg.phiN;
                    const float rm = HarmGeometry::radiusAt(cfg, irm);
                    const float rp = HarmGeometry::radiusAt(cfg, irp);
                    const float sqrtGRm = KerrSchild::metric(rm, theta, cfg.spin).sqrtMinusG;
                    const float sqrtGRp = KerrSchild::metric(rp, theta, cfg.spin).sqrtMinusG;
                    const float sqrtGTm = KerrSchild::metric(geom.r, thm, cfg.spin).sqrtMinusG;
                    const float sqrtGTp = KerrSchild::metric(geom.r, thp, cfg.spin).sqrtMinusG;
                    const float radial = (sqrtGRp * at(irp, it, ip, 5) - sqrtGRm * at(irm, it, ip, 5)) / std::max(rp - rm, 1e-4f);
                    const float polar = (sqrtGTp * at(ir, itp, ip, 6) - sqrtGTm * at(ir, itm, ip, 6)) /
                        std::max((itp - itm) * dtheta, 1e-4f);
                    const float azimuth = geom.metric.sqrtMinusG * (at(ir, it, ipp, 7) - at(ir, it, ipm, 7)) / std::max(2.0f * dphi, 1e-4f);
                    const float divB = (radial + polar + azimuth) / std::max(geom.metric.sqrtMinusG, 1.0e-8f);
                    out.divBL1 += std::abs(divB) * geom.volume;
                    out.divBMax = std::max(out.divBMax, std::abs(divB));
                    divBVolume += geom.volume;
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
