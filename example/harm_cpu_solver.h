#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "harm_config.h"
#include "harm_constrained_transport.h"
#include "harm_diagnostics.h"
#include "harm_flux.h"
#include "harm_grid.h"
#include "harm_kerr_schild.h"
#include "harm_primitive_recovery.h"

namespace harm {

class CpuSolver {
public:
    static void step(const Config& cfg, Grid& grid, Diagnostics& diagnostics, float& time) {
        if (grid.packed.size() < packedFloatCount(cfg.cellCount()) || cfg.paused) {
            return;
        }

        std::vector<float> next = grid.packed;
        const int substeps = std::clamp(cfg.substeps, 1, 12);
        const float dt = std::clamp(cfg.dt, 0.00005f, 0.03f);
        for (int sub = 0; sub < substeps; ++sub) {
            stepOnce(cfg, grid.packed, next, dt);
            grid.packed.swap(next);
            time += dt;
        }
        diagnostics = DiagnosticsSampler::compute(cfg, grid.packed, false);
    }

private:
    static void stepOnce(const Config& cfg, const std::vector<float>& in, std::vector<float>& out, float dt) {
        const float logRange = std::max(std::log(cfg.rout) - std::log(cfg.rin), 1.0e-6f);
        const float dtheta = 0.84f * kPi / static_cast<float>(std::max(cfg.thetaN, 1));
        const float dphi = 2.0f * kPi / static_cast<float>(std::max(cfg.phiN, 1));

        auto index = [&](int ir, int it, int ip) -> size_t {
            int p = ip % cfg.phiN;
            if (p < 0) p += cfg.phiN;
            const int r = std::clamp(ir, 0, cfg.radialN - 1);
            const int t = std::clamp(it, 0, cfg.thetaN - 1);
            return (static_cast<size_t>(p) * static_cast<size_t>(cfg.thetaN) + static_cast<size_t>(t)) *
                static_cast<size_t>(cfg.radialN) + static_cast<size_t>(r);
        };

        auto radius = [&](int ir) -> float {
            const float x = (static_cast<float>(std::clamp(ir, 0, cfg.radialN - 1)) + 0.5f) / static_cast<float>(cfg.radialN);
            return std::exp(std::log(cfg.rin) + x * logRange);
        };

        auto theta = [&](int it) -> float {
            const float y = (static_cast<float>(std::clamp(it, 0, cfg.thetaN - 1)) + 0.5f) / static_cast<float>(cfg.thetaN);
            return 0.08f * kPi + y * 0.84f * kPi;
        };

        auto primAt = [&](int ir, int it, int ip) -> Primitive {
            const float r = radius(ir);
            const float th = theta(it);
            return HarmState::fromPacked(&in[index(ir, it, ip) * 12], r, th);
        };

        for (int ip = 0; ip < cfg.phiN; ++ip) {
            for (int it = 0; it < cfg.thetaN; ++it) {
                const float th = theta(it);
                const float sinTh = std::max(std::sin(th), 0.08f);
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const float r = radius(ir);
                    const float dr = std::max(r * logRange / static_cast<float>(cfg.radialN), 1.0e-4f);
                    const Metric metric = KerrSchild::metric(r, th, cfg.spin);
                    const Primitive c = primAt(ir, it, ip);
                    Conserved u = HarmState::primitiveToConserved(c, metric);

                    const Flux frm = HarmFlux::hll(primAt(ir - 1, it, ip), c, metric, 0);
                    const Flux frp = HarmFlux::hll(c, primAt(ir + 1, it, ip), metric, 0);
                    const Flux ftm = HarmFlux::hll(primAt(ir, it - 1, ip), c, metric, 1);
                    const Flux ftp = HarmFlux::hll(c, primAt(ir, it + 1, ip), metric, 1);
                    const Flux fpm = HarmFlux::hll(primAt(ir, it, ip - 1), c, metric, 2);
                    const Flux fpp = HarmFlux::hll(c, primAt(ir, it, ip + 1), metric, 2);

                    addFlux(u, frm, dt / dr);
                    addFlux(u, frp, -dt / dr);
                    addFlux(u, ftm, dt / std::max(r * dtheta, 1.0e-4f));
                    addFlux(u, ftp, -dt / std::max(r * dtheta, 1.0e-4f));
                    addFlux(u, fpm, dt / std::max(r * sinTh * dphi, 1.0e-4f));
                    addFlux(u, fpp, -dt / std::max(r * sinTh * dphi, 1.0e-4f));
                    addMetricSources(u, c, metric, dt);

                    const RecoveryResult recovery = PrimitiveRecovery::recover(u, c, metric, cfg.rhoFloor, cfg.uFloor);
                    Primitive recovered = recovery.primitive;
                    applyDivergenceControl(recovered,
                        primAt(ir - 1, it, ip), primAt(ir + 1, it, ip),
                        primAt(ir, it - 1, ip), primAt(ir, it + 1, ip),
                        primAt(ir, it, ip - 1), primAt(ir, it, ip + 1),
                        r, th, dr, dtheta, dphi, dt);
                    applyBoundaries(cfg, ir, it, recovered);
                    HarmState::toPacked(recovered, &out[index(ir, it, ip) * 12], r, th);
                    out[index(ir, it, ip) * 12 + 8] = (recovery.failed || recovery.usedEntropyFallback) ? 1.0f : 0.0f;
                }
            }
        }
    }

    static void addFlux(Conserved& u, const Flux& f, float scale) {
        u.D += scale * f.D;
        u.S += scale * f.S;
        u.tau += scale * f.tau;
        u.B += scale * f.B;
    }

    static void addMetricSources(Conserved& u, const Primitive& p, const Metric& metric, float dt) {
        const StressEnergy stress = HarmState::stressEnergyContravariant(p, metric);
        const MetricDerivatives deriv = KerrSchild::derivatives(metric.r, metric.theta, metric.spin);
        float srcR = 0.0f;
        float srcTheta = 0.0f;
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                srcR += stress.T[mu][nu] * deriv.dr[mu][nu];
                srcTheta += stress.T[mu][nu] * deriv.dtheta[mu][nu];
            }
        }
        u.S.x += 0.5f * dt * metric.sqrtMinusG * srcR;
        u.S.y += 0.5f * dt * metric.sqrtMinusG * srcTheta;
    }

    static void applyDivergenceControl(Primitive& p,
                                       const Primitive& rm, const Primitive& rp,
                                       const Primitive& tm, const Primitive& tp,
                                       const Primitive& pm, const Primitive& pp,
                                       float r, float theta, float dr, float dtheta, float dphi, float dt) {
        const float divB = ConstrainedTransport::sphericalDivB(rm, rp, tm, tp, pm, pp, r, theta, dr, dtheta, dphi);
        const float sinTh = std::max(std::sin(theta), 0.08f);
        const float damp = std::clamp(0.35f * dt, 0.0f, 0.25f);
        p.B.x -= damp * divB * dr;
        p.B.y -= damp * divB * r * dtheta;
        p.B.z -= 0.5f * damp * divB * r * sinTh * dphi;
    }

    static void applyBoundaries(const Config& cfg, int ir, int it, Primitive& p) {
        if (ir < 3) {
            p.v.x = std::min(p.v.x, -0.05f);
            p.rho = std::max(0.82f * p.rho, cfg.rhoFloor);
            p.u = std::max(0.82f * p.u, cfg.uFloor);
        }
        if (ir > cfg.radialN - 4) {
            p.rho = std::max(0.92f * p.rho, cfg.rhoFloor);
            p.u = std::max(0.92f * p.u, cfg.uFloor);
        }
        const int polarBand = std::min(it, cfg.thetaN - 1 - it);
        if (polarBand < 2) {
            p.rho = std::max(0.80f * p.rho, cfg.rhoFloor);
            p.u = std::max(0.80f * p.u, cfg.uFloor);
            p.v.y *= 0.5f;
        }
        p.rho = std::max(p.rho, cfg.rhoFloor);
        p.u = std::max(p.u, cfg.uFloor);
    }
};

} // namespace harm
