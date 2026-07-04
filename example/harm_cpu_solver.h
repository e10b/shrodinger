#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "harm_boundaries.h"
#include "harm_config.h"
#include "harm_constrained_transport.h"
#include "harm_diagnostics.h"
#include "harm_flux.h"
#include "harm_geometry.h"
#include "harm_grid.h"
#include "harm_kerr_schild.h"
#include "harm_primitive_recovery.h"
#include "harm_timestep.h"

namespace harm {

struct SolverRegion {
    int ir0 = 0;
    int it0 = 0;
    int ip0 = 0;
    int nr = 0;
    int nt = 0;
    int np = 0;

    static SolverRegion full(const Config& cfg) {
        return {0, 0, 0, cfg.radialN, cfg.thetaN, cfg.phiN};
    }
};

class CpuSolver {
public:
    static void step(const Config& cfg, Grid& grid, Diagnostics& diagnostics, float& time) {
        if (grid.packed.size() < packedFloatCount(cfg.cellCount()) || cfg.paused) {
            return;
        }

        std::vector<float> next = grid.packed;
        const int substeps = std::clamp(cfg.substeps, 1, 12);
        for (int sub = 0; sub < substeps; ++sub) {
            const float dt = HarmTimeStepper::stableDt(cfg, grid.packed);
            stepOnce(cfg, grid.packed, next, dt, SolverRegion::full(cfg));
            grid.packed.swap(next);
            time += dt;
        }
        diagnostics = DiagnosticsSampler::compute(cfg, grid.packed, false);
    }

    static void stepRegion(const Config& cfg, Grid& grid, const SolverRegion& region, float dt) {
        if (grid.packed.size() < packedFloatCount(cfg.cellCount()) || cfg.paused) {
            return;
        }
        std::vector<float> next = grid.packed;
        stepOnce(cfg, grid.packed, next, dt, clippedRegion(cfg, region));
        grid.packed.swap(next);
    }

private:
    static SolverRegion clippedRegion(const Config& cfg, SolverRegion region) {
        region.ir0 = std::clamp(region.ir0, 0, cfg.radialN - 1);
        region.it0 = std::clamp(region.it0, 0, cfg.thetaN - 1);
        region.ip0 = std::clamp(region.ip0, 0, cfg.phiN - 1);
        region.nr = std::clamp(region.nr, 1, cfg.radialN - region.ir0);
        region.nt = std::clamp(region.nt, 1, cfg.thetaN - region.it0);
        region.np = std::clamp(region.np, 1, cfg.phiN - region.ip0);
        return region;
    }

    static void stepOnce(const Config& cfg, const std::vector<float>& in, std::vector<float>& out, float dt, const SolverRegion& region) {
        auto index = [&](int ir, int it, int ip) -> size_t {
            int p = ip % cfg.phiN;
            if (p < 0) p += cfg.phiN;
            const int r = std::clamp(ir, 0, cfg.radialN - 1);
            const int t = std::clamp(it, 0, cfg.thetaN - 1);
            return (static_cast<size_t>(p) * static_cast<size_t>(cfg.thetaN) + static_cast<size_t>(t)) *
                static_cast<size_t>(cfg.radialN) + static_cast<size_t>(r);
        };

        auto primAt = [&](int ir, int it, int ip) -> Primitive {
            const CellGeometry geom = HarmGeometry::cell(cfg, ir, it);
            return HarmState::fromPacked(&in[index(ir, it, ip) * 12], geom.r, geom.theta);
        };

        auto facePrim = [&](int ir0, int it0, int ip0, int dir, float side) -> Primitive {
            if (!cfg.highOrder) {
                return primAt(ir0, it0, ip0);
            }
            int dmR = 0;
            int dmT = 0;
            int dmP = 0;
            if (dir == 0) dmR = 1;
            if (dir == 1) dmT = 1;
            if (dir == 2) dmP = 1;
            return HarmFlux::reconstructMc(
                primAt(ir0 - dmR, it0 - dmT, ip0 - dmP),
                primAt(ir0, it0, ip0),
                primAt(ir0 + dmR, it0 + dmT, ip0 + dmP),
                side);
        };

        for (int ip = region.ip0; ip < region.ip0 + region.np; ++ip) {
            for (int it = region.it0; it < region.it0 + region.nt; ++it) {
                for (int ir = region.ir0; ir < region.ir0 + region.nr; ++ir) {
                    const CellGeometry geom = HarmGeometry::cell(cfg, ir, it);
                    const Primitive c = primAt(ir, it, ip);
                    Conserved u = HarmState::primitiveToConserved(c, geom.metric);

                    const Flux frm = HarmFlux::hll(facePrim(ir - 1, it, ip, 0, 0.5f), facePrim(ir, it, ip, 0, -0.5f), geom.metric, 0);
                    const Flux frp = HarmFlux::hll(facePrim(ir, it, ip, 0, 0.5f), facePrim(ir + 1, it, ip, 0, -0.5f), geom.metric, 0);
                    const Flux ftm = HarmFlux::hll(facePrim(ir, it - 1, ip, 1, 0.5f), facePrim(ir, it, ip, 1, -0.5f), geom.metric, 1);
                    const Flux ftp = HarmFlux::hll(facePrim(ir, it, ip, 1, 0.5f), facePrim(ir, it + 1, ip, 1, -0.5f), geom.metric, 1);
                    const Flux fpm = HarmFlux::hll(facePrim(ir, it, ip - 1, 2, 0.5f), facePrim(ir, it, ip, 2, -0.5f), geom.metric, 2);
                    const Flux fpp = HarmFlux::hll(facePrim(ir, it, ip, 2, 0.5f), facePrim(ir, it, ip + 1, 2, -0.5f), geom.metric, 2);

                    addFlux(u, frm, dt / geom.radialLength());
                    addFlux(u, frp, -dt / geom.radialLength());
                    addFlux(u, ftm, dt / geom.thetaLength());
                    addFlux(u, ftp, -dt / geom.thetaLength());
                    addFlux(u, fpm, dt / geom.phiLength());
                    addFlux(u, fpp, -dt / geom.phiLength());
                    addMetricSources(u, c, geom.metric, dt);

                    const RecoveryResult recovery = PrimitiveRecovery::recover(u, c, geom.metric, cfg.rhoFloor, cfg.uFloor);
                    Primitive recovered = recovery.primitive;
                    applyDivergenceControl(recovered,
                        primAt(ir - 1, it, ip), primAt(ir + 1, it, ip),
                        primAt(ir, it - 1, ip), primAt(ir, it + 1, ip),
                        primAt(ir, it, ip - 1), primAt(ir, it, ip + 1),
                        geom, dt);
                    HarmBoundaries::applyOutflow(cfg, ir, it, recovered);
                    HarmState::toPacked(recovered, &out[index(ir, it, ip) * 12], geom.r, geom.theta);
                    out[index(ir, it, ip) * 12 + 8] = recovery.failed ? 1.0f : 0.0f;
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
                                       const CellGeometry& geom, float dt) {
        const float divB = ConstrainedTransport::sphericalDivB(rm, rp, tm, tp, pm, pp, geom.r, geom.theta, geom.dr, geom.dtheta, geom.dphi);
        const float damp = std::clamp(0.35f * dt, 0.0f, 0.25f);
        p.B.x -= damp * divB * geom.radialLength();
        p.B.y -= damp * divB * geom.thetaLength();
        p.B.z -= 0.5f * damp * divB * geom.phiLength();
    }
};

} // namespace harm
