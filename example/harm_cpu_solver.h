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
    enum class Integrator {
        Heun,
        Midpoint,
    };

    static void step(const Config& cfg, Grid& grid, Diagnostics& diagnostics, float& time) {
        stepWithIntegrator(cfg, grid, diagnostics, time, Integrator::Heun);
    }

    static void stepWithIntegrator(const Config& cfg, Grid& grid, Diagnostics& diagnostics,
                                   float& time, Integrator integrator) {
        if (grid.packed.size() < packedFloatCount(cfg.cellCount()) || cfg.paused) {
            return;
        }

        std::vector<float> next = grid.packed;
        std::vector<float> stage2 = grid.packed;
        const int substeps = std::clamp(cfg.substeps, 1, Config::kMaxSubstepsPerFrame);
        for (int sub = 0; sub < substeps; ++sub) {
            const float dt = HarmTimeStepper::stableDt(cfg, grid.packed);
            if (integrator == Integrator::Midpoint) {
                stepOnce(cfg, grid.packed, stage2, 0.5f * dt, SolverRegion::full(cfg));
                stepFromBase(cfg, grid.packed, stage2, next, dt, SolverRegion::full(cfg));
            } else {
                stepOnce(cfg, grid.packed, next, dt, SolverRegion::full(cfg));
                stepOnce(cfg, next, stage2, dt, SolverRegion::full(cfg));
                averageStages(cfg, grid.packed, stage2, next, SolverRegion::full(cfg));
            }
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
        auto entropyAt = [&](int ir, int it, int ip) {
            const size_t idx = index(ir, it, ip);
            const float stored = in[idx * 12 + 8];
            if (stored > 0.0f) return stored;
            const CellGeometry geom = HarmGeometry::cell(cfg, ir, it);
            return HarmState::conservedEntropy(primAt(ir, it, ip), geom.metric);
        };
        auto entropyFlux = [&](int ar, int at, int ap, int br, int bt, int bp, int dir) {
            const Primitive a = primAt(ar, at, ap);
            const Primitive b = primAt(br, bt, bp);
            const float qa = entropyAt(ar, at, ap);
            const float qb = entropyAt(br, bt, bp);
            const float speed = std::max(HarmFlux::maxSignalSpeed(a, HarmGeometry::cell(cfg, ar, at).metric, dir),
                                         HarmFlux::maxSignalSpeed(b, HarmGeometry::cell(cfg, br, bt).metric, dir));
            return 0.5f * (qa * a.v[dir] + qb * b.v[dir] - speed * (qb - qa));
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
                    addFlux(u, ftm, dt / geom.dtheta);
                    addFlux(u, ftp, -dt / geom.dtheta);
                    addFlux(u, fpm, dt / geom.dphi);
                    addFlux(u, fpp, -dt / geom.dphi);
                    addMetricSources(u, c, geom.metric, dt);
                    float entropy = entropyAt(ir, it, ip);
                    entropy += dt * ((entropyFlux(ir - 1, it, ip, ir, it, ip, 0) - entropyFlux(ir, it, ip, ir + 1, it, ip, 0)) / geom.radialLength() +
                                     (entropyFlux(ir, it - 1, ip, ir, it, ip, 1) - entropyFlux(ir, it, ip, ir, it + 1, ip, 1)) / geom.dtheta +
                                     (entropyFlux(ir, it, ip - 1, ir, it, ip, 2) - entropyFlux(ir, it, ip, ir, it, ip + 1, 2)) / geom.dphi);
                    entropy = std::max(entropy, 1.0e-20f);

                    // Evolve the densitized magnetic field as a discrete curl
                    // of edge-centered ideal-MHD EMFs.  This is the CT update;
                    // do not use the independently reconstructed cell-face B
                    // fluxes below, whose divergence does not cancel exactly.
                    u.B = constrainedTransportUpdate(cfg, in, ir, it, ip, dt, index);

                    const RecoveryResult recovery = PrimitiveRecovery::recover(u, c, geom.metric, cfg.rhoFloorAt(geom.r), cfg.uFloorAt(geom.r), entropy);
                    Primitive recovered = recovery.primitive;
                    HarmBoundaries::applyOutflow(cfg, ir, it, recovered);
                    HarmState::toPacked(recovered, &out[index(ir, it, ip) * 12], geom.r, geom.theta);
                    out[index(ir, it, ip) * 12 + 8] = entropy;
                    out[index(ir, it, ip) * 12 + 9] = recovery.usedEntropyFallback && !recovery.failed ? 1.0f : 0.0f;
                    out[index(ir, it, ip) * 12 + 10] = recovery.failed ? 1.0f : 0.0f;
                }
            }
        }
    }

    static void addFlux(Conserved& u, const Flux& f, float scale) {
        u.D += scale * f.D;
        u.S += scale * f.S;
        u.tau += scale * f.tau;
        (void)f.B;
    }

    static void stepFromBase(const Config& cfg, const std::vector<float>& base,
                             const std::vector<float>& midpoint, std::vector<float>& out,
                             float dt, const SolverRegion& region) {
        // The GPU midpoint operator uses the base conserved state and evaluates
        // all flux/source/CT terms from the midpoint primitives.  Reuse the
        // Euler operator by replacing its implicit base after evaluating the
        // midpoint increment.
        std::vector<float> eulerMid = midpoint;
        stepOnce(cfg, midpoint, eulerMid, dt, region);
        out = base;
        for (int ip = region.ip0; ip < region.ip0 + region.np; ++ip) {
            for (int it = region.it0; it < region.it0 + region.nt; ++it) {
                for (int ir = region.ir0; ir < region.ir0 + region.nr; ++ir) {
                    const size_t idx = (static_cast<size_t>(ip) * cfg.thetaN + it) * cfg.radialN + ir;
                    const CellGeometry geom = HarmGeometry::cell(cfg, ir, it);
                    const Primitive pb = HarmState::fromPacked(&base[idx * 12], geom.r, geom.theta);
                    const Primitive pm = HarmState::fromPacked(&midpoint[idx * 12], geom.r, geom.theta);
                    const Primitive pe = HarmState::fromPacked(&eulerMid[idx * 12], geom.r, geom.theta);
                    const Conserved ub = HarmState::primitiveToConserved(pb, geom.metric);
                    const Conserved um = HarmState::primitiveToConserved(pm, geom.metric);
                    const Conserved ue = HarmState::primitiveToConserved(pe, geom.metric);
                    Conserved target{};
                    target.D = ub.D + (ue.D - um.D);
                    target.S = ub.S + (ue.S - um.S);
                    target.tau = ub.tau + (ue.tau - um.tau);
                    target.B = ub.B + (ue.B - um.B);
                    const float entropy = std::max(base[idx * 12 + 8] + eulerMid[idx * 12 + 8] - midpoint[idx * 12 + 8], 1.0e-20f);
                    const RecoveryResult recovery = PrimitiveRecovery::recover(target, pm, geom.metric, cfg.rhoFloorAt(geom.r), cfg.uFloorAt(geom.r), entropy);
                    Primitive p = recovery.primitive;
                    HarmBoundaries::applyOutflow(cfg, ir, it, p);
                    HarmState::toPacked(p, &out[idx * 12], geom.r, geom.theta);
                    out[idx * 12 + 8] = entropy;
                    out[idx * 12 + 9] = recovery.usedEntropyFallback && !recovery.failed ? 1.0f : 0.0f;
                    out[idx * 12 + 10] = recovery.failed ? 1.0f : 0.0f;
                }
            }
        }
    }

    template <typename IndexFn>
    static glm::vec3 constrainedTransportUpdate(const Config& cfg, const std::vector<float>& in,
                                                int ir, int it, int ip, float dt, IndexFn index) {
        auto prim = [&](int r, int t, int p) {
            const CellGeometry g = HarmGeometry::cell(cfg, r, t);
            return HarmState::fromPacked(&in[index(r, t, p) * 12], g.r, g.theta);
        };
        auto densitizedE = [&](int r, int t, int p) {
            const CellGeometry g = HarmGeometry::cell(cfg, r, t);
            // E_i = -sqrt(-g) epsilon_ijk v^j B^k.  A common edge EMF is
            // obtained by averaging the adjacent cell-centered values.
            return g.metric.sqrtMinusG * ConstrainedTransport::idealElectricField(prim(r, t, p));
        };
        auto edgeEr = [&](int r, int t, int p) {
            return 0.25f * (densitizedE(r, t, p).x + densitizedE(r, t - 1, p).x +
                            densitizedE(r, t, p - 1).x + densitizedE(r, t - 1, p - 1).x);
        };
        auto edgeEt = [&](int r, int t, int p) {
            return 0.25f * (densitizedE(r, t, p).y + densitizedE(r - 1, t, p).y +
                            densitizedE(r, t, p - 1).y + densitizedE(r - 1, t, p - 1).y);
        };
        auto edgeEp = [&](int r, int t, int p) {
            return 0.25f * (densitizedE(r, t, p).z + densitizedE(r - 1, t, p).z +
                            densitizedE(r, t - 1, p).z + densitizedE(r - 1, t - 1, p).z);
        };

        const CellGeometry g = HarmGeometry::cell(cfg, ir, it);
        const Primitive c = prim(ir, it, ip);
        glm::vec3 Bdens = g.metric.sqrtMinusG * c.B;
        const float epThetaP = edgeEp(ir, it + 1, ip);
        const float epThetaM = edgeEp(ir, it, ip);
        const float etPhiP = edgeEt(ir, it, ip + 1);
        const float etPhiM = edgeEt(ir, it, ip);
        const float erPhiP = edgeEr(ir, it, ip + 1);
        const float erPhiM = edgeEr(ir, it, ip);
        const float epRadialP = edgeEp(ir + 1, it, ip);
        const float epRadialM = edgeEp(ir, it, ip);
        const float etRadialP = edgeEt(ir + 1, it, ip);
        const float etRadialM = edgeEt(ir, it, ip);
        const float erThetaP = edgeEr(ir, it + 1, ip);
        const float erThetaM = edgeEr(ir, it, ip);
        Bdens.x -= dt * ((epThetaP - epThetaM) / g.dtheta - (etPhiP - etPhiM) / g.dphi);
        Bdens.y -= dt * ((erPhiP - erPhiM) / g.dphi - (epRadialP - epRadialM) / g.dr);
        Bdens.z -= dt * ((etRadialP - etRadialM) / g.dr - (erThetaP - erThetaM) / g.dtheta);
        return Bdens;
    }

    static void averageStages(const Config& cfg, const std::vector<float>& initial,
                              const std::vector<float>& secondEuler, std::vector<float>& out,
                              const SolverRegion& region) {
        auto index = [&](int ir, int it, int ip) -> size_t {
            int p = ip % cfg.phiN;
            if (p < 0) p += cfg.phiN;
            return (static_cast<size_t>(p) * cfg.thetaN + std::clamp(it, 0, cfg.thetaN - 1)) * cfg.radialN +
                   std::clamp(ir, 0, cfg.radialN - 1);
        };
        for (int ip = region.ip0; ip < region.ip0 + region.np; ++ip) {
            for (int it = region.it0; it < region.it0 + region.nt; ++it) {
                for (int ir = region.ir0; ir < region.ir0 + region.nr; ++ir) {
                    const size_t idx = index(ir, it, ip);
                    const CellGeometry geom = HarmGeometry::cell(cfg, ir, it);
                    const Primitive p0 = HarmState::fromPacked(&initial[idx * 12], geom.r, geom.theta);
                    const Primitive p2 = HarmState::fromPacked(&secondEuler[idx * 12], geom.r, geom.theta);
                    const Conserved u0 = HarmState::primitiveToConserved(p0, geom.metric);
                    const Conserved u2 = HarmState::primitiveToConserved(p2, geom.metric);
                    Conserved averaged{};
                    averaged.D = 0.5f * (u0.D + u2.D);
                    averaged.S = 0.5f * (u0.S + u2.S);
                    averaged.tau = 0.5f * (u0.tau + u2.tau);
                    averaged.B = 0.5f * (u0.B + u2.B);
                    const float entropy = std::max(0.5f * (initial[idx * 12 + 8] + secondEuler[idx * 12 + 8]), 1.0e-20f);
                    const RecoveryResult recovery = PrimitiveRecovery::recover(averaged, p2, geom.metric, cfg.rhoFloorAt(geom.r), cfg.uFloorAt(geom.r), entropy);
                    Primitive p = recovery.primitive;
                    HarmBoundaries::applyOutflow(cfg, ir, it, p);
                    HarmState::toPacked(p, &out[idx * 12], geom.r, geom.theta);
                    out[idx * 12 + 8] = entropy;
                    out[idx * 12 + 9] = recovery.usedEntropyFallback && !recovery.failed ? 1.0f : 0.0f;
                    out[idx * 12 + 10] = recovery.failed ? 1.0f : 0.0f;
                }
            }
        }
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

};

} // namespace harm
