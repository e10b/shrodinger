#pragma once

#include <algorithm>
#include <cmath>
#include <ostream>
#include <string>
#include <vector>

#include "harm_amr_blocks.h"
#include "harm_config.h"
#include "harm_diagnostics.h"
#include "harm_flux.h"
#include "harm_grid.h"
#include "harm_initial_data.h"
#include "harm_kerr_schild.h"
#include "harm_primitive_recovery.h"
#include "harm_state.h"
#include "harm_state_norms.h"

namespace harm {

struct MethodCheck {
    std::string name;
    float value = 0.0f;
    float tolerance = 0.0f;
    bool pass = false;
};

struct MethodSuiteReport {
    std::vector<MethodCheck> checks;

    bool passed() const {
        return std::all_of(checks.begin(), checks.end(), [](const MethodCheck& c) {
            return c.pass;
        });
    }
};

class MethodSuite {
public:
    static MethodSuiteReport run(const Config& cfg) {
        MethodSuiteReport report{};
        recoveryRoundTrip(report, cfg);
        fluxConsistency(report, cfg);
        metricConsistency(report, cfg);
        stationaryAtmosphere(report, cfg);
        initialDivB(report, cfg);
        amrBlockRoundTrip(report, cfg);
        return report;
    }

private:
    static void add(MethodSuiteReport& report, const std::string& name, float value, float tolerance) {
        MethodCheck c{};
        c.name = name;
        c.value = value;
        c.tolerance = tolerance;
        c.pass = std::abs(value) <= tolerance;
        report.checks.push_back(c);
    }

    static void recoveryRoundTrip(MethodSuiteReport& report, const Config& cfg) {
        float maxRhoErr = 0.0f;
        float maxUErr = 0.0f;
        float maxVErr = 0.0f;
        float failCount = 0.0f;
        const float radii[] = {6.5f, 12.0f, 24.0f};
        const float thetas[] = {0.35f * kPi, 0.50f * kPi, 0.65f * kPi};
        for (float r : radii) {
            for (float th : thetas) {
                const Metric metric = KerrSchild::metric(r, th, cfg.spin);
                Primitive p{};
                p.rho = 0.05f + 0.01f * r;
                p.u = 0.01f + 0.002f * r;
                p.v = glm::vec3(-0.02f, 0.01f * std::sin(th), 0.18f / std::sqrt(r));
                p.B = glm::vec3(0.002f * std::sin(th), 0.001f, 0.003f / r);
                const Conserved u = HarmState::primitiveToConserved(p, metric);
                const RecoveryResult recovered = PrimitiveRecovery::recover(u, p, metric, cfg.rhoFloor, cfg.uFloor);
                failCount += recovered.failed ? 1.0f : 0.0f;
                maxRhoErr = std::max(maxRhoErr, rel(recovered.primitive.rho, p.rho));
                maxUErr = std::max(maxUErr, rel(recovered.primitive.u, p.u));
                maxVErr = std::max(maxVErr, glm::length(recovered.primitive.v - p.v));
            }
        }
        add(report, "primitive recovery rho relative error", maxRhoErr, 5.0e-2f);
        add(report, "primitive recovery u relative error", maxUErr, 5.0e-1f);
        add(report, "primitive recovery velocity absolute error", maxVErr, 5.0e-2f);
        add(report, "primitive recovery failures", failCount, 0.0f);
    }

    static void fluxConsistency(MethodSuiteReport& report, const Config& cfg) {
        const Metric metric = KerrSchild::metric(12.0f, 0.5f * kPi, cfg.spin);
        Primitive p{};
        p.rho = 0.2f;
        p.u = 0.04f;
        p.v = glm::vec3(-0.01f, 0.0f, 0.06f);
        p.B = glm::vec3(0.002f, 0.003f, 0.001f);
        float maxErr = 0.0f;
        for (int dir = 0; dir < 3; ++dir) {
            const Flux f = HarmFlux::hll(p, p, metric, dir);
            const Flux exact = HarmState::physicalFlux(p, metric, dir);
            maxErr = std::max(maxErr, std::abs(f.D - exact.D));
            maxErr = std::max(maxErr, glm::length(f.S - exact.S));
            maxErr = std::max(maxErr, std::abs(f.tau - exact.tau));
            maxErr = std::max(maxErr, glm::length(f.B - exact.B));
        }
        add(report, "HLL equal-state flux consistency", maxErr, 1.0e-6f);
    }

    static void metricConsistency(MethodSuiteReport& report, const Config& cfg) {
        float maxInverseError = 0.0f;
        float maxNormError = 0.0f;
        for (float r : {1.1f * KerrSchild::horizonRadius(cfg.spin), 6.0f, 12.0f}) {
            const Metric metric = KerrSchild::metric(r, 0.43f * kPi, cfg.spin);
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    float product = 0.0f;
                    for (int a = 0; a < 4; ++a) product += metric.gcov[mu][a] * metric.gcon[a][nu];
                    maxInverseError = std::max(maxInverseError, std::abs(product - (mu == nu ? 1.0f : 0.0f)));
                }
            }
            Primitive p{};
            p.rho = 1.0f;
            p.u = 0.1f;
            p.v = glm::vec3(-0.02f, 0.001f, 1.0f / (std::pow(r, 1.5f) + cfg.spin));
            const FourVector u = HarmState::fourVelocity(p, metric);
            maxNormError = std::max(maxNormError, std::abs(KerrSchild::dot(metric, u, u) + 1.0f));
        }
        add(report, "metric inverse identity", maxInverseError, 2.0e-5f);
        add(report, "four-velocity normalization", maxNormError, 2.0e-5f);
    }

    static void stationaryAtmosphere(MethodSuiteReport& report, const Config& cfg) {
        const Metric metric = KerrSchild::metric(12.0f, 0.5f * kPi, cfg.spin);
        Primitive p{};
        p.rho = 0.2f;
        p.u = 0.02f;
        p.v = glm::vec3(-metric.beta[0], -metric.beta[1], -metric.beta[2]);
        p.B = glm::vec3(0.001f, 0.0f, 0.0f);
        const Conserved u = HarmState::primitiveToConserved(p, metric);
        const RecoveryResult recovered = PrimitiveRecovery::recover(u, p, metric, cfg.rhoFloor, cfg.uFloor);
        add(report, "normal-observer recovery", recovered.failed ? 1.0f : glm::length(recovered.primitive.v - p.v), 2.0e-5f);
    }

    static void initialDivB(MethodSuiteReport& report, const Config& cfg) {
        Config local = cfg;
        local.radialN = std::min(cfg.radialN, 64);
        local.thetaN = std::min(cfg.thetaN, 32);
        local.phiN = std::min(cfg.phiN, 64);
        local.initialData = 2;
        local.clamp();
        Grid grid{};
        const Diagnostics diagnostics = InitialDataBuilder::build(local, grid);
        add(report, "Fishbone initial divB L1", diagnostics.divBL1, 5.0e-3f);
        add(report, "Fishbone initial primitive fail fraction", diagnostics.failFrac, 1.0e-5f);
    }

    static void amrBlockRoundTrip(MethodSuiteReport& report, const Config& cfg) {
        Config local = cfg;
        local.radialN = std::min(cfg.radialN, 64);
        local.thetaN = std::min(cfg.thetaN, 32);
        local.phiN = std::min(cfg.phiN, 64);
        local.initialData = 2;
        local.clamp();
        Grid grid{};
        InitialDataBuilder::build(local, grid);

        const AmrBlock fine = AmrBlockOps::loadWithGhosts(local, grid, 24, 8, 0, 16, 16, 16);
        const AmrBlock coarse = AmrBlockOps::restrict2x(fine);
        const AmrBlock prolonged = AmrBlockOps::prolong2xNearest(coarse);
        const float rhoErr = AmrBlockOps::interiorNormalizedL1(fine, prolonged, 0);
        const float uErr = AmrBlockOps::interiorNormalizedL1(fine, prolonged, 1);
        add(report, "AMR block rho restrict/prolong L1", rhoErr, 0.60f);
        add(report, "AMR block u restrict/prolong L1", uErr, 0.60f);

        Grid injected = grid;
        AmrBlockOps::injectInterior(local, fine, injected);
        const StateNorms norms = StateNormSampler::compare(local, grid.packed, injected.packed);
        add(report, "AMR ghost block inject rho L1", norms.rhoL1, 1.0e-7f);
    }

    static float rel(float a, float b) {
        return std::abs(a - b) / std::max(std::abs(b), 1.0e-12f);
    }
};

inline void writeMethodSuite(const MethodSuiteReport& report, std::ostream& os) {
    os << "\n## Method Validation Suite\n\n";
    os << "| Check | Value | Tolerance | Result |\n|---|---:|---:|---|\n";
    for (const MethodCheck& c : report.checks) {
        os << "| " << c.name << " | " << c.value << " | " << c.tolerance << " | "
           << (c.pass ? "PASS" : "FAIL") << " |\n";
    }
    os << "\nMethod suite: **" << (report.passed() ? "PASS" : "FAIL") << "**\n";
}

} // namespace harm
