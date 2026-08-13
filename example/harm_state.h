#pragma once

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>

#include "harm_kerr_schild.h"
#include "harm_types.h"

namespace harm {

constexpr float kAdiabaticGamma = 4.0f / 3.0f;

struct Primitive {
    float rho = 0.0f;
    float u = 0.0f;
    glm::vec3 v = glm::vec3(0.0f);
    glm::vec3 B = glm::vec3(0.0f);
};

struct Conserved {
    float D = 0.0f;
    glm::vec3 S = glm::vec3(0.0f);
    float tau = 0.0f;
    glm::vec3 B = glm::vec3(0.0f);
};

struct Flux {
    float D = 0.0f;
    glm::vec3 S = glm::vec3(0.0f);
    float tau = 0.0f;
    glm::vec3 B = glm::vec3(0.0f);
};

struct StressEnergy {
    float T[4][4] = {};
};

class HarmState {
public:
    static float pressure(const Primitive& p) {
        return (kAdiabaticGamma - 1.0f) * std::max(p.u, 0.0f);
    }

    static float lorentzFactor(const Primitive& p) {
        // This overload is retained for diagnostics that do not have a metric.
        // Solver code must use fourVelocity()/lorentzFactor(p, metric): packed
        // velocities are coordinate transport velocities dx^i/dt, not Cartesian
        // three-velocities.
        const float v2 = std::clamp(glm::dot(p.v, p.v), 0.0f, 0.999f);
        return 1.0f / std::sqrt(std::max(1.0f - v2, 1.0e-8f));
    }

    static float lorentzFactor(const Primitive& p, const Metric& metric) {
        return std::max(metric.alpha * fourVelocity(p, metric)[0], 1.0f);
    }

    static float enthalpy(const Primitive& p) {
        return std::max(p.rho + p.u + pressure(p), p.rho);
    }

    static Conserved primitiveToConserved(const Primitive& p0, const Metric& metric) {
        Primitive p = sanitize(p0);
        const FourVector ucon = fourVelocity(p, metric);
        const StressEnergy stress = stressEnergyContravariant(p, metric);
        const float rhoUt = p.rho * ucon[0];
        Conserved u{};
        u.D = metric.sqrtMinusG * rhoUt;
        for (int j = 0; j < 3; ++j) {
            float mixed = 0.0f;
            for (int mu = 0; mu < 4; ++mu) {
                mixed += stress.T[0][mu] * metric.gcov[mu][j + 1];
            }
            u.S[j] = metric.sqrtMinusG * mixed;
        }
        float tMixedT = 0.0f;
        for (int mu = 0; mu < 4; ++mu) {
            tMixedT += stress.T[0][mu] * metric.gcov[mu][0];
        }
        u.tau = metric.sqrtMinusG * (-tMixedT - rhoUt);
        u.B = metric.sqrtMinusG * p.B;
        return u;
    }

    static FourVector fourVelocity(const Primitive& p0, const Metric& metric) {
        const Primitive p = sanitize(p0);
        // p.v is the coordinate transport velocity u^i/u^t.  Normalize
        // u^mu = u^t(1,v^i) with g_mu_nu u^mu u^nu = -1.
        float norm = metric.gcov[0][0];
        for (int i = 0; i < 3; ++i) {
            norm += 2.0f * metric.gcov[0][i + 1] * p.v[i];
            for (int j = 0; j < 3; ++j) {
                norm += metric.gcov[i + 1][j + 1] * p.v[i] * p.v[j];
            }
        }
        const float ut = 1.0f / std::sqrt(std::max(-norm, 1.0e-10f));
        FourVector u{};
        u[0] = ut;
        for (int i = 0; i < 3; ++i) u[i + 1] = ut * p.v[i];
        return u;
    }

    static FourVector magneticFourVector(const Primitive& p0, const Metric& metric) {
        const Primitive p = sanitize(p0);
        const FourVector ucon = fourVelocity(p, metric);
        const float ut = std::max(ucon[0], 1.0e-8f);
        const float b0 = p.B.x * KerrSchild::lower(metric, ucon, 1) +
            p.B.y * KerrSchild::lower(metric, ucon, 2) +
            p.B.z * KerrSchild::lower(metric, ucon, 3);

        FourVector b{};
        b[0] = b0;
        b[1] = (p.B.x + b0 * ucon[1]) / ut;
        b[2] = (p.B.y + b0 * ucon[2]) / ut;
        b[3] = (p.B.z + b0 * ucon[3]) / ut;
        return b;
    }

    static StressEnergy stressEnergyContravariant(const Primitive& p0, const Metric& metric) {
        const Primitive p = sanitize(p0);
        const FourVector ucon = fourVelocity(p, metric);
        const FourVector bcon = magneticFourVector(p, metric);
        const float pg = pressure(p);
        const float bsq = std::max(KerrSchild::dot(metric, bcon, bcon), 0.0f);
        const float w = p.rho + p.u + pg + bsq;
        StressEnergy out{};
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                out.T[mu][nu] = w * ucon[mu] * ucon[nu] + (pg + 0.5f * bsq) * metric.gcon[mu][nu] - bcon[mu] * bcon[nu];
            }
        }
        return out;
    }

    static Flux physicalFlux(const Primitive& p0, const Metric& metric, int dir) {
        Primitive p = sanitize(p0);
        const FourVector ucon = fourVelocity(p, metric);
        const FourVector bcon = magneticFourVector(p, metric);
        const StressEnergy stress = stressEnergyContravariant(p, metric);
        const int mu = dir + 1;

        Flux f{};
        const float rhoUi = p.rho * ucon[mu];
        f.D = metric.sqrtMinusG * rhoUi;
        for (int j = 0; j < 3; ++j) {
            float mixed = 0.0f;
            for (int nu = 0; nu < 4; ++nu) {
                mixed += stress.T[mu][nu] * metric.gcov[nu][j + 1];
            }
            f.S[j] = metric.sqrtMinusG * mixed;
        }
        float mixedT = 0.0f;
        for (int nu = 0; nu < 4; ++nu) mixedT += stress.T[mu][nu] * metric.gcov[nu][0];
        f.tau = metric.sqrtMinusG * (-mixedT - rhoUi);
        for (int j = 0; j < 3; ++j) {
            f.B[j] = metric.sqrtMinusG * (bcon[j + 1] * ucon[mu] - bcon[mu] * ucon[j + 1]);
        }
        f.B[dir] = 0.0f;
        return f;
    }

    static Primitive sanitize(const Primitive& in) {
        Primitive out = in;
        out.rho = std::max(out.rho, 1.0e-12f);
        out.u = std::max(out.u, 1.0e-12f);
        // Coordinate velocities are metric-dependent and are limited in
        // fourVelocity(), where the timelike normalization is available.
        for (int i = 0; i < 3; ++i) out.v[i] = std::clamp(out.v[i], -0.999f, 0.999f);
        return out;
    }

    static Primitive fromPacked(const float* c, float r, float theta) {
        Primitive p{};
        p.rho = c[0];
        p.u = c[1];
        p.v = glm::vec3(c[2], c[3], c[4]);
        p.B = glm::vec3(c[5], c[6], c[7]);
        (void)r;
        (void)theta;
        return sanitize(p);
    }

    static void toPacked(const Primitive& p0, float* c, float r, float theta) {
        const Primitive p = sanitize(p0);
        c[0] = p.rho;
        c[1] = p.u;
        c[2] = p.v.x;
        c[3] = p.v.y;
        c[4] = p.v.z;
        c[5] = p.B.x;
        c[6] = p.B.y;
        c[7] = p.B.z;
        (void)r;
        (void)theta;
    }
};

} // namespace harm
