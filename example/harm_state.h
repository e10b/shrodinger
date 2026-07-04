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
        const float v2 = std::clamp(glm::dot(p.v, p.v), 0.0f, 0.999f);
        return 1.0f / std::sqrt(std::max(1.0f - v2, 1.0e-8f));
    }

    static float enthalpy(const Primitive& p) {
        return std::max(p.rho + p.u + pressure(p), p.rho);
    }

    static Conserved primitiveToConserved(const Primitive& p0, const Metric& metric) {
        Primitive p = sanitize(p0);
        const float W = lorentzFactor(p);
        const float bsq = glm::dot(p.B, p.B);
        const float w = enthalpy(p) + bsq;
        Conserved u{};
        u.D = metric.sqrtMinusG * p.rho * W;
        u.S = metric.sqrtMinusG * (w * W * W * p.v - glm::dot(p.v, p.B) * p.B);
        u.tau = metric.sqrtMinusG * (w * W * W - pressure(p) - 0.5f * bsq - p.rho * W);
        u.B = metric.sqrtMinusG * p.B;
        return u;
    }

    static FourVector fourVelocity(const Primitive& p0, const Metric& metric) {
        const Primitive p = sanitize(p0);
        return KerrSchild::normalObserverVelocity(metric, p.v);
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
        const float W = lorentzFactor(p);
        const float bsq = glm::dot(p.B, p.B);
        const float w = enthalpy(p) + bsq;
        const float vdir = p.v[dir];
        const float bdir = p.B[dir];
        const float vdotB = glm::dot(p.v, p.B);
        const float ptot = pressure(p) + 0.5f * bsq;

        Flux f{};
        f.D = metric.sqrtMinusG * p.rho * W * vdir;
        f.S = metric.sqrtMinusG * (w * W * W * p.v * vdir - p.B * bdir);
        f.S[dir] += metric.sqrtMinusG * ptot;
        f.tau = metric.sqrtMinusG * ((w * W * W - p.rho * W) * vdir - vdotB * bdir);
        f.B = metric.sqrtMinusG * (p.B * vdir - p.v * bdir);
        f.B[dir] = 0.0f;
        return f;
    }

    static Primitive sanitize(const Primitive& in) {
        Primitive out = in;
        out.rho = std::max(out.rho, 1.0e-12f);
        out.u = std::max(out.u, 1.0e-12f);
        const float v2 = glm::dot(out.v, out.v);
        if (v2 > 0.92f) {
            out.v *= std::sqrt(0.92f / std::max(v2, 1.0e-12f));
        }
        return out;
    }

    static Primitive fromPacked(const float* c, float r, float theta) {
        const float sinTh = std::max(std::sin(theta), 0.08f);
        Primitive p{};
        p.rho = c[0];
        p.u = c[1];
        p.v = glm::vec3(c[2], r * c[3], r * sinTh * c[4]);
        p.B = glm::vec3(c[5], c[6], c[7]);
        return sanitize(p);
    }

    static void toPacked(const Primitive& p0, float* c, float r, float theta) {
        const Primitive p = sanitize(p0);
        const float sinTh = std::max(std::sin(theta), 0.08f);
        c[0] = p.rho;
        c[1] = p.u;
        c[2] = p.v.x;
        c[3] = p.v.y / std::max(r, 1.0e-6f);
        c[4] = p.v.z / std::max(r * sinTh, 1.0e-6f);
        c[5] = p.B.x;
        c[6] = p.B.y;
        c[7] = p.B.z;
    }
};

} // namespace harm
