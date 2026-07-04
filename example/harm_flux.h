#pragma once

#include <algorithm>
#include <cmath>

#include "harm_state.h"

namespace harm {

class HarmFlux {
public:
    static Flux hll(const Primitive& left, const Primitive& right, const Metric& metric, int dir) {
        const Conserved ul = HarmState::primitiveToConserved(left, metric);
        const Conserved ur = HarmState::primitiveToConserved(right, metric);
        const Flux fl = HarmState::physicalFlux(left, metric, dir);
        const Flux fr = HarmState::physicalFlux(right, metric, dir);
        const float a = std::max(maxSignalSpeed(left, dir), maxSignalSpeed(right, dir));
        Flux out{};
        out.D = 0.5f * (fl.D + fr.D) - 0.5f * a * (ur.D - ul.D);
        out.S = 0.5f * (fl.S + fr.S) - 0.5f * a * (ur.S - ul.S);
        out.tau = 0.5f * (fl.tau + fr.tau) - 0.5f * a * (ur.tau - ul.tau);
        out.B = 0.5f * (fl.B + fr.B) - 0.5f * a * (ur.B - ul.B);
        return out;
    }

    static float maxSignalSpeed(const Primitive& p, int dir) {
        const float rho = std::max(p.rho, 1.0e-12f);
        const float press = HarmState::pressure(p);
        const float bsq = glm::dot(p.B, p.B);
        const float h = std::max(rho + p.u + press + bsq, rho);
        const float cs2 = std::clamp(kAdiabaticGamma * press / std::max(h, 1.0e-8f), 0.0f, 0.66f);
        const float va2 = std::clamp(bsq / std::max(h, 1.0e-8f), 0.0f, 0.95f);
        const float cf = std::sqrt(std::clamp(cs2 + va2 - cs2 * va2, 0.0f, 0.98f));
        return std::clamp(std::abs(p.v[dir]) + cf, 1.0e-4f, 0.999f);
    }

    static Primitive reconstructMc(const Primitive& left, const Primitive& center, const Primitive& right, float side) {
        Primitive slope{};
        slope.rho = mc(center.rho - left.rho, right.rho - center.rho);
        slope.u = mc(center.u - left.u, right.u - center.u);
        slope.v = glm::vec3(
            mc(center.v.x - left.v.x, right.v.x - center.v.x),
            mc(center.v.y - left.v.y, right.v.y - center.v.y),
            mc(center.v.z - left.v.z, right.v.z - center.v.z));
        slope.B = glm::vec3(
            mc(center.B.x - left.B.x, right.B.x - center.B.x),
            mc(center.B.y - left.B.y, right.B.y - center.B.y),
            mc(center.B.z - left.B.z, right.B.z - center.B.z));

        Primitive out{};
        out.rho = center.rho + side * slope.rho;
        out.u = center.u + side * slope.u;
        out.v = center.v + side * slope.v;
        out.B = center.B + side * slope.B;
        return HarmState::sanitize(out);
    }

private:
    static float minmod(float a, float b) {
        if (a * b <= 0.0f) return 0.0f;
        return (a > 0.0f) ? std::min(a, b) : std::max(a, b);
    }

    static float mc(float a, float b) {
        return minmod(0.5f * (a + b), minmod(2.0f * a, 2.0f * b));
    }
};

} // namespace harm
