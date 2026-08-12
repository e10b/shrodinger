#pragma once

#include <algorithm>
#include <cmath>

#include "harm_state.h"

namespace harm {

struct RecoveryResult {
    Primitive primitive{};
    int iterations = 0;
    bool usedEntropyFallback = false;
    bool failed = false;
};

class PrimitiveRecovery {
public:
    static RecoveryResult recover(const Conserved& u, const Primitive& guess, const Metric& metric, float rhoFloor, float uFloor) {
        RecoveryResult result{};
        Primitive p = HarmState::sanitize(guess);
        const glm::vec3 B = u.B / std::max(metric.sqrtMinusG, 1.0e-8f);
        const float D = std::max(u.D / std::max(metric.sqrtMinusG, 1.0e-8f), rhoFloor);
        const glm::vec3 S = u.S / std::max(metric.sqrtMinusG, 1.0e-8f);
        const float tau = std::max(u.tau / std::max(metric.sqrtMinusG, 1.0e-8f), uFloor);
        const float bsq = glm::dot(B, B);
        const float s2 = glm::dot(S, S);

        float W = std::max(D + tau + HarmState::pressure(p) + bsq, D + uFloor + bsq);
        bool converged = false;
        for (int i = 0; i < 24; ++i) {
            const float q = W + bsq;
            const float v2 = std::clamp(s2 / std::max(q * q, 1.0e-12f), 0.0f, 0.92f);
            const float gamma = 1.0f / std::sqrt(std::max(1.0f - v2, 1.0e-8f));
            const float rho = std::max(D / gamma, rhoFloor);
            const float internal = std::max((W / (gamma * gamma) - rho) / kAdiabaticGamma, uFloor);
            const float pressure = (kAdiabaticGamma - 1.0f) * internal;
            const float f = W - pressure + 0.5f * bsq + 0.5f * s2 / std::max(q * q, 1.0e-12f) * W - (tau + D);
            const float eps = std::max(1.0e-3f * W, 1.0e-6f);
            const float fp = residualAt(W + eps, D, tau, s2, bsq, rhoFloor, uFloor);
            const float fm = residualAt(std::max(W - eps, D + uFloor + bsq), D, tau, s2, bsq, rhoFloor, uFloor);
            const float dfdW = (fp - fm) / std::max((W + eps) - std::max(W - eps, D + uFloor + bsq), 1.0e-8f);
            const float step = f / ((std::abs(dfdW) > 1.0e-8f) ? dfdW : ((dfdW < 0.0f) ? -1.0e-8f : 1.0e-8f));
            const float dW = std::max(0.35f * W, 1.0e-8f);
            W -= std::clamp(step, -dW, dW);
            W = std::max(W, D + uFloor + bsq);
            result.iterations = i + 1;
            if (std::abs(f) < 1.0e-6f * std::max(tau + D + bsq, 1.0f)) {
                converged = true;
                break;
            }
        }

        p.B = B;
        p.v = S / std::max(W + bsq, 1.0e-8f);
        const float v2 = glm::dot(p.v, p.v);
        if (v2 > 0.92f) {
            p.v *= std::sqrt(0.92f / std::max(v2, 1.0e-8f));
        }
        const float gamma = HarmState::lorentzFactor(p);
        p.rho = std::max(D / gamma, rhoFloor);
        p.u = std::max((W / (gamma * gamma) - p.rho) / kAdiabaticGamma, uFloor);

        if (!converged || !finite(p)) {
            result.usedEntropyFallback = true;
            p = HarmState::sanitize(guess);
            p.rho = std::max(D / std::max(HarmState::lorentzFactor(p), 1.0f), rhoFloor);
            p.u = std::max(p.u, uFloor);
            p.B = B;
            result.failed = !finite(p);
        }

        result.primitive = HarmState::sanitize(p);
        return result;
    }

private:
    static float residualAt(float W, float D, float tau, float s2, float bsq, float rhoFloor, float uFloor) {
        const float q = W + bsq;
        const float v2 = std::clamp(s2 / std::max(q * q, 1.0e-12f), 0.0f, 0.92f);
        const float gamma = 1.0f / std::sqrt(std::max(1.0f - v2, 1.0e-8f));
        const float rho = std::max(D / gamma, rhoFloor);
        const float internal = std::max((W / (gamma * gamma) - rho) / kAdiabaticGamma, uFloor);
        const float pressure = (kAdiabaticGamma - 1.0f) * internal;
        return W - pressure + 0.5f * bsq + 0.5f * v2 * W - (tau + D);
    }

    static bool finite(const Primitive& p) {
        return std::isfinite(p.rho) && std::isfinite(p.u) &&
            std::isfinite(p.v.x) && std::isfinite(p.v.y) && std::isfinite(p.v.z) &&
            std::isfinite(p.B.x) && std::isfinite(p.B.y) && std::isfinite(p.B.z);
    }
};

} // namespace harm
