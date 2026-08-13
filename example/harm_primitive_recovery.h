#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "harm_state.h"

namespace harm {

struct RecoveryResult {
    Primitive primitive{};
    int iterations = 0;
    bool usedEntropyFallback = false;
    bool failed = false;
};

// Deliberately conservative reference inversion.  It solves the five
// hydrodynamic primitives against the same covariant conserved map used by
// HarmState.  Production GRMHD codes use specialized 1D/2D schemes for speed;
// this small dense Newton solve is easier to audit and is therefore useful as
// the CPU correctness oracle for the GPU implementation.
class PrimitiveRecovery {
public:
    static RecoveryResult recover(const Conserved& target, const Primitive& guess,
                                  const Metric& metric, float rhoFloor, float uFloor,
                                  float entropyConserved = -1.0f) {
        RecoveryResult result{};
        Primitive p = enforceTimelike(guess, metric, rhoFloor, uFloor);
        p.B = target.B / std::max(metric.sqrtMinusG, 1.0e-20f);

        std::array<double, 5> x = encode(p, rhoFloor, uFloor);
        const std::array<double, 5> scale = conservedScale(target);
        bool converged = false;

        for (int iteration = 0; iteration < 14; ++iteration) {
            p = decode(x, p.B, metric, rhoFloor, uFloor);
            const std::array<double, 5> residual = normalizedResidual(p, target, metric, scale);
            const double norm = maxAbs(residual);
            const double conservedError = rawRelativeResidual(p, target, metric);
            result.iterations = iteration + 1;
            if (norm < 2.0e-7 || conservedError < 2.0e-6) {
                converged = true;
                break;
            }

            double jacobian[5][5] = {};
            for (int column = 0; column < 5; ++column) {
                std::array<double, 5> xp = x;
                std::array<double, 5> xm = x;
                const double h = (column < 2) ? 2.0e-5 : 2.0e-6;
                xp[column] += h;
                xm[column] -= h;
                const Primitive pp = decode(xp, p.B, metric, rhoFloor, uFloor);
                const Primitive pm = decode(xm, p.B, metric, rhoFloor, uFloor);
                const auto rp = normalizedResidual(pp, target, metric, scale);
                const auto rm = normalizedResidual(pm, target, metric, scale);
                for (int row = 0; row < 5; ++row) jacobian[row][column] = (rp[row] - rm[row]) / (2.0 * h);
            }

            std::array<double, 5> rhs{};
            for (int i = 0; i < 5; ++i) rhs[i] = -residual[i];
            std::array<double, 5> step{};
            if (!solve5(jacobian, rhs, step)) break;

            // Backtracking keeps every accepted iterate physical and reduces
            // the true conserved residual, including in magnetized cells.
            bool accepted = false;
            double damping = 1.0;
            for (int trial = 0; trial < 9; ++trial) {
                std::array<double, 5> candidate = x;
                for (int i = 0; i < 5; ++i) {
                    const double cap = (i < 2) ? 2.0 : 0.35;
                    candidate[i] += damping * std::clamp(step[i], -cap, cap);
                }
                const Primitive pc = decode(candidate, p.B, metric, rhoFloor, uFloor);
                const double candidateNorm = maxAbs(normalizedResidual(pc, target, metric, scale));
                if (std::isfinite(candidateNorm) && candidateNorm < norm) {
                    x = candidate;
                    accepted = true;
                    break;
                }
                damping *= 0.5;
            }
            if (!accepted) break;
        }

        p = decode(x, p.B, metric, rhoFloor, uFloor);
        const double finalNorm = maxAbs(normalizedResidual(p, target, metric, scale));
        converged = converged || finalNorm < 2.0e-5 || rawRelativeResidual(p, target, metric) < 5.0e-4;
        if (!converged || !finite(p)) {
            result.usedEntropyFallback = true;
            p = enforceTimelike(guess, metric, rhoFloor, uFloor);
            p.B = target.B / std::max(metric.sqrtMinusG, 1.0e-20f);
            if (entropyConserved > 0.0f && target.D > 0.0f) {
                const float entropyConstant = entropyConserved / target.D;
                p.u = std::max(HarmState::internalEnergyFromEntropy(p.rho, entropyConstant), uFloor);
                result.failed = false;
                result.primitive = p;
                return result;
            }
            // Atmosphere cells are deliberately reset by the floor policy and
            // are not primitive-inversion failures.  Count a failure only when
            // a resolved fluid cell cannot be inverted.
            result.failed = !finite(p) || guess.rho > 8.0f * rhoFloor || guess.u > 8.0f * uFloor;
        }
        result.primitive = p;
        return result;
    }

    static Primitive enforceTimelike(const Primitive& input, const Metric& metric,
                                     float rhoFloor, float uFloor) {
        Primitive p = input;
        p.rho = std::max(p.rho, rhoFloor);
        p.u = std::max(p.u, uFloor);
        // Coordinate velocity of the normal observer is -beta^i.  Scaling the
        // displacement from that observer preserves frame dragging while
        // enforcing a finite Lorentz factor.
        glm::vec3 normal(-metric.beta[0], -metric.beta[1], -metric.beta[2]);
        glm::vec3 delta = p.v - normal;
        auto norm = [&](const glm::vec3& v) {
            double q = metric.gcov[0][0];
            for (int i = 0; i < 3; ++i) {
                q += 2.0 * metric.gcov[0][i + 1] * v[i];
                for (int j = 0; j < 3; ++j) q += metric.gcov[i + 1][j + 1] * v[i] * v[j];
            }
            return q;
        };
        const double targetNorm = -metric.alpha * metric.alpha / (50.0 * 50.0);
        if (!std::isfinite(norm(p.v)) || norm(p.v) > targetNorm) {
            double lo = 0.0, hi = 1.0;
            for (int i = 0; i < 48; ++i) {
                const double mid = 0.5 * (lo + hi);
                if (norm(normal + static_cast<float>(mid) * delta) < targetNorm) lo = mid;
                else hi = mid;
            }
            p.v = normal + static_cast<float>(0.995 * lo) * delta;
        }
        return p;
    }

private:
    static std::array<double, 5> encode(const Primitive& p, float rhoFloor, float uFloor) {
        return {std::log(std::max(p.rho, rhoFloor)), std::log(std::max(p.u, uFloor)), p.v.x, p.v.y, p.v.z};
    }

    static Primitive decode(const std::array<double, 5>& x, const glm::vec3& B,
                            const Metric& metric, float rhoFloor, float uFloor) {
        Primitive p{};
        p.rho = std::max(static_cast<float>(std::exp(std::clamp(x[0], -40.0, 40.0))), rhoFloor);
        p.u = std::max(static_cast<float>(std::exp(std::clamp(x[1], -40.0, 40.0))), uFloor);
        p.v = glm::vec3(static_cast<float>(x[2]), static_cast<float>(x[3]), static_cast<float>(x[4]));
        p.B = B;
        return enforceTimelike(p, metric, rhoFloor, uFloor);
    }

    static std::array<double, 5> conservedScale(const Conserved& u) {
        return {std::max(std::abs(static_cast<double>(u.D)), 1.0e-10),
                std::max(std::abs(static_cast<double>(u.S.x)), std::abs(static_cast<double>(u.D)) * 1.0e-5 + 1.0e-10),
                std::max(std::abs(static_cast<double>(u.S.y)), std::abs(static_cast<double>(u.D)) * 1.0e-5 + 1.0e-10),
                std::max(std::abs(static_cast<double>(u.S.z)), std::abs(static_cast<double>(u.D)) * 1.0e-5 + 1.0e-10),
                std::max(std::abs(static_cast<double>(u.tau)), std::abs(static_cast<double>(u.D)) * 1.0e-5 + 1.0e-10)};
    }

    static std::array<double, 5> normalizedResidual(const Primitive& p, const Conserved& target,
                                                    const Metric& metric, const std::array<double, 5>& scale) {
        const Conserved u = HarmState::primitiveToConserved(p, metric);
        return {(u.D - target.D) / scale[0], (u.S.x - target.S.x) / scale[1],
                (u.S.y - target.S.y) / scale[2], (u.S.z - target.S.z) / scale[3],
                (u.tau - target.tau) / scale[4]};
    }

    static double rawRelativeResidual(const Primitive& p, const Conserved& target, const Metric& metric) {
        const Conserved u = HarmState::primitiveToConserved(p, metric);
        const double numerator = std::abs(u.D - target.D) + glm::length(u.S - target.S) + std::abs(u.tau - target.tau);
        const double denominator = std::abs(target.D) + glm::length(target.S) + std::abs(target.tau) + 1.0e-20;
        return numerator / denominator;
    }

    static double maxAbs(const std::array<double, 5>& a) {
        double out = 0.0;
        for (double v : a) out = std::max(out, std::abs(v));
        return out;
    }

    static bool solve5(double a[5][5], std::array<double, 5> b, std::array<double, 5>& x) {
        for (int col = 0; col < 5; ++col) {
            int pivot = col;
            for (int row = col + 1; row < 5; ++row) if (std::abs(a[row][col]) > std::abs(a[pivot][col])) pivot = row;
            if (std::abs(a[pivot][col]) < 1.0e-12) return false;
            if (pivot != col) {
                for (int j = col; j < 5; ++j) std::swap(a[col][j], a[pivot][j]);
                std::swap(b[col], b[pivot]);
            }
            const double inv = 1.0 / a[col][col];
            for (int j = col; j < 5; ++j) a[col][j] *= inv;
            b[col] *= inv;
            for (int row = 0; row < 5; ++row) {
                if (row == col) continue;
                const double f = a[row][col];
                for (int j = col; j < 5; ++j) a[row][j] -= f * a[col][j];
                b[row] -= f * b[col];
            }
        }
        x = b;
        return true;
    }

    static bool finite(const Primitive& p) {
        return std::isfinite(p.rho) && std::isfinite(p.u) &&
               std::isfinite(p.v.x) && std::isfinite(p.v.y) && std::isfinite(p.v.z) &&
               std::isfinite(p.B.x) && std::isfinite(p.B.y) && std::isfinite(p.B.z);
    }
};

} // namespace harm
