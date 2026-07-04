#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <glm/glm.hpp>

#include "harm_types.h"

namespace harm {

struct FourVector {
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float& operator[](int i) { return v[i]; }
    float operator[](int i) const { return v[i]; }
};

struct Metric {
    float gcov[4][4] = {};
    float gcon[4][4] = {};
    float alpha = 1.0f;
    float beta[3] = {};
    float sqrtMinusG = 1.0f;
    float r = 1.0f;
    float theta = 0.5f * kPi;
    float spin = 0.0f;
};

struct MetricDerivatives {
    float dr[4][4] = {};
    float dtheta[4][4] = {};
};

class KerrSchild {
public:
    static float horizonRadius(float spin) {
        const float a = std::clamp(spin, -0.999f, 0.999f);
        return 1.0f + std::sqrt(std::max(1.0f - a * a, 0.0f));
    }

    static Metric metric(float r, float theta, float spin) {
        Metric m{};
        m.r = std::max(r, 1.0e-4f);
        m.theta = std::clamp(theta, 1.0e-4f, kPi - 1.0e-4f);
        m.spin = std::clamp(spin, -0.999f, 0.999f);

        const float a = m.spin;
        const float rr = m.r * m.r;
        const float sinTh = std::sin(m.theta);
        const float cosTh = std::cos(m.theta);
        const float sin2 = sinTh * sinTh;
        const float sigma = rr + a * a * cosTh * cosTh;
        const float twoMrOverSigma = 2.0f * m.r / std::max(sigma, 1.0e-8f);

        m.gcov[0][0] = -(1.0f - twoMrOverSigma);
        m.gcov[0][1] = twoMrOverSigma;
        m.gcov[1][0] = m.gcov[0][1];
        m.gcov[0][3] = -twoMrOverSigma * a * sin2;
        m.gcov[3][0] = m.gcov[0][3];
        m.gcov[1][1] = 1.0f + twoMrOverSigma;
        m.gcov[1][3] = -(1.0f + twoMrOverSigma) * a * sin2;
        m.gcov[3][1] = m.gcov[1][3];
        m.gcov[2][2] = sigma;
        m.gcov[3][3] = (rr + a * a + twoMrOverSigma * a * a * sin2) * sin2;

        invert4(m.gcov, m.gcon);
        m.alpha = 1.0f / std::sqrt(std::max(-m.gcon[0][0], 1.0e-8f));
        for (int i = 0; i < 3; ++i) {
            m.beta[i] = m.alpha * m.alpha * m.gcon[0][i + 1];
        }
        m.sqrtMinusG = std::max(sigma * sinTh, 1.0e-8f);
        return m;
    }

    static MetricDerivatives derivatives(float r, float theta, float spin) {
        const float drStep = std::max(1.0e-4f, 1.0e-3f * std::max(r, 1.0f));
        const float dtStep = 1.0e-4f;
        const Metric rp = metric(r + drStep, theta, spin);
        const Metric rm = metric(std::max(r - drStep, horizonRadius(spin) * 1.0001f), theta, spin);
        const Metric tp = metric(r, std::min(theta + dtStep, kPi - 1.0e-4f), spin);
        const Metric tm = metric(r, std::max(theta - dtStep, 1.0e-4f), spin);

        MetricDerivatives d{};
        const float invDr = 1.0f / std::max((r + drStep) - std::max(r - drStep, horizonRadius(spin) * 1.0001f), 1.0e-8f);
        const float invDt = 1.0f / std::max(std::min(theta + dtStep, kPi - 1.0e-4f) - std::max(theta - dtStep, 1.0e-4f), 1.0e-8f);
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                d.dr[mu][nu] = (rp.gcov[mu][nu] - rm.gcov[mu][nu]) * invDr;
                d.dtheta[mu][nu] = (tp.gcov[mu][nu] - tm.gcov[mu][nu]) * invDt;
            }
        }
        return d;
    }

    static float lower(const Metric& m, const FourVector& a, int mu) {
        float out = 0.0f;
        for (int nu = 0; nu < 4; ++nu) {
            out += m.gcov[mu][nu] * a[nu];
        }
        return out;
    }

    static float dot(const Metric& m, const FourVector& a, const FourVector& b) {
        float out = 0.0f;
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                out += m.gcov[mu][nu] * a[mu] * b[nu];
            }
        }
        return out;
    }

    static FourVector normalObserverVelocity(const Metric& m, const glm::vec3& v) {
        const float gamma = lorentzFactor(v);
        FourVector u{};
        u[0] = gamma / std::max(m.alpha, 1.0e-6f);
        for (int i = 0; i < 3; ++i) {
            u[i + 1] = gamma * (v[i] - m.beta[i] / std::max(m.alpha, 1.0e-6f));
        }
        return u;
    }

private:
    static float lorentzFactor(const glm::vec3& v) {
        const float v2 = std::clamp(glm::dot(v, v), 0.0f, 0.999f);
        return 1.0f / std::sqrt(std::max(1.0f - v2, 1.0e-6f));
    }

    static void invert4(const float in[4][4], float out[4][4]) {
        float a[4][8] = {};
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                a[r][c] = in[r][c];
            }
            a[r][r + 4] = 1.0f;
        }

        for (int i = 0; i < 4; ++i) {
            int pivot = i;
            for (int r = i + 1; r < 4; ++r) {
                if (std::abs(a[r][i]) > std::abs(a[pivot][i])) {
                    pivot = r;
                }
            }
            if (pivot != i) {
                for (int c = 0; c < 8; ++c) {
                    std::swap(a[i][c], a[pivot][c]);
                }
            }
            const float div = (std::abs(a[i][i]) > 1.0e-12f) ? a[i][i] : ((a[i][i] < 0.0f) ? -1.0e-12f : 1.0e-12f);
            for (int c = 0; c < 8; ++c) {
                a[i][c] /= div;
            }
            for (int r = 0; r < 4; ++r) {
                if (r == i) continue;
                const float f = a[r][i];
                for (int c = 0; c < 8; ++c) {
                    a[r][c] -= f * a[i][c];
                }
            }
        }

        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out[r][c] = a[r][c + 4];
            }
        }
    }
};

} // namespace harm
