#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "harm_config.h"
#include "harm_types.h"

namespace harm {

struct StateNorms {
    float rhoL1 = 0.0f;
    float rhoL2 = 0.0f;
    float rhoLinf = 0.0f;
    float uL1 = 0.0f;
    float magneticL1 = 0.0f;
    float velocityL1 = 0.0f;
};

class StateNormSampler {
public:
    static StateNorms compare(const Config& cfg, const std::vector<float>& a, const std::vector<float>& b) {
        StateNorms out{};
        const size_t floats = packedFloatCount(cfg.cellCount());
        if (a.size() < floats || b.size() < floats) {
            return out;
        }

        float rhoL2 = 0.0f;
        for (size_t i = 0; i < cfg.cellCount(); ++i) {
            const size_t base = i * 12;
            const float rhoScale = std::max(std::abs(a[base + 0]), cfg.rhoFloor);
            const float uScale = std::max(std::abs(a[base + 1]), cfg.uFloor);
            const float rhoErr = std::abs(b[base + 0] - a[base + 0]) / rhoScale;
            const float uErr = std::abs(b[base + 1] - a[base + 1]) / uScale;
            const float vErr = std::abs(b[base + 2] - a[base + 2]) +
                std::abs(b[base + 3] - a[base + 3]) +
                std::abs(b[base + 4] - a[base + 4]);
            const float bMagA = std::sqrt(a[base + 5] * a[base + 5] + a[base + 6] * a[base + 6] + a[base + 7] * a[base + 7]);
            const float bMagB = std::sqrt(b[base + 5] * b[base + 5] + b[base + 6] * b[base + 6] + b[base + 7] * b[base + 7]);
            const float bErr = std::abs(bMagB - bMagA) / std::max(bMagA, 1.0e-12f);

            out.rhoL1 += rhoErr;
            rhoL2 += rhoErr * rhoErr;
            out.rhoLinf = std::max(out.rhoLinf, rhoErr);
            out.uL1 += uErr;
            out.velocityL1 += vErr;
            out.magneticL1 += bErr;
        }

        const float invCells = 1.0f / std::max(static_cast<float>(cfg.cellCount()), 1.0f);
        out.rhoL1 *= invCells;
        out.rhoL2 = std::sqrt(rhoL2 * invCells);
        out.uL1 *= invCells;
        out.velocityL1 *= invCells;
        out.magneticL1 *= invCells;
        return out;
    }
};

} // namespace harm
