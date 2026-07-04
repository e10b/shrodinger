#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "harm_config.h"
#include "harm_types.h"

namespace harm {

class Grid {
public:
    std::vector<float> packed;
    std::vector<float> readback;

    void resize(const Config& cfg) {
        packed.assign(packedFloatCount(cfg.cellCount()), 0.0f);
    }

    size_t index(const Config& cfg, int ir, int it, int ip) const {
        const int r = std::clamp(ir, 0, cfg.radialN - 1);
        const int t = std::clamp(it, 0, cfg.thetaN - 1);
        int p = ip % cfg.phiN;
        if (p < 0) p += cfg.phiN;
        return (static_cast<size_t>(p) * static_cast<size_t>(cfg.thetaN) + static_cast<size_t>(t))
             * static_cast<size_t>(cfg.radialN) + static_cast<size_t>(r);
    }

    float* cell(size_t idx) {
        return packed.data() + idx * 12;
    }

    const float* cell(size_t idx) const {
        return packed.data() + idx * 12;
    }
};

} // namespace harm
