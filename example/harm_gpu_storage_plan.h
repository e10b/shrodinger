#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "webgpu/webgpu.hpp"

#include "harm_config.h"
#include "harm_types.h"

namespace harm {

struct GpuStoragePlan {
    size_t cells = 0;
    size_t bytesPerState = 0;
    size_t maxBindingBytes = 0;
    int phiSlabs = 1;
    int maxPhiPerSlab = 0;
    bool requiresTiling = false;

    size_t pingPongBytes() const {
        return bytesPerState * 2;
    }
};

inline size_t usableStorageBindingBytes(const wgpu::Limits& limits) {
    size_t limit = static_cast<size_t>(limits.maxStorageBufferBindingSize);
    if (limit == 0) {
        return 0;
    }
    return limit;
}

inline GpuStoragePlan makeGpuStoragePlan(const Config& cfg, const wgpu::Limits& limits) {
    GpuStoragePlan plan{};
    plan.cells = cfg.maxCellCount();
    plan.bytesPerState = packedByteCount(plan.cells);
    plan.maxBindingBytes = usableStorageBindingBytes(limits);
    if (plan.maxBindingBytes == 0) {
        return plan;
    }

    const int thetaMax = std::max(cfg.maxGrid / 2, 1);
    const size_t bytesPerPhiPlane =
        static_cast<size_t>(cfg.maxGrid) * static_cast<size_t>(thetaMax) * sizeof(PackedPrim);
    plan.maxPhiPerSlab = static_cast<int>(std::max<size_t>(1, plan.maxBindingBytes / bytesPerPhiPlane));
    plan.phiSlabs = std::max(1, (cfg.maxGrid + plan.maxPhiPerSlab - 1) / plan.maxPhiPerSlab);
    plan.requiresTiling = plan.bytesPerState > plan.maxBindingBytes;
    return plan;
}

} // namespace harm
