#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "webgpu/webgpu.hpp"

#include "harm_config.h"
#include "harm_types.h"

namespace harm {

struct GpuStoragePlan {
    size_t cells = 0;
    size_t bytesPerState = 0;
    size_t bytesPerSlabState = 0;
    size_t maxBindingBytes = 0;
    int phiSlabs = 1;
    int maxPhiPerSlab = 0;
    int phiSplit = 0;
    int slabPhi = 0;
    bool requiresTiling = false;
    std::vector<int> slabPhiCounts;

    size_t pingPongBytes() const {
        return bytesPerSlabState * static_cast<size_t>(phiSlabs) * 2u;
    }
};

inline size_t usableStorageBindingBytes(const wgpu::Limits& limits) {
    size_t limit = static_cast<size_t>(limits.maxStorageBufferBindingSize);
    if (limit == 0) {
        return 0;
    }
    constexpr size_t kTargetStorageBindingBytes = 512ull * 1024ull * 1024ull;
    return std::min(limit, kTargetStorageBindingBytes);
}

inline GpuStoragePlan makeGpuStoragePlan(const Config& cfg, const wgpu::Limits& limits) {
    GpuStoragePlan plan{};
    plan.cells = cfg.cellCount();
    plan.bytesPerState = gpuPackedByteCount(plan.cells);
    plan.maxBindingBytes = usableStorageBindingBytes(limits);
    if (plan.maxBindingBytes == 0) {
        return plan;
    }

    const int radialN = std::max(cfg.radialN, 1);
    const int thetaN = std::max(cfg.thetaN, 1);
    const int phiN = std::max(cfg.phiN, 1);
    const size_t bytesPerPhiPlane =
        gpuPackedByteCount(static_cast<size_t>(radialN) * static_cast<size_t>(thetaN));
    plan.maxPhiPerSlab = static_cast<int>(std::max<size_t>(1, plan.maxBindingBytes / bytesPerPhiPlane));
    plan.requiresTiling = plan.bytesPerState > plan.maxBindingBytes;
    plan.phiSlabs = std::clamp(
        static_cast<int>((static_cast<size_t>(phiN) + static_cast<size_t>(plan.maxPhiPerSlab) - 1u) /
                         static_cast<size_t>(plan.maxPhiPerSlab)),
        1,
        Config::kMaxPhiSlabs);
    plan.slabPhi = std::max(1, (phiN + plan.phiSlabs - 1) / plan.phiSlabs);
    plan.phiSplit = plan.slabPhi;
    plan.slabPhiCounts.clear();
    for (int slab = 0; slab < plan.phiSlabs; ++slab) {
        const int start = slab * plan.slabPhi;
        plan.slabPhiCounts.push_back(std::clamp(phiN - start, 0, plan.slabPhi));
    }
    plan.bytesPerSlabState = static_cast<size_t>(plan.slabPhi) * bytesPerPhiPlane;
    if (plan.bytesPerSlabState > plan.maxBindingBytes || plan.phiSlabs > Config::kMaxPhiSlabs) {
        plan.requiresTiling = true;
        plan.phiSlabs = Config::kMaxPhiSlabs + 1;
    }
    return plan;
}

} // namespace harm
