#pragma once

#include <algorithm>
#include <cstddef>

namespace harm {

struct Config {
    int maxGrid = 96;
    int radialN = 64;
    int thetaN = 32;
    int phiN = 64;
    float rin = 1.85f;
    float rout = 42.0f;
    float dt = 0.02f;
    int substeps = 12;
    int viewMode = 5;
    int initialData = 0;
    float spin = 0.7f;
    float magneticLoop = 0.055f;
    float rhoFloor = 1e-5f;
    float uFloor = 1e-6f;
    float colorScale = 1.2f;
    bool enableGravity = false;
    int lensingMode = 0;
    bool highOrder = false;
    bool paused = false;
    bool useGpu = true;
    bool liveGpuDiagnostics = true;
    int readbackInterval = 12;

    void setMaxGrid(int size) {
        maxGrid = std::clamp(size, 32, 96);
        radialN = std::clamp(size, 32, maxGrid);
        thetaN = std::clamp(size / 2, 16, maxGrid);
        phiN = std::clamp(size, 16, maxGrid);
    }

    void clamp() {
        radialN = std::clamp(radialN, 32, maxGrid);
        thetaN = std::clamp(thetaN, 16, maxGrid);
        phiN = std::clamp(phiN, 16, maxGrid);
        substeps = std::clamp(substeps, 1, 12);
        dt = std::clamp(dt, 0.00005f, 0.03f);
        rin = std::max(rin, 1.05f);
        rout = std::max(rout, rin + 4.0f);
        spin = std::clamp(spin, -0.98f, 0.98f);
        rhoFloor = std::max(rhoFloor, 1e-8f);
        uFloor = std::max(uFloor, 1e-9f);
        viewMode = std::clamp(viewMode, 0, 11);
        initialData = std::clamp(initialData, 0, 1);
        lensingMode = std::clamp(lensingMode, 0, 2);
        readbackInterval = std::clamp(readbackInterval, 1, 120);
    }

    size_t cellCount() const {
        return static_cast<size_t>(radialN) * static_cast<size_t>(thetaN) * static_cast<size_t>(phiN);
    }

    size_t maxCellCount() const {
        return static_cast<size_t>(maxGrid) * static_cast<size_t>(maxGrid / 2) * static_cast<size_t>(maxGrid);
    }
};

} // namespace harm
