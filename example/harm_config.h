#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>

namespace harm {

struct Config {
    static constexpr int kSingleBufferMaxGrid = 128;
    static constexpr int kDefaultMaxGrid = 196;
    static constexpr int kTiledMaxGrid = 512;
    static constexpr int kMaxPhiSlabs = 4;
    static constexpr int kMaxSubstepsPerFrame = 240;

    int maxGrid = kDefaultMaxGrid;
    // Default to the literal low-resolution Porth comparison tier.  This is a
    // true cubic mesh, not the older N x N/2 x N display-oriented shortcut.
    int radialN = 96;
    int thetaN = 96;
    int phiN = 96;
    float rin = 1.10f;
    float rout = 50.0f;
    float dt = 0.0005f;
    int substeps = 1;
    int viewMode = 5;
    int initialData = 2;
    float spin = 0.9375f;
    float magneticLoop = 0.055f;
    float rhoFloor = 1e-5f;
    float uFloor = 1e-7f;
    float colorScale = 1.2f;
    bool enableGravity = false;
    int lensingMode = 0;
    bool highOrder = true;
    bool paused = false;
    bool useGpu = true;
    bool liveGpuDiagnostics = true;
    int readbackInterval = 12;

    void setMaxGrid(int size) {
        maxGrid = std::clamp(size, 32, kTiledMaxGrid);
        radialN = std::clamp(size, 32, maxGrid);
        thetaN = std::clamp(size, 16, maxGrid);
        phiN = std::clamp(size, 16, maxGrid);
    }

    void clamp() {
        spin = std::clamp(spin, -0.98f, 0.98f);
        const float horizon = 1.0f + std::sqrt(std::max(1.0f - spin * spin, 0.0f));
        radialN = std::clamp(radialN, 32, maxGrid);
        thetaN = std::clamp(thetaN, 16, maxGrid);
        phiN = std::clamp(phiN, 16, maxGrid);
        substeps = std::clamp(substeps, 1, kMaxSubstepsPerFrame);
        dt = std::clamp(dt, 0.00005f, 0.03f);
        // Kerr-Schild coordinates are horizon penetrating.  Keep the excision
        // surface inside the outer horizon so no inner boundary condition can
        // causally contaminate the exterior solution.
        rin = std::max(rin, horizon * 0.80f);
        rout = std::max(rout, rin + 4.0f);
        rhoFloor = std::max(rhoFloor, 1e-8f);
        uFloor = std::max(uFloor, 1e-9f);
        viewMode = std::clamp(viewMode, 0, 11);
        initialData = std::clamp(initialData, 0, 2);
        lensingMode = std::clamp(lensingMode, 0, 2);
        readbackInterval = std::clamp(readbackInterval, 1, 120);
    }

    size_t cellCount() const {
        return static_cast<size_t>(radialN) * static_cast<size_t>(thetaN) * static_cast<size_t>(phiN);
    }

    size_t maxCellCount() const {
        return static_cast<size_t>(maxGrid) * static_cast<size_t>(maxGrid) * static_cast<size_t>(maxGrid);
    }

    // The Porth et al. comparison permits code-dependent atmosphere schemes.
    // These coefficients follow the common r^-3/2 density and r^-5/2 internal
    // energy prescription used by several participating finite-volume codes.
    float rhoFloorAt(float r) const {
        return rhoFloor * std::pow(std::max(r, 1.0f), -1.5f);
    }

    float uFloorAt(float r) const {
        return uFloor * std::pow(std::max(r, 1.0f), -2.5f);
    }
};

} // namespace harm
