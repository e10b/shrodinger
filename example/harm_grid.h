#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "harm_config.h"
#include "harm_geometry.h"
#include "harm_types.h"

namespace harm {

class Grid {
public:
    std::vector<float> packed;
    std::vector<float> readback;
    // Densitized magnetic flux on each cell's lower r/theta/phi face.
    // Populated by GPU readback and persisted by production checkpoints.
    std::vector<float> faceFlux;

    void resize(const Config& cfg) {
        packed.assign(packedFloatCount(cfg.cellCount()), 0.0f);
        faceFlux.clear();
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

    void initializeFaceFlux(const Config& cfg) {
        faceFlux.assign(3u * cfg.cellCount(), 0.0f);
        auto densitized = [&](int ir, int it, int ip, int component) {
            const size_t idx = index(cfg, ir, it, ip);
            const CellGeometry geom = HarmGeometry::cell(cfg, ir, it);
            return geom.metric.sqrtMinusG * packed[idx * 12u + 5u + static_cast<size_t>(component)];
        };
        for (int ip = 0; ip < cfg.phiN; ++ip) for (int it = 0; it < cfg.thetaN; ++it) for (int ir = 0; ir < cfg.radialN; ++ir) {
            const size_t idx = index(cfg, ir, it, ip);
            faceFlux[idx * 3u] = 0.5f * (densitized(ir, it, ip, 0) + densitized(ir - 1, it, ip, 0));
            faceFlux[idx * 3u + 1u] = it == 0 ? 0.0f : 0.5f * (densitized(ir, it, ip, 1) + densitized(ir, it - 1, ip, 1));
            faceFlux[idx * 3u + 2u] = 0.5f * (densitized(ir, it, ip, 2) + densitized(ir, it, ip - 1, 2));
        }
        reconstructCellCenteredB(cfg);
    }

    // Initialize the poloidal face field as one discrete curl of a single
    // edge-centered A_phi.  The divergence operator below then telescopes to
    // roundoff by construction, unlike averaging an already differentiated
    // cell-centered field onto faces.
    void initializeFaceFluxFromAphi(const Config& cfg, const std::vector<float>& cellAphi,
                                    float scale = 1.0f) {
        if (cellAphi.size() != cfg.cellCount()) {
            initializeFaceFlux(cfg);
            return;
        }
        faceFlux.assign(3u * cfg.cellCount(), 0.0f);
        const float dtheta = HarmGeometry::dtheta(cfg);
        auto edgeA = [&](int ir, int it, int ip) {
            // A_phi vanishes on the polar axis. Radial ghost values use the
            // nearest cell, giving a zero-normal-gradient radial boundary.
            if (it <= 0 || it >= cfg.thetaN) return 0.0f;
            return 0.25f * scale * (
                cellAphi[index(cfg, ir - 1, it - 1, ip)] +
                cellAphi[index(cfg, ir,     it - 1, ip)] +
                cellAphi[index(cfg, ir - 1, it,     ip)] +
                cellAphi[index(cfg, ir,     it,     ip)]);
        };
        for (int ip = 0; ip < cfg.phiN; ++ip) for (int it = 0; it < cfg.thetaN; ++it) for (int ir = 0; ir < cfg.radialN; ++ir) {
            const size_t idx = index(cfg, ir, it, ip);
            const float dr = HarmGeometry::cell(cfg, ir, it).radialLength();
            // Lower-face densitized fluxes. Their forward differences form
            // d_r d_theta A_phi - d_theta d_r A_phi exactly.
            faceFlux[idx * 3u] = (edgeA(ir, it + 1, ip) - edgeA(ir, it, ip)) /
                std::max(dtheta, 1.0e-8f);
            faceFlux[idx * 3u + 1u] = it == 0 ? 0.0f :
                -(edgeA(ir + 1, it, ip) - edgeA(ir, it, ip)) /
                 std::max(dr, 1.0e-8f);
            faceFlux[idx * 3u + 2u] = 0.0f;
        }
        reconstructCellCenteredB(cfg);
    }

    void reconstructCellCenteredB(const Config& cfg) {
        if (faceFlux.size() != 3u * cfg.cellCount()) return;
        auto face = [&](int ir, int it, int ip, int component) {
            if (component == 1 && (it <= 0 || it >= cfg.thetaN)) return 0.0f;
            return faceFlux[index(cfg, ir, it, ip) * 3u + static_cast<size_t>(component)];
        };
        for (int ip = 0; ip < cfg.phiN; ++ip) for (int it = 0; it < cfg.thetaN; ++it) for (int ir = 0; ir < cfg.radialN; ++ir) {
            const size_t idx = index(cfg, ir, it, ip);
            const CellGeometry geom = HarmGeometry::cell(cfg, ir, it);
            const float inv = 1.0f / std::max(geom.metric.sqrtMinusG, 1.0e-20f);
            packed[idx * 12u + 5u] = 0.5f * (face(ir, it, ip, 0) + face(ir + 1, it, ip, 0)) * inv;
            packed[idx * 12u + 6u] = 0.5f * (face(ir, it, ip, 1) + face(ir, it + 1, ip, 1)) * inv;
            packed[idx * 12u + 7u] = 0.5f * (face(ir, it, ip, 2) + face(ir, it, ip + 1, 2)) * inv;
        }
    }

};

} // namespace harm
