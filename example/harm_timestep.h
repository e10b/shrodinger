#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "harm_config.h"
#include "harm_flux.h"
#include "harm_geometry.h"
#include "harm_grid.h"
#include "harm_state.h"

namespace harm {

class HarmTimeStepper {
public:
    static float stableDt(const Config& cfg, const std::vector<float>& packed, float cflTarget = 0.42f) {
        if (packed.size() < packedFloatCount(cfg.cellCount())) {
            return std::clamp(cfg.dt, 0.00005f, 0.03f);
        }

        float dt = std::clamp(cfg.dt, 0.00005f, 0.03f);
        for (int ip = 0; ip < cfg.phiN; ++ip) {
            for (int it = 0; it < cfg.thetaN; ++it) {
                const CellGeometry geom = HarmGeometry::cell(cfg, 0, it);
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const CellGeometry cell = HarmGeometry::cell(cfg, ir, it);
                    const size_t idx = (static_cast<size_t>(ip) * static_cast<size_t>(cfg.thetaN) + static_cast<size_t>(it)) *
                        static_cast<size_t>(cfg.radialN) + static_cast<size_t>(ir);
                    const Primitive p = HarmState::fromPacked(&packed[idx * 12], cell.r, cell.theta);
                    const float speed = std::max({
                        HarmFlux::maxSignalSpeed(p, cell.metric, 0) * cell.radialLength() / std::max(cell.dr, 1.0e-8f),
                        HarmFlux::maxSignalSpeed(p, cell.metric, 1) * cell.thetaLength() / std::max(cell.dtheta, 1.0e-8f),
                        HarmFlux::maxSignalSpeed(p, cell.metric, 2) * cell.phiLength() / std::max(cell.dphi, 1.0e-8f),
                        1.0e-5f,
                    });
                    dt = std::min(dt, cflTarget * cell.minLength() / speed);
                }
                (void)geom;
            }
        }
        return std::clamp(dt, 0.00005f, std::clamp(cfg.dt, 0.00005f, 0.03f));
    }
};

} // namespace harm
