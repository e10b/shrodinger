#pragma once

#include <algorithm>

#include "harm_config.h"
#include "harm_geometry.h"
#include "harm_state.h"

namespace harm {

class HarmBoundaries {
public:
    static void applyOutflow(const Config& cfg, int ir, int it, Primitive& p) {
        if (ir == 0) {
            // Excision boundary: prohibit information-carrying outflow from
            // the excised region, without deleting mass/energy by hand.
            p.v.x = std::min(p.v.x, 0.0f);
        }
        if (ir == cfg.radialN - 1) {
            p.v.x = std::max(p.v.x, 0.0f);
        }

        const int polarBand = std::min(it, cfg.thetaN - 1 - it);
        if (polarBand == 0) {
            p.v.y = 0.0f;
        }

        const float r = HarmGeometry::radiusAt(cfg, ir);
        p.rho = std::max(p.rho, cfg.rhoFloorAt(r));
        p.u = std::max(p.u, cfg.uFloorAt(r));
    }
};

} // namespace harm
