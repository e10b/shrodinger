#pragma once

#include <algorithm>

#include "harm_config.h"
#include "harm_state.h"

namespace harm {

class HarmBoundaries {
public:
    static void applyOutflow(const Config& cfg, int ir, int it, Primitive& p) {
        if (ir < 3) {
            p.v.x = std::min(p.v.x, -0.05f);
            p.rho = std::max(0.82f * p.rho, cfg.rhoFloor);
            p.u = std::max(0.82f * p.u, cfg.uFloor);
        }
        if (ir > cfg.radialN - 4) {
            p.v.x = std::min(p.v.x, 0.25f);
            p.rho = std::max(0.92f * p.rho, cfg.rhoFloor);
            p.u = std::max(0.92f * p.u, cfg.uFloor);
        }

        const int polarBand = std::min(it, cfg.thetaN - 1 - it);
        if (polarBand < 2) {
            p.rho = std::max(0.80f * p.rho, cfg.rhoFloor);
            p.u = std::max(0.80f * p.u, cfg.uFloor);
            p.v.y *= 0.5f;
            p.B.y *= 0.5f;
        }

        p.rho = std::max(p.rho, cfg.rhoFloor);
        p.u = std::max(p.u, cfg.uFloor);
    }
};

} // namespace harm
