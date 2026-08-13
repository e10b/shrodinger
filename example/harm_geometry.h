#pragma once

#include <algorithm>
#include <cmath>

#include "harm_config.h"
#include "harm_kerr_schild.h"
#include "harm_types.h"

namespace harm {

struct CellGeometry {
    float r = 0.0f;
    float theta = 0.0f;
    float sinTheta = 0.0f;
    float dr = 0.0f;
    float dtheta = 0.0f;
    float dphi = 0.0f;
    float volume = 0.0f;
    Metric metric{};

    float radialLength() const {
        return dr;
    }

    float thetaLength() const {
        return std::max(r * dtheta, 1.0e-6f);
    }

    float phiLength() const {
        return std::max(r * sinTheta * dphi, 1.0e-6f);
    }

    float minLength() const {
        return std::min(radialLength(), std::min(thetaLength(), phiLength()));
    }
};

class HarmGeometry {
public:
    static float rin(const Config& cfg) {
        return std::max(cfg.rin, 1.0e-3f);
    }

    static float rout(const Config& cfg) {
        return std::max(cfg.rout, rin(cfg) + 4.0f);
    }

    static float logRange(const Config& cfg) {
        return std::max(std::log(rout(cfg)) - std::log(rin(cfg)), 1.0e-6f);
    }

    static float radiusAt(const Config& cfg, int ir) {
        const float x = (static_cast<float>(std::clamp(ir, 0, cfg.radialN - 1)) + 0.5f) /
            static_cast<float>(std::max(cfg.radialN, 1));
        return std::exp(std::log(rin(cfg)) + x * logRange(cfg));
    }

    static float thetaAt(const Config& cfg, int it) {
        const float y = (static_cast<float>(std::clamp(it, 0, cfg.thetaN - 1)) + 0.5f) /
            static_cast<float>(std::max(cfg.thetaN, 1));
        return y * thetaSpan();
    }

    static float dtheta(const Config& cfg) {
        return thetaSpan() / static_cast<float>(std::max(cfg.thetaN, 1));
    }

    static float dphi(const Config& cfg) {
        return 2.0f * kPi / static_cast<float>(std::max(cfg.phiN, 1));
    }

    static CellGeometry cell(const Config& cfg, int ir, int it) {
        CellGeometry g{};
        g.r = radiusAt(cfg, ir);
        g.theta = thetaAt(cfg, it);
        g.sinTheta = std::max(std::sin(g.theta), 1.0e-4f);
        g.dr = std::max(g.r * logRange(cfg) / static_cast<float>(std::max(cfg.radialN, 1)), 1.0e-4f);
        g.dtheta = dtheta(cfg);
        g.dphi = dphi(cfg);
        g.metric = KerrSchild::metric(g.r, g.theta, cfg.spin);
        g.volume = g.metric.sqrtMinusG * g.dr * g.dtheta * g.dphi;
        return g;
    }

private:
    static constexpr float thetaSpan() {
        return kPi;
    }
};

} // namespace harm
