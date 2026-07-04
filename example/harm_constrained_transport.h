#pragma once

#include <algorithm>

#include <glm/glm.hpp>

#include "harm_state.h"

namespace harm {

class ConstrainedTransport {
public:
    static glm::vec3 idealElectricField(const Primitive& p) {
        return -glm::cross(p.v, p.B);
    }

    static glm::vec3 curlElectric(
        const glm::vec3& eRMinus,
        const glm::vec3& eRPlus,
        const glm::vec3& eThetaMinus,
        const glm::vec3& eThetaPlus,
        const glm::vec3& ePhiMinus,
        const glm::vec3& ePhiPlus,
        float r,
        float theta,
        float dr,
        float dtheta,
        float dphi) {
        const float sinTh = std::max(std::sin(theta), 0.08f);
        const float rMinus = std::max(r - dr, 1.0e-4f);
        const float rPlus = r + dr;
        const float thMinus = std::max(theta - dtheta, 1.0e-4f);
        const float thPlus = std::min(theta + dtheta, kPi - 1.0e-4f);
        glm::vec3 curl{};
        curl.x = ((std::sin(thPlus) * eThetaPlus.z - std::sin(thMinus) * eThetaMinus.z) / std::max(2.0f * dtheta, 1.0e-8f) -
                  (ePhiPlus.y - ePhiMinus.y) / std::max(2.0f * dphi, 1.0e-8f)) / std::max(r * sinTh, 1.0e-8f);
        curl.y = ((ePhiPlus.x - ePhiMinus.x) / std::max(2.0f * dphi, 1.0e-8f) / sinTh -
                  (rPlus * eRPlus.z - rMinus * eRMinus.z) / std::max(2.0f * dr, 1.0e-8f)) / std::max(r, 1.0e-8f);
        curl.z = ((rPlus * eRPlus.y - rMinus * eRMinus.y) / std::max(2.0f * dr, 1.0e-8f) -
                  (eThetaPlus.x - eThetaMinus.x) / std::max(2.0f * dtheta, 1.0e-8f)) / std::max(r, 1.0e-8f);
        return curl;
    }

    static float sphericalDivB(const Primitive& rm, const Primitive& rp,
                               const Primitive& tm, const Primitive& tp,
                               const Primitive& pm, const Primitive& pp,
                               float r, float theta, float dr, float dtheta, float dphi) {
        const float sinTh = std::max(std::sin(theta), 0.08f);
        const float rMinus = std::max(r - dr, 1.0e-4f);
        const float rPlus = r + dr;
        const float thMinus = std::max(theta - dtheta, 1.0e-4f);
        const float thPlus = std::min(theta + dtheta, kPi - 1.0e-4f);
        const float radial = (rPlus * rPlus * rp.B.x - rMinus * rMinus * rm.B.x) / std::max(2.0f * dr, 1.0e-8f);
        const float polar = (std::sin(thPlus) * tp.B.y - std::sin(thMinus) * tm.B.y) / std::max(2.0f * dtheta, 1.0e-8f);
        const float azimuth = (pp.B.z - pm.B.z) / std::max(2.0f * dphi, 1.0e-8f);
        return radial / std::max(r * r, 1.0e-8f) + polar / std::max(r * sinTh, 1.0e-8f) + azimuth / std::max(r * sinTh, 1.0e-8f);
    }
};

} // namespace harm
