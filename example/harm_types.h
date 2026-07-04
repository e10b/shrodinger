#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace harm {

constexpr float kPi = 3.14159265358979323846f;

struct alignas(16) RenderUniform {
    glm::vec4 mode;    // x:lensing mode, w:view mode
    glm::vec4 tuning;  // x:color scale, y:r_in, z:zoom, w:spin
    glm::vec4 render;  // x:time, y:aspect, z:inclination, w:yaw
    glm::vec4 pan;     // xy:pan, z:r_in, w:gravity enabled
    glm::vec4 grid;    // x:n1, y:r_out, z:n2, w:n3
};

struct alignas(16) ComputeParams {
    uint32_t gridN = 0;
    uint32_t thetaN = 0;
    uint32_t phiN = 0;
    uint32_t substeps = 0;
    float dt = 0.0f;
    float rin = 0.0f;
    float rout = 0.0f;
    float spin = 0.0f;
    float rhoFloor = 0.0f;
    float uFloor = 0.0f;
    float magneticLoop = 0.0f;
    float time = 0.0f;
    uint32_t problem = 0;
    float highOrder = 0.0f;
    float pad2 = 0.0f;
    float pad3 = 0.0f;
};

struct PackedPrim {
    float rho = 0.0f;
    float u = 0.0f;
    float ur = 0.0f;
    float utheta = 0.0f;
    float uphi = 0.0f;
    float br = 0.0f;
    float btheta = 0.0f;
    float bphi = 0.0f;
    float fail = 0.0f;
    float pad0 = 0.0f;
    float pad1 = 0.0f;
    float pad2 = 0.0f;
};

struct Diagnostics {
    float betaMin = 0.0f;
    float betaMean = 0.0f;
    float phiBH = 0.0f;
    float sigmaMax = 0.0f;
    float mass = 0.0f;
    float internalEnergy = 0.0f;
    float magneticEnergy = 0.0f;
    float mdot = 0.0f;
    float angularMomentum = 0.0f;
    float divBL1 = 0.0f;
    float divBMax = 0.0f;
    float floorMassFrac = 0.0f;
    float failFrac = 0.0f;
    float maxLorentz = 1.0f;
    float cfl = 0.0f;
    bool gpuLive = false;
};

struct CameraKeyframe {
    float time = 0.0f;
    float yaw = 0.0f;
    float inclination = 1.18f;
    float zoom = 0.070f;
};

inline size_t packedFloatCount(size_t cells) {
    return cells * 12;
}

inline size_t packedByteCount(size_t cells) {
    return packedFloatCount(cells) * sizeof(float);
}

} // namespace harm
