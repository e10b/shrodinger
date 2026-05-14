#pragma once

#include <glm/glm.hpp>
#include <SDL3/SDL.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct CityMeshData {
    static constexpr int kFloatsPerVertex = 9; // position xyz, normal xyz, uv xy, texture layer

    std::vector<float> interleaved;
    std::vector<uint32_t> indices;
    glm::vec3 boundsMin{-1.0f};
    glm::vec3 boundsMax{1.0f};

    int texArrayW = 256;
    int texArrayH = 256;
    /** Layer 0 is solid white; additional layers are base-color textures (RGBA8, texArrayW x texArrayH). */
    std::vector<std::vector<uint8_t>> texLayers_;
    std::unordered_map<std::size_t, uint32_t> gltfImageToLayer_;
};

struct FlyCameraState {
    glm::vec3 position{0.0f, 2.0f, 8.0f};
    float yaw = 0.0f;
    float pitch = 0.0f;
    float moveSpeed = 28.0f;
    float mouseSensitivity = 0.0025f;
    bool lookSamplePrev = false;
    float lastLookMx = 0.0f;
    float lastLookMy = 0.0f;

    void resetToBounds(glm::vec3 bmin, glm::vec3 bmax);
};

bool loadCityGlbInto(const std::string& path, CityMeshData& out, std::string& errMsg);

void flyCameraUpdate(FlyCameraState& cam, float dt, SDL_Window* window, bool imguiWantsKeyboard, bool imguiWantsMouse);

glm::mat4 flyCameraView(const FlyCameraState& cam);
