#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>

#include "SDL3/SDL.h"
#include "context.h"
#include "imgui.h"
#include "harm_types.h"

namespace harm {

class CameraController {
public:
    float yaw = 0.0f;
    float inclination = 1.18f;
    float zoom = 0.070f;
    glm::vec2 pan = glm::vec2(0.0f);
    bool playing = false;
    std::vector<CameraKeyframe> keyframes;

    void process(int width, int height, float aspect) {
        ImGuiIO& io = ImGui::GetIO();
        const bool allowMouse = !io.WantCaptureMouse;
        const float wheel = Context::Instance().consumeWheelDelta();
        if (wheel != 0.0f) {
            zoom *= std::exp(wheel * 0.08f);
            zoom = std::clamp(zoom, 0.003f, 8.0f);
        }

        float mx = 0.0f;
        float my = 0.0f;
        const Uint32 mask = allowMouse ? SDL_GetMouseState(&mx, &my) : 0;
        const bool leftDown = (mask & SDL_BUTTON_LMASK) != 0;
        const bool panDown = (mask & SDL_BUTTON_MMASK) != 0 || (mask & SDL_BUTTON_RMASK) != 0;
        const bool draggingNow = leftDown || panDown;
        if (!draggingNow) {
            dragging_ = false;
            return;
        }
        if (!dragging_) {
            dragging_ = true;
            lastMouse_ = glm::vec2(mx, my);
            return;
        }

        const glm::vec2 current(mx, my);
        const glm::vec2 delta = current - lastMouse_;
        lastMouse_ = current;
        const float safeWidth = std::max(static_cast<float>(width), 1.0f);
        const float safeHeight = std::max(static_cast<float>(height), 1.0f);
        if (leftDown) {
            yaw += delta.x * (2.4f / safeWidth);
            inclination += delta.y * (1.8f / safeHeight);
            if (yaw > kPi) yaw -= 2.0f * kPi;
            if (yaw < -kPi) yaw += 2.0f * kPi;
            inclination = std::clamp(inclination, 0.05f, 1.45f);
        } else if (panDown) {
            pan.x -= delta.x * (2.0f * aspect / safeWidth) / std::max(zoom, 1e-6f);
            pan.y += delta.y * (2.0f / safeHeight) / std::max(zoom, 1e-6f);
        }
    }

    void applyAnimation(float time) {
        if (!playing || keyframes.empty()) return;
        if (keyframes.size() == 1) {
            yaw = keyframes[0].yaw;
            inclination = keyframes[0].inclination;
            zoom = keyframes[0].zoom;
            return;
        }
        if (time <= keyframes.front().time) {
            set(keyframes.front());
            return;
        }
        if (time >= keyframes.back().time) {
            set(keyframes.back());
            return;
        }
        for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
            const auto& a = keyframes[i];
            const auto& b = keyframes[i + 1];
            if (time >= a.time && time < b.time) {
                const float t = (time - a.time) / std::max(b.time - a.time, 1e-6f);
                yaw = a.yaw + t * (b.yaw - a.yaw);
                inclination = a.inclination + t * (b.inclination - a.inclination);
                zoom = std::exp(std::log(std::max(a.zoom, 1e-6f)) + t * (std::log(std::max(b.zoom, 1e-6f)) - std::log(std::max(a.zoom, 1e-6f))));
                return;
            }
        }
    }

private:
    void set(const CameraKeyframe& k) {
        yaw = k.yaw;
        inclination = k.inclination;
        zoom = k.zoom;
    }

    bool dragging_ = false;
    glm::vec2 lastMouse_ = glm::vec2(0.0f);
};

} // namespace harm
