#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "context.h"
#include "imgui.h"
#include "wgfx.h"

class Quad {
public:
    static Quad& Instance() {
        static Quad instance;
        return instance;
    }

    bool isCityWalkMode() const {
        return false;
    }

    bool shouldDrawMainSceneGeometry() const {
        return true;
    }

    wgfx::Pipeline* pipeline = nullptr;

    void dispatchCompute3d() {}

    void drawImGuiPanel() {
        ImGui::Begin("Basics: Circle");
        ImGui::Text("Back to basics: single circle renderer");
        ImGui::SliderFloat2("center", glm::value_ptr(circleCenter_), -1.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("radius", &circleRadius_, 0.02f, 0.95f, "%.3f");
        ImGui::SliderFloat("edge softness", &circleSoftness_, 0.0005f, 0.25f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::ColorEdit3("color", glm::value_ptr(circleColor_));
        ImGui::SliderFloat("glow", &circleGlow_, 0.0f, 2.0f, "%.3f");
        ImGui::SliderFloat("background", &circleBackground_, 0.0f, 0.2f, "%.3f");
        ImGui::End();
    }

    void render(float dt) {
        time_ += std::max(dt, 0.0f);

        int width = 1280;
        int height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);

        gpu2dState_.orbital = glm::vec4(circleColor_, std::max(circleBackground_, 0.0f));
        gpu2dState_.tuning = glm::vec4(
            std::clamp(circleRadius_, 0.01f, 0.99f),
            std::max(circleSoftness_, 0.0001f),
            std::max(circleGlow_, 0.0f),
            0.0f);
        gpu2dState_.render = glm::vec4(time_, aspect, 0.0f, 0.0f);
        gpu2dState_.pan = glm::vec4(circleCenter_.x, circleCenter_.y, 0.0f, 0.0f);
        gpu2dState_.tdse = glm::vec4(0.0f);

        writeRenderUniform(pipeline2d_, reinterpret_cast<const float*>(&gpu2dState_));
        pipeline2d_->setVertexBuffer(vbo2d_.get());
        pipeline2d_->setIndexBuffer(ibo2d_.get());
        pipeline = pipeline2d_;
    }

private:
    struct alignas(16) Gpu2dState {
        glm::vec4 orbital;
        glm::vec4 tuning;
        glm::vec4 render;
        glm::vec4 pan;
        glm::vec4 tdse;
    };

    std::unique_ptr<wgfx::VertexBuffer> vbo2d_;
    std::unique_ptr<wgfx::IndexBuffer> ibo2d_;
    wgfx::Pipeline* pipeline2d_ = nullptr;
    wgfx::Uniform* stateUniform2d_ = nullptr;

    Gpu2dState gpu2dState_{};
    float time_ = 0.0f;

    glm::vec2 circleCenter_ = glm::vec2(0.0f);
    float circleRadius_ = 0.35f;
    float circleSoftness_ = 0.008f;
    glm::vec3 circleColor_ = glm::vec3(1.0f);
    float circleGlow_ = 0.25f;
    float circleBackground_ = 0.0f;

    static void writeRenderUniform(wgfx::Pipeline* activePipeline, const float* data) {
        if (!activePipeline || activePipeline->uniforms.uniforms.empty()) return;
        wgfx::Uniform* uniform = activePipeline->uniforms.uniforms.at(0);
        wgfx::queue.writeBuffer(uniform->buffer, 0, data, uniform->minBindingSize);
        if (activePipeline->uniforms.dynamicOffsets.empty()) {
            activePipeline->uniforms.dynamicOffsets.resize(1, 0);
        }
        activePipeline->uniforms.dynamicOffsets[0] = 0;
    }

    void init2dBuffers() {
        const std::vector<float> vertices = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f
        };
        const std::vector<uint16_t> indices = { 0, 1, 2, 0, 2, 3 };

        vbo2d_.reset(wgfx::createVertexBuffer(vertices));
        vbo2d_->setTopology(PrimitiveTopology::TriangleList);
        vbo2d_->setAttribute(0, wgfx::vec3f, 0);

        ibo2d_.reset(wgfx::createIndexBuffer(indices));
        pipeline2d_->setVertexBuffer(vbo2d_.get());
        pipeline2d_->setIndexBuffer(ibo2d_.get());
    }

    Quad() {
        pipeline2d_ = wgfx::loadPipeline(
            wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "circle_2d.wgsl").c_str()));
        stateUniform2d_ = wgfx::createUniform(0, sizeof(Gpu2dState), reinterpret_cast<const float*>(&gpu2dState_));
        pipeline2d_->uniforms.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        pipeline2d_->uniforms.setUniform(stateUniform2d_);
        pipeline2d_->targets = 1;
        pipeline2d_->useDepth = false;

        init2dBuffers();
        pipeline2d_->init(vbo2d_.get());

        pipeline = pipeline2d_;
        pipeline->setVertexBuffer(vbo2d_.get());
        pipeline->setIndexBuffer(ibo2d_.get());
    }

    Quad(const Quad&) = delete;
    void operator=(const Quad&) = delete;
};
