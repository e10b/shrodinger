#pragma once

#include <algorithm>
#include <string>

#include <glm/glm.hpp>

#include "wgfx.h"
#include "harm_camera.h"
#include "harm_config.h"
#include "harm_fullscreen_quad.h"
#include "harm_gpu_compute.h"
#include "harm_types.h"

namespace harm {

class Renderer {
public:
    wgfx::Pipeline* pipeline = nullptr;

    void init(FullscreenQuad& quad, GpuCompute& gpu) {
        pipeline = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "harm_grmhd.wgsl").c_str()));
        uniform_ = wgfx::createUniform(0, sizeof(RenderUniform), reinterpret_cast<const float*>(&state_));
        pipeline->uniforms.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        pipeline->uniforms.setUniform(uniform_);
        storageView_ = new wgfx::Uniform();
        storageView_->isReadOnly = true;
        storageView_->binding = 1;
        storageView_->minBindingSize = gpu.gpuB->minBindingSize;
        storageView_->buffer = gpu.gpuB->buffer;
        storageView_->entry.binding = 1;
        storageView_->entry.buffer = gpu.gpuB->buffer;
        storageView_->entry.offset = 0;
        storageView_->entry.size = static_cast<uint64_t>(gpu.gpuB->minBindingSize);
        pipeline->uniforms.setStorage(storageView_);
        storageView1_ = new wgfx::Uniform();
        storageView1_->isReadOnly = true;
        storageView1_->binding = 2;
        storageView1_->minBindingSize = gpu.gpuB1->minBindingSize;
        storageView1_->buffer = gpu.gpuB1->buffer;
        storageView1_->entry.binding = 2;
        storageView1_->entry.buffer = gpu.gpuB1->buffer;
        storageView1_->entry.offset = 0;
        storageView1_->entry.size = static_cast<uint64_t>(gpu.gpuB1->minBindingSize);
        pipeline->uniforms.setStorage(storageView1_);
        pipeline->targets = 1;
        pipeline->useDepth = false;
        pipeline->init(quad.vertexBuffer());
        quad.bind(pipeline);
    }

    void update(const Config& cfg, const CameraController& camera, float time, float aspect) {
        state_.mode = glm::vec4(
            static_cast<float>(cfg.lensingMode),
            static_cast<float>((cfg.maxGrid + 1) / 2),
            1.0f,
            static_cast<float>(cfg.viewMode));
        state_.tuning = glm::vec4(std::max(cfg.colorScale, 0.001f), cfg.rin, std::max(camera.zoom, 1e-6f), cfg.spin);
        state_.render = glm::vec4(time, aspect, camera.inclination, camera.yaw);
        state_.pan = glm::vec4(camera.pan.x, camera.pan.y, cfg.rin, cfg.enableGravity ? 1.0f : 0.0f);
        state_.grid = glm::vec4(
            static_cast<float>(cfg.radialN),
            std::max(cfg.rout, 1.0f),
            static_cast<float>(cfg.thetaN),
            static_cast<float>(cfg.phiN));
        wgfx::queue.writeBuffer(uniform_->buffer, 0, &state_, sizeof(RenderUniform));
        if (pipeline->uniforms.dynamicOffsets.empty()) {
            pipeline->uniforms.dynamicOffsets.resize(1, 0);
        }
        pipeline->uniforms.dynamicOffsets[0] = 0;
    }

private:
    RenderUniform state_{};
    wgfx::Uniform* uniform_ = nullptr;
    wgfx::Uniform* storageView_ = nullptr;
    wgfx::Uniform* storageView1_ = nullptr;
};

} // namespace harm
