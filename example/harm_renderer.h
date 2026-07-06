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
        gpu_ = &gpu;
        pipeline = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "harm_grmhd.wgsl").c_str()));
        uniform_ = wgfx::createUniform(0, sizeof(RenderUniform), reinterpret_cast<const float*>(&state_));
        pipeline->uniforms.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        pipeline->uniforms.setUniform(uniform_);
        for (int slab = 0; slab < Config::kMaxPhiSlabs; ++slab) {
            wgfx::Uniform* source = gpu.stateB(slab);
            storageViews_[slab] = new wgfx::Uniform();
            storageViews_[slab]->isReadOnly = true;
            storageViews_[slab]->binding = 1 + slab;
            storageViews_[slab]->minBindingSize = source->minBindingSize;
            storageViews_[slab]->buffer = source->buffer;
            storageViews_[slab]->entry.binding = 1 + slab;
            storageViews_[slab]->entry.buffer = source->buffer;
            storageViews_[slab]->entry.offset = 0;
            storageViews_[slab]->entry.size = static_cast<uint64_t>(source->minBindingSize);
            pipeline->uniforms.setStorage(storageViews_[slab]);
        }
        pipeline->targets = 1;
        pipeline->useDepth = false;
        pipeline->init(quad.vertexBuffer());
        quad.bind(pipeline);
    }

    void update(const Config& cfg, const CameraController& camera, float time, float aspect) {
        state_.mode = glm::vec4(
            static_cast<float>(cfg.lensingMode),
            static_cast<float>(gpu_ ? gpu_->activeSlabPhi(cfg) : cfg.phiN),
            static_cast<float>(gpu_ ? gpu_->activePhiSlabs(cfg) : 1),
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
    GpuCompute* gpu_ = nullptr;
    wgfx::Uniform* uniform_ = nullptr;
    wgfx::Uniform* storageViews_[Config::kMaxPhiSlabs] = {};
};

} // namespace harm
