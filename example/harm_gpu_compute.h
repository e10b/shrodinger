#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

#include "webgpu/webgpu.hpp"
#include "wgfx.h"
#include "harm_config.h"
#include "harm_diagnostics.h"
#include "harm_grid.h"
#include "harm_gpu_storage_plan.h"
#include "harm_types.h"

namespace harm {

class GpuCompute {
public:
    wgfx::Uniform* gpuA = nullptr;
    wgfx::Uniform* gpuB = nullptr;

    void init(const Config& cfg) {
        storagePlan_ = makeGpuStoragePlan(cfg, wgfx::deviceLimits);
        std::cout << "HARM GPU storage: state buffer "
            << (storagePlan_.bytesPerState / (1024.0 * 1024.0)) << " MiB, ping-pong "
            << (storagePlan_.pingPongBytes() / (1024.0 * 1024.0)) << " MiB, binding cap "
            << (storagePlan_.maxBindingBytes / (1024.0 * 1024.0)) << " MiB";
        if (storagePlan_.requiresTiling) {
            std::cout << ", needs " << storagePlan_.phiSlabs << " phi slabs";
        }
        std::cout << "\n";
        if (storagePlan_.requiresTiling) {
            throw std::runtime_error(
                "Requested HARM grid exceeds the current single-storage-buffer WebGPU path. "
                "The storage planner has selected phi-slab tiling, but tiled compute/render shaders are not wired yet.");
        }

        const size_t bytes = storagePlan_.bytesPerState;
        gpuA = wgfx::createStorage(1, bytes, nullptr, false);
        gpuB = wgfx::createStorage(2, bytes, nullptr, false);

        wgpu::BufferDescriptor desc = {};
        desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        desc.size = bytes;
        desc.mappedAtCreation = false;
        readbackBuffer_ = wgpu::Device(wgfx::device).createBuffer(desc);

        const std::string src = wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "harm_grmhd_compute.wgsl").c_str());
        paramsUniform_ = wgfx::createUniform(0, sizeof(ComputeParams), reinterpret_cast<const float*>(&params_));
        makeCompute(step_, src, "harm_step");
        makeCompute(copy_, src, "copy_b_to_a");
    }

    void uploadInitial(const Grid& grid) {
        if (!gpuA || !gpuB || grid.packed.empty()) return;
        const size_t bytes = grid.packed.size() * sizeof(float);
        wgfx::queue.writeBuffer(gpuA->buffer, 0, grid.packed.data(), bytes);
        wgfx::queue.writeBuffer(gpuB->buffer, 0, grid.packed.data(), bytes);
    }

    void dispatch(wgfx::ComputePass& pass, const Config& cfg, float& time) {
        if (!step_ || !copy_ || cfg.paused) return;
        params_.gridN = static_cast<uint32_t>(cfg.radialN);
        params_.thetaN = static_cast<uint32_t>(cfg.thetaN);
        params_.phiN = static_cast<uint32_t>(cfg.phiN);
        params_.substeps = static_cast<uint32_t>(std::clamp(cfg.substeps, 1, 12));
        params_.dt = std::clamp(cfg.dt, 0.00005f, 0.03f);
        params_.rin = std::max(cfg.rin, 1.05f);
        params_.rout = std::max(cfg.rout, params_.rin + 4.0f);
        params_.spin = std::clamp(cfg.spin, -0.98f, 0.98f);
        params_.rhoFloor = std::max(cfg.rhoFloor, 1e-8f);
        params_.uFloor = std::max(cfg.uFloor, 1e-9f);
        params_.magneticLoop = cfg.magneticLoop;
        params_.time = time;
        params_.problem = static_cast<uint32_t>(std::clamp(cfg.initialData, 0, 1));
        params_.highOrder = cfg.highOrder ? 1.0f : 0.0f;

        wgfx::queue.writeBuffer(paramsUniform_->buffer, 0, &params_, sizeof(ComputeParams));
        pinUniformOffset(step_);
        pinUniformOffset(copy_);

        const uint32_t wg1 = workgroups(cfg.radialN);
        const uint32_t wg2 = workgroups(cfg.thetaN);
        const uint32_t wg3 = (static_cast<uint32_t>(cfg.phiN) + 3u) / 4u;
        for (int i = 0; i < std::clamp(cfg.substeps, 1, 12); ++i) {
            pass.drawXYZ(step_, wg1, wg2, wg3);
            pass.drawXYZ(copy_, wg1, wg2, wg3);
            time += params_.dt;
        }
    }

    void queueReadback(const Config& cfg) {
        if (!cfg.liveGpuDiagnostics || !readbackBuffer_ || !gpuB || readbackPending_ || cfg.paused) return;
        readbackFrame_ = (readbackFrame_ + 1) % std::clamp(cfg.readbackInterval, 1, 120);
        if (readbackFrame_ != 0) return;
        wgfx::encoder.copyBufferToBuffer(gpuB->buffer, 0, readbackBuffer_, 0, packedByteCount(cfg.cellCount()));
        readbackPending_ = true;
    }

    bool consumeReadback(const Config& cfg, Grid& grid, Diagnostics& diagnostics) {
        if (!readbackPending_ || !readbackBuffer_) return false;
        const size_t bytes = packedByteCount(cfg.cellCount());
        bool done = false;
        wgpuBufferMapAsync((WGPUBuffer)readbackBuffer_, WGPUMapMode_Read, 0, bytes,
            [](WGPUBufferMapAsyncStatus, void* userdata) {
                *static_cast<bool*>(userdata) = true;
            },
            &done);
        while (!done) {
            wgpuDevicePoll((WGPUDevice)wgfx::device, true, nullptr);
        }
        const void* data = readbackBuffer_.getConstMappedRange(0, bytes);
        if (data != nullptr) {
            const float* floats = static_cast<const float*>(data);
            grid.readback.assign(floats, floats + bytes / sizeof(float));
            diagnostics = DiagnosticsSampler::compute(cfg, grid.readback, true);
        }
        readbackBuffer_.unmap();
        readbackPending_ = false;
        return true;
    }

private:
    ComputeParams params_{};
    wgfx::Uniform* paramsUniform_ = nullptr;
    wgfx::Compute* step_ = nullptr;
    wgfx::Compute* copy_ = nullptr;
    wgpu::Buffer readbackBuffer_ = nullptr;
    GpuStoragePlan storagePlan_{};
    bool readbackPending_ = false;
    int readbackFrame_ = 0;

    void makeCompute(wgfx::Compute*& out, const std::string& source, const std::string& entry) {
        out = wgfx::loadCompute(source);
        out->entryPoint = entry;
        out->uniforms.visibility = wgpu::ShaderStage::Compute;
        out->uniforms.setUniform(paramsUniform_);
        out->uniforms.setStorage(gpuA);
        out->uniforms.setStorage(gpuB);
        out->init();
    }

    static uint32_t workgroups(int n) {
        return (static_cast<uint32_t>(n) + 7u) / 8u;
    }

    static void pinUniformOffset(wgfx::Compute* c) {
        if (!c) return;
        c->uniforms.dynamicOffsets.resize(1, 0);
        c->uniforms.dynamicOffsets[0] = 0;
        if (!c->uniforms.uniforms.empty()) {
            c->uniforms.uniforms[0]->quantity = 0;
        }
    }
};

} // namespace harm
