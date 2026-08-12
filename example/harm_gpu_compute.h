#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
    wgfx::Uniform* gpuA1 = nullptr;
    wgfx::Uniform* gpuB1 = nullptr;

    wgfx::Uniform* stateB(int slab) const {
        return (slab >= 0 && slab < static_cast<int>(gpuBSlabs_.size())) ? gpuBSlabs_[slab] : nullptr;
    }

    int activeSlabPhi(const Config& cfg) const {
        return slabPhi(cfg);
    }

    int activePhiSlabs(const Config& cfg) const {
        return phiSlabs(cfg);
    }

    void init(const Config& cfg) {
        storagePlan_ = makeGpuStoragePlan(cfg, wgfx::deviceLimits);
        if (storagePlan_.phiSlabs > Config::kMaxPhiSlabs) {
            throw std::runtime_error(
                "Requested HARM grid needs more phi slabs than the WebGPU HARM storage path supports.");
        }

        std::cout << "HARM GPU storage: full state "
            << (storagePlan_.bytesPerState / (1024.0 * 1024.0)) << " MiB, slab state "
            << (storagePlan_.bytesPerSlabState / (1024.0 * 1024.0)) << " MiB, ping-pong "
            << (storagePlan_.pingPongBytes() / (1024.0 * 1024.0)) << " MiB, binding cap "
            << (storagePlan_.maxBindingBytes / (1024.0 * 1024.0)) << " MiB, phi slabs "
            << storagePlan_.phiSlabs << ", slab phi " << storagePlan_.slabPhi << "\n";

        const size_t bytes = storagePlan_.bytesPerSlabState;
        gpuASlabs_.clear();
        gpuBSlabs_.clear();
        for (int slab = 0; slab < Config::kMaxPhiSlabs; ++slab) {
            const size_t slabBytes = slab < storagePlan_.phiSlabs ? bytes : gpuPackedByteCount(1);
            gpuASlabs_.push_back(wgfx::createStorage(1 + slab, slabBytes, nullptr, false));
        }
        for (int slab = 0; slab < Config::kMaxPhiSlabs; ++slab) {
            const size_t slabBytes = slab < storagePlan_.phiSlabs ? bytes : gpuPackedByteCount(1);
            gpuBSlabs_.push_back(wgfx::createStorage(1 + Config::kMaxPhiSlabs + slab, slabBytes, nullptr, false));
        }
        gpuA = gpuASlabs_[0];
        gpuA1 = gpuASlabs_[1];
        gpuB = gpuBSlabs_[0];
        gpuB1 = gpuBSlabs_[1];

        if (cfg.liveGpuDiagnostics) {
            wgpu::BufferDescriptor desc{};
            desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
            desc.size = packedByteCount(cfg.cellCount());
            desc.mappedAtCreation = false;
            readbackBuffer_ = wgpu::Device(wgfx::device).createBuffer(desc);
        } else {
            std::cout << "HARM GPU live readback disabled for compact GPU storage.\n";
        }

        const std::string src = wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "harm_grmhd_compute.wgsl").c_str());
        paramsUniform_ = wgfx::createUniform(0, sizeof(ComputeParams), reinterpret_cast<const float*>(&params_));
        makeCompute(step_, src, "harm_step");
        makeCompute(copy_, src, "copy_b_to_a");
    }

    void uploadInitial(const Config& cfg, const Grid& grid) {
        if (gpuASlabs_.empty() || gpuBSlabs_.empty() || grid.packed.empty()) return;
        for (int slab = 0; slab < phiSlabs(cfg); ++slab) {
            const SlabCopy copy = slabCopy(cfg, slab);
            writeSlab(gpuASlabs_[slab], grid.packed, copy);
            writeSlab(gpuBSlabs_[slab], grid.packed, copy);
        }
    }

    void dispatch(wgfx::ComputePass& pass, const Config& cfg, float& time) {
        if (!step_ || !copy_ || cfg.paused) return;
        params_.gridN = static_cast<uint32_t>(cfg.radialN);
        params_.thetaN = static_cast<uint32_t>(cfg.thetaN);
        params_.phiN = static_cast<uint32_t>(cfg.phiN);
        params_.substeps = static_cast<uint32_t>(std::clamp(cfg.substeps, 1, Config::kMaxSubstepsPerFrame));
        params_.dt = effectiveMovieTimeStep(cfg);
        params_.rin = std::max(cfg.rin, 1.05f);
        params_.rout = std::max(cfg.rout, params_.rin + 4.0f);
        params_.spin = std::clamp(cfg.spin, -0.98f, 0.98f);
        params_.rhoFloor = std::max(cfg.rhoFloor, 1e-8f);
        params_.uFloor = std::max(cfg.uFloor, 1e-9f);
        params_.magneticLoop = cfg.magneticLoop;
        params_.time = time;
        params_.problem = static_cast<uint32_t>(std::clamp(cfg.initialData, 0, 1));
        params_.highOrder = cfg.highOrder ? 1.0f : 0.0f;
        params_.phiSplit = static_cast<uint32_t>(slabPhi(cfg));
        params_.tiledStorage = static_cast<uint32_t>(phiSlabs(cfg));

        wgfx::queue.writeBuffer(paramsUniform_->buffer, 0, &params_, sizeof(ComputeParams));
        pinUniformOffset(step_);
        pinUniformOffset(copy_);

        const uint32_t wg1 = workgroups(cfg.radialN);
        const uint32_t wg2 = workgroups(cfg.thetaN);
        const uint32_t wg3 = (static_cast<uint32_t>(cfg.phiN) + 3u) / 4u;
        for (int i = 0; i < std::clamp(cfg.substeps, 1, Config::kMaxSubstepsPerFrame); ++i) {
            pass.drawXYZ(step_, wg1, wg2, wg3);
            pass.drawXYZ(copy_, wg1, wg2, wg3);
            time += params_.dt;
        }
    }

    void queueReadback(const Config& cfg) {
        if (!cfg.liveGpuDiagnostics || !readbackBuffer_ || !gpuB || readbackPending_ || cfg.paused) return;
        readbackFrame_ = (readbackFrame_ + 1) % std::clamp(cfg.readbackInterval, 1, 120);
        if (readbackFrame_ != 0) return;
        for (int slab = 0; slab < phiSlabs(cfg); ++slab) {
            const SlabCopy copy = slabCopy(cfg, slab);
            wgfx::encoder.copyBufferToBuffer(gpuBSlabs_[slab]->buffer, 0, readbackBuffer_, copy.fullOffsetBytes, copy.bytes);
        }
        readbackPending_ = true;
    }

    bool consumeReadback(const Config& cfg, Grid& grid, Diagnostics& diagnostics) {
        if (!readbackPending_ || !readbackBuffer_) return false;
        const size_t bytes = gpuPackedByteCount(cfg.cellCount());
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
            const size_t cells = cfg.cellCount();
            grid.readback.assign(packedFloatCount(cells), 0.0f);
            for (size_t i = 0; i < cells; ++i) {
                for (int j = 0; j < 8; ++j) {
                    grid.readback[i * 12 + j] = floats[i * 8 + j];
                }
            }
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
    std::vector<wgfx::Uniform*> gpuASlabs_;
    std::vector<wgfx::Uniform*> gpuBSlabs_;
    bool readbackPending_ = false;
    int readbackFrame_ = 0;

    void makeCompute(wgfx::Compute*& out, const std::string& source, const std::string& entry) {
        out = wgfx::loadCompute(source);
        out->entryPoint = entry;
        out->uniforms.visibility = wgpu::ShaderStage::Compute;
        out->uniforms.setUniform(paramsUniform_);
        for (wgfx::Uniform* slab : gpuASlabs_) {
            out->uniforms.setStorage(slab);
        }
        for (wgfx::Uniform* slab : gpuBSlabs_) {
            out->uniforms.setStorage(slab);
        }
        out->init();
    }

    static uint32_t workgroups(int n) {
        return (static_cast<uint32_t>(n) + 7u) / 8u;
    }

    static float effectiveMovieTimeStep(const Config& cfg) {
        const float requested = std::clamp(cfg.dt, 0.00005f, 0.03f);
        const float rin = std::max(cfg.rin, 1.05f);
        const float rout = std::max(cfg.rout, rin + 4.0f);
        constexpr float safety = 0.35f;

        const float n1 = static_cast<float>(std::max(cfg.radialN, 1));
        const float n2 = static_cast<float>(std::max(cfg.thetaN, 1));
        const float n3 = static_cast<float>(std::max(cfg.phiN, 1));
        const float dr = rin * std::log(rout / rin) / n1;
        const float dtheta = rin * (0.84f * 3.14159265358979323846f) / n2;
        const float dphi = rin * 0.08f * (2.0f * 3.14159265358979323846f) / n3;
        const float stable = safety * std::min({dr, dtheta, dphi});
        return std::clamp(std::min(requested, stable), 0.00005f, requested);
    }

    static void pinUniformOffset(wgfx::Compute* c) {
        if (!c) return;
        c->uniforms.dynamicOffsets.resize(1, 0);
        c->uniforms.dynamicOffsets[0] = 0;
        if (!c->uniforms.uniforms.empty()) {
            c->uniforms.uniforms[0]->quantity = 0;
        }
    }

    struct SlabCopy {
        size_t fullOffsetBytes = 0;
        size_t bytes = 0;
        size_t firstCell = 0;
        size_t cells = 0;
    };

    int phiSlabs(const Config& cfg) const {
        return std::clamp((cfg.phiN + slabPhi(cfg) - 1) / slabPhi(cfg), 1, storagePlan_.phiSlabs);
    }

    int slabPhi(const Config& cfg) const {
        return std::max(1, std::min(storagePlan_.slabPhi, cfg.phiN));
    }

    SlabCopy slabCopy(const Config& cfg, int slab) const {
        const size_t planeCells = static_cast<size_t>(cfg.radialN) * static_cast<size_t>(cfg.thetaN);
        const size_t startPhi = static_cast<size_t>(std::max(slab, 0)) * static_cast<size_t>(slabPhi(cfg));
        const size_t countPhi = startPhi < static_cast<size_t>(cfg.phiN)
            ? std::min(static_cast<size_t>(slabPhi(cfg)), static_cast<size_t>(cfg.phiN) - startPhi)
            : 0u;
        const size_t firstCell = startPhi * planeCells;
        const size_t gpuBytes = gpuPackedByteCount(countPhi * planeCells);
        return {firstCell * 12u * sizeof(float), gpuBytes, firstCell, countPhi * planeCells};
    }

    static void writeSlab(wgfx::Uniform* target, const std::vector<float>& packed, const SlabCopy& copy) {
        if (!target || copy.bytes == 0) return;
        constexpr size_t kUploadChunkBytes = 64ull * 1024ull * 1024ull;
        const size_t maxChunkCells = std::max<size_t>(1, kUploadChunkBytes / (kGpuPackedFloatCount * sizeof(float)));
        std::vector<float> chunk;
        size_t cellOffset = 0;
        while (cellOffset < copy.cells) {
            const size_t chunkCells = std::min(maxChunkCells, copy.cells - cellOffset);
            chunk.resize(chunkCells * kGpuPackedFloatCount);
            for (size_t i = 0; i < chunkCells; ++i) {
                const size_t src = (copy.firstCell + cellOffset + i) * 12u;
                const size_t dst = i * kGpuPackedFloatCount;
                for (size_t k = 0; k < kGpuPackedFloatCount; ++k) {
                    chunk[dst + k] = packed[src + k];
                }
            }
            const size_t dstByteOffset = cellOffset * kGpuPackedFloatCount * sizeof(float);
            wgfx::queue.writeBuffer(target->buffer, dstByteOffset, chunk.data(), chunkCells * kGpuPackedFloatCount * sizeof(float));
            cellOffset += chunkCells;
        }
    }
};

} // namespace harm
