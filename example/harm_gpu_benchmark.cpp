#define WGPU_IMPLEMENTATION

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "SDL3/SDL.h"
#include "wgfx.h"

#include "harm_config.h"
#include "harm_diagnostics.h"
#include "harm_gpu_compute.h"
#include "harm_grid.h"
#include "harm_initial_data.h"
#include "harm_kerr_schild.h"

namespace {

struct BenchCase {
    int grid = 96;
    int radialN = 96;
    int thetaN = 48;
    int phiN = 96;
    uint64_t cells = 0;
    int frames = 8;
    int substeps = 1;
    double seconds = 0.0;
    float simulatedTime = 0.0f;
    float finalDt = 0.0f;
    harm::Diagnostics initial{};
    harm::Diagnostics evolved{};
};

struct CheckpointHeader {
    char magic[8] = {'H', 'A', 'R', 'M', '3', '2', 'C', 'P'};
    uint32_t version = 1;
    uint32_t radialN = 0;
    uint32_t thetaN = 0;
    uint32_t phiN = 0;
    float simulatedTime = 0.0f;
    float actualDt = 0.0f;
    uint64_t completedFrames = 0;
    uint64_t packedFloats = 0;
};

bool saveCheckpoint(const std::string& path, const harm::Config& cfg, const harm::Grid& grid,
                    float simulatedTime, float actualDt, uint64_t completedFrames) {
    if (path.empty() || grid.readback.size() != harm::packedFloatCount(cfg.cellCount())) return false;
    CheckpointHeader header{};
    header.radialN = static_cast<uint32_t>(cfg.radialN);
    header.thetaN = static_cast<uint32_t>(cfg.thetaN);
    header.phiN = static_cast<uint32_t>(cfg.phiN);
    header.simulatedTime = simulatedTime;
    header.actualDt = actualDt;
    header.completedFrames = completedFrames;
    header.packedFloats = grid.readback.size();
    const std::string temporary = path + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(grid.readback.data()),
              static_cast<std::streamsize>(grid.readback.size() * sizeof(float)));
    out.close();
    if (!out) return false;
    std::remove(path.c_str());
    return std::rename(temporary.c_str(), path.c_str()) == 0;
}

bool loadCheckpoint(const std::string& path, const harm::Config& cfg, harm::Grid& grid,
                    float& simulatedTime, float& actualDt, uint64_t& completedFrames) {
    if (path.empty()) return false;
    std::ifstream in(path, std::ios::binary);
    CheckpointHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    const CheckpointHeader expected{};
    if (!in || std::memcmp(header.magic, expected.magic, sizeof(header.magic)) != 0 ||
        header.version != 1 || header.radialN != static_cast<uint32_t>(cfg.radialN) ||
        header.thetaN != static_cast<uint32_t>(cfg.thetaN) ||
        header.phiN != static_cast<uint32_t>(cfg.phiN) ||
        header.packedFloats != harm::packedFloatCount(cfg.cellCount())) {
        return false;
    }
    grid.packed.resize(static_cast<size_t>(header.packedFloats));
    in.read(reinterpret_cast<char*>(grid.packed.data()),
            static_cast<std::streamsize>(grid.packed.size() * sizeof(float)));
    if (!in) return false;
    simulatedTime = header.simulatedTime;
    actualDt = header.actualDt;
    completedFrames = header.completedFrames;
    return true;
}

std::vector<int> parseGridList(const std::string& value) {
    std::vector<int> grids;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            grids.push_back(std::stoi(item));
        }
    }
    return grids;
}

bool initHeadlessWebGpu() {
    wgfx::instance = wgpuCreateInstance(nullptr);
    std::cout << "Requesting headless adapter...\n";
    wgpu::RequestAdapterOptions adapterOpts = {};
    wgfx::adapter = wgfx::instance.requestAdapter(adapterOpts);
    if (!wgfx::adapter) {
        return false;
    }

    WGPUAdapterProperties adapterInfo{};
    wgpuAdapterGetProperties(static_cast<WGPUAdapter>(wgfx::adapter), &adapterInfo);
    std::cout << "WebGPU adapter: " << (adapterInfo.name ? adapterInfo.name : "unknown")
        << " | vendor=" << (adapterInfo.vendorName ? adapterInfo.vendorName : "unknown")
        << " | backend=" << static_cast<int>(adapterInfo.backendType)
        << " | type=" << static_cast<int>(adapterInfo.adapterType) << "\n";

    wgpu::SupportedLimits adapterLimits;
    wgfx::adapter.getLimits(&adapterLimits);
    wgpu::RequiredLimits requiredLimits = {};
    requiredLimits.limits = adapterLimits.limits;
    requiredLimits.limits.maxStorageBufferBindingSize = std::min<uint64_t>(
        adapterLimits.limits.maxStorageBufferBindingSize, 1073741824ull);
    requiredLimits.limits.maxBufferSize = std::min<uint64_t>(
        adapterLimits.limits.maxBufferSize, 1073741824ull);

    wgpu::DeviceDescriptor deviceDesc = {};
    deviceDesc.label = "Headless HARM GPU benchmark device";
    deviceDesc.requiredLimits = &requiredLimits;
    deviceDesc.defaultQueue.label = "Headless HARM GPU benchmark queue";
    wgfx::device = wgfx::adapter.requestDevice(deviceDesc);
    if (!wgfx::device) {
        return false;
    }

    wgpu::SupportedLimits deviceLimits;
    wgfx::device.getLimits(&deviceLimits);
    wgfx::deviceLimits = deviceLimits.limits;
    wgfx::queue = wgfx::device.getQueue();
    wgfx::surfaceFormat = wgpu::TextureFormat::BGRA8UnormSrgb;
    std::cout << "Device maxStorageBufferBindingSize: "
        << wgfx::deviceLimits.maxStorageBufferBindingSize << " bytes\n";
    return static_cast<bool>(wgfx::queue);
}

void waitForGpu(wgfx::Uniform* storage) {
    if (!storage) return;
    wgpu::BufferDescriptor fenceDesc = {};
    fenceDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    fenceDesc.size = 4;
    wgpu::Buffer fence = wgpu::Device(wgfx::device).createBuffer(fenceDesc);

    wgpu::CommandEncoder encoder = wgpu::Device(wgfx::device).createCommandEncoder();
    encoder.copyBufferToBuffer(storage->buffer, 0, fence, 0, 4);
    wgpu::CommandBuffer cmd = encoder.finish();
    wgpu::Queue(wgfx::queue).submit(1, &cmd);

    bool done = false;
    wgpuBufferMapAsync((WGPUBuffer)fence, WGPUMapMode_Read, 0, 4,
        [](WGPUBufferMapAsyncStatus, void* userdata) {
            *static_cast<bool*>(userdata) = true;
        },
        &done);
    while (!done) {
        wgpuDevicePoll((WGPUDevice)wgfx::device, true, nullptr);
    }
    fence.unmap();
}

harm::Config makeConfig(int n, int substeps, float dt, bool highOrder, bool cubic) {
    harm::Config cfg{};
    cfg.spin = 0.9375f;
    cfg.rin = 1.10f;
    cfg.rout = 50.0f;
    cfg.radialN = std::clamp(n, 32, harm::Config::kTiledMaxGrid);
    cfg.thetaN = std::clamp(cubic ? n : n / 2, 16, harm::Config::kTiledMaxGrid);
    cfg.phiN = std::clamp(n, 32, harm::Config::kTiledMaxGrid);
    cfg.maxGrid = harm::Config::kTiledMaxGrid;
    cfg.initialData = 2;
    cfg.useGpu = true;
    cfg.liveGpuDiagnostics = true;
    cfg.readbackInterval = 1;
    cfg.substeps = std::clamp(substeps, 1, harm::Config::kMaxSubstepsPerFrame);
    cfg.dt = dt;
    cfg.highOrder = highOrder;
    cfg.clamp();
    return cfg;
}

BenchCase runCase(int n, int frames, int substeps, float dt, bool highOrder, bool cubic, int resizeFrom,
                  float targetTime, int checkpointEvery, const std::string& checkpointPath,
                  const std::string& resumePath) {
    harm::Config cfg = makeConfig(n, substeps, dt, highOrder, cubic);
    harm::Grid grid{};
    const harm::Diagnostics initial = harm::InitialDataBuilder::build(cfg, grid);

    harm::GpuCompute gpu{};
    if (resizeFrom > 0) {
        harm::Config oldCfg = makeConfig(resizeFrom, substeps, dt, highOrder, true);
        harm::Grid oldGrid{};
        harm::InitialDataBuilder::build(oldCfg, oldGrid);
        gpu.init(oldCfg);
        gpu.uploadInitial(oldCfg, oldGrid);
        std::cout << "Reallocating GPU HARM storage from " << resizeFrom << "^3 to " << n << "^3...\n";
    }
    gpu.init(cfg);
    float resumedTime = 0.0f;
    float resumedDt = dt;
    uint64_t completedFrames = 0;
    if (!resumePath.empty()) {
        if (!loadCheckpoint(resumePath, cfg, grid, resumedTime, resumedDt, completedFrames)) {
            throw std::runtime_error("Could not load compatible checkpoint: " + resumePath);
        }
        std::cout << "Resuming checkpoint at t=" << resumedTime << " after "
                  << completedFrames << " frames\n";
        gpu.uploadState(cfg, grid, resumedTime, resumedDt);
    } else {
        gpu.uploadInitial(cfg, grid);
    }

    wgfx::ComputePass computePass;
    float time = resumedTime;
    float measuredTime = resumedTime;
    int executedFrames = 0;
    waitForGpu(gpu.stateB(0));
    const auto start = std::chrono::steady_clock::now();
    const int progressInterval = std::max(1, checkpointEvery);
    const bool runToTime = targetTime > resumedTime;
    while ((runToTime && measuredTime < targetTime) || (!runToTime && executedFrames < frames)) {
        const int batch = runToTime ? progressInterval : std::min(progressInterval, frames - executedFrames);
        for (int frame = 0; frame < batch; ++frame) {
        wgfx::start();
        computePass.prepare();
        gpu.dispatch(computePass, cfg, time);
        computePass.end();
        wgpu::CommandBuffer cmd = wgfx::encoder.finish();
        wgfx::queue.submit(1, &cmd);
        wgfx::encoder.release();
        wgfx::encoder = nullptr;
        }
        executedFrames += batch;
        completedFrames += static_cast<uint64_t>(batch);
        waitForGpu(gpu.stateB(0));
        harm::Diagnostics progress{};
        wgfx::start();
        gpu.queueReadback(cfg);
        wgpu::CommandBuffer progressCmd = wgfx::encoder.finish();
        wgfx::queue.submit(1, &progressCmd);
        wgfx::encoder.release();
        wgfx::encoder = nullptr;
        if (!gpu.consumeReadback(cfg, grid, progress)) {
            throw std::runtime_error("GPU benchmark progress readback failed");
        }
        measuredTime = gpu.simulatedTime();
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "PROGRESS t=" << std::fixed << std::setprecision(5) << measuredTime
                  << " dt=" << std::scientific << gpu.actualTimeStep()
                  << " frames=" << completedFrames << " wall_s=" << std::fixed << elapsed
                  << " fail=" << std::scientific << progress.failFrac
                  << " divB_L1=" << progress.divBL1 << std::endl;
        if (!checkpointPath.empty() &&
            !saveCheckpoint(checkpointPath, cfg, grid, measuredTime, gpu.actualTimeStep(), completedFrames)) {
            throw std::runtime_error("Could not write checkpoint: " + checkpointPath);
        }
    }
    waitForGpu(gpu.stateB(0));
    const auto stop = std::chrono::steady_clock::now();

    // Read back the final device state once, outside the timed region, so the
    // benchmark also catches non-finite or catastrophically drifting runs.
    harm::Diagnostics evolved{};
    wgfx::start();
    gpu.queueReadback(cfg);
    wgpu::CommandBuffer readbackCmd = wgfx::encoder.finish();
    wgfx::queue.submit(1, &readbackCmd);
    wgfx::encoder.release();
    wgfx::encoder = nullptr;
    if (!gpu.consumeReadback(cfg, grid, evolved)) {
        throw std::runtime_error("GPU benchmark final readback failed");
    }
    time = gpu.simulatedTime();

    BenchCase result{};
    result.grid = n;
    result.radialN = cfg.radialN;
    result.thetaN = cfg.thetaN;
    result.phiN = cfg.phiN;
    result.cells = cfg.cellCount();
    result.frames = executedFrames;
    result.substeps = cfg.substeps;
    result.seconds = std::chrono::duration<double>(stop - start).count();
    result.simulatedTime = time;
    result.finalDt = gpu.actualTimeStep();
    result.initial = initial;
    result.evolved = evolved;
    return result;
}

void writeMarkdown(const std::vector<BenchCase>& results, const std::string& path) {
    std::ofstream out(path);
    out << "# HARM GPU Benchmark\n\n";
    out << "| Grid | Cells | Frames | Wall seconds | GPU cell-updates/s | Simulated time | Final adaptive dt | Mass drift | U drift | B-energy drift | divB L1 | Recovery fail |\n";
    out << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const BenchCase& r : results) {
        const double updates = static_cast<double>(r.cells) * static_cast<double>(r.frames) * static_cast<double>(r.substeps);
        const double cellUpdatesPerSecond = r.seconds > 0.0 ? updates / r.seconds : 0.0;
        auto drift = [](float a, float b) { return (b - a) / std::max(std::abs(a), 1.0e-20f); };
        out << "| " << r.radialN << "x" << r.thetaN << "x" << r.phiN << " | " << r.cells << " | " << r.frames
            << " | " << std::fixed << std::setprecision(3) << r.seconds
            << " | " << std::scientific << std::setprecision(3) << cellUpdatesPerSecond
            << " | " << std::fixed << std::setprecision(5) << r.simulatedTime
            << " | " << std::scientific << r.finalDt
            << " | " << std::scientific << drift(r.initial.mass, r.evolved.mass)
            << " | " << drift(r.initial.internalEnergy, r.evolved.internalEnergy)
            << " | " << drift(r.initial.magneticEnergy, r.evolved.magneticEnergy)
            << " | " << r.evolved.divBL1 << " | " << r.evolved.failFrac << " |\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> grids = {96, 128, 192, 512};
    int frames = 8;
    int substeps = 1;
    float dt = 0.0005f;
    bool highOrder = false;
    bool cubic = true;
    int resizeFrom = 0;
    float targetTime = -1.0f;
    int checkpointEvery = 100;
    std::string checkpointPath;
    std::string resumePath;
    std::string outPath = "harm_gpu_benchmark.md";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--grid" || arg == "--grids") && i + 1 < argc) {
            grids = parseGridList(argv[++i]);
        } else if (arg == "--frames" && i + 1 < argc) {
            frames = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--substeps" && i + 1 < argc) {
            substeps = std::clamp(std::stoi(argv[++i]), 1, harm::Config::kMaxSubstepsPerFrame);
        } else if (arg == "--dt" && i + 1 < argc) {
            dt = std::stof(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            outPath = argv[++i];
        } else if (arg == "--high-order") {
            highOrder = true;
        } else if (arg == "--cubic") {
            cubic = true;
        } else if (arg == "--resize-from" && i + 1 < argc) {
            resizeFrom = std::stoi(argv[++i]);
        } else if (arg == "--target-time" && i + 1 < argc) {
            targetTime = std::stof(argv[++i]);
        } else if (arg == "--checkpoint-every" && i + 1 < argc) {
            checkpointEvery = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            checkpointPath = argv[++i];
        } else if (arg == "--resume" && i + 1 < argc) {
            resumePath = argv[++i];
        }
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL init warning; continuing to headless WebGPU adapter request\n";
    }
    SDL_Window* window = SDL_CreateWindow("HARM GPU benchmark", 64, 64, SDL_WINDOW_HIDDEN);
    if (window) {
        wgfx::init(wgfx::getSurface(window));
    } else if (!initHeadlessWebGpu()) {
        std::cerr << "Failed to initialize hidden-window or headless WebGPU.\n";
        SDL_Quit();
        return 2;
    }

    std::vector<BenchCase> results;
    for (int grid : grids) {
        std::cout << "Running GPU HARM benchmark at " << grid << (cubic ? "^3" : " x N/2 x N") << "...\n";
        results.push_back(runCase(grid, frames, substeps, dt, highOrder, cubic, resizeFrom,
                                  targetTime, checkpointEvery, checkpointPath, resumePath));
    }
    writeMarkdown(results, outPath);
    std::cout << "Wrote " << outPath << "\n";
    if (window) {
        SDL_DestroyWindow(window);
    }
    SDL_Quit();
    return 0;
}
