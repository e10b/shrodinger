#define WGPU_IMPLEMENTATION

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "SDL3/SDL.h"
#include "wgfx.h"

#include "harm_config.h"
#include "harm_cpu_solver.h"
#include "harm_diagnostics.h"
#include "harm_gpu_compute.h"
#include "harm_grid.h"
#include "harm_initial_data.h"
#include "harm_kerr_schild.h"
#include "harm_state_norms.h"

namespace {

void writeDiagnostics(const char* label, const harm::Diagnostics& d, std::ostream& os) {
    os << "\n### " << label << "\n\n";
    os << "| Metric | Value |\n|---|---:|\n";
    os << "| mass | " << d.mass << " |\n";
    os << "| internal energy | " << d.internalEnergy << " |\n";
    os << "| magnetic energy | " << d.magneticEnergy << " |\n";
    os << "| mdot | " << d.mdot << " |\n";
    os << "| beta min | " << d.betaMin << " |\n";
    os << "| beta mean | " << d.betaMean << " |\n";
    os << "| divB L1 | " << d.divBL1 << " |\n";
    os << "| divB max | " << d.divBMax << " |\n";
    os << "| fail fraction | " << d.failFrac << " |\n";
    os << "| floor mass fraction | " << d.floorMassFrac << " |\n";
    os << "| Qtheta | " << d.qTheta << " |\n";
    os << "| Qphi | " << d.qPhi << " |\n";
}

void writeNorms(const harm::StateNorms& n, std::ostream& os) {
    os << "\n## CPU/GPU State Norms\n\n";
    os << "| Norm | Value |\n|---|---:|\n";
    os << "| rho L1 relative | " << n.rhoL1 << " |\n";
    os << "| rho L2 relative | " << n.rhoL2 << " |\n";
    os << "| rho Linf relative | " << n.rhoLinf << " |\n";
    os << "| internal energy L1 relative | " << n.uL1 << " |\n";
    os << "| velocity L1 absolute | " << n.velocityL1 << " |\n";
    os << "| magnetic magnitude L1 relative | " << n.magneticL1 << " |\n";
}

float relDelta(float a, float b, float floor = 1.0e-12f) {
    return std::abs(b - a) / std::max(std::abs(a), floor);
}

void writeDiagnosticDeltas(const harm::Diagnostics& cpu, const harm::Diagnostics& gpu, std::ostream& os) {
    os << "\n## CPU/GPU Diagnostic Deltas\n\n";
    os << "| Metric | Relative/absolute delta |\n|---|---:|\n";
    os << "| mass relative | " << relDelta(cpu.mass, gpu.mass) << " |\n";
    os << "| internal energy relative | " << relDelta(cpu.internalEnergy, gpu.internalEnergy) << " |\n";
    os << "| magnetic energy relative | " << relDelta(cpu.magneticEnergy, gpu.magneticEnergy) << " |\n";
    os << "| mdot relative | " << relDelta(cpu.mdot, gpu.mdot, 1.0e-20f) << " |\n";
    os << "| divB L1 relative | " << relDelta(cpu.divBL1, gpu.divBL1, 1.0e-20f) << " |\n";
    os << "| fail fraction absolute | " << std::abs(gpu.failFrac - cpu.failFrac) << " |\n";
    os << "| floor mass fraction absolute | " << std::abs(gpu.floorMassFrac - cpu.floorMassFrac) << " |\n";
    os << "| Qtheta relative | " << relDelta(cpu.qTheta, gpu.qTheta) << " |\n";
    os << "| Qphi relative | " << relDelta(cpu.qPhi, gpu.qPhi) << " |\n";
}

} // namespace

int main(int argc, char** argv) {
    harm::Config cfg{};
    cfg.spin = 0.9375f;
    cfg.rin = 1.10f;
    cfg.rout = 50.0f;
    cfg.radialN = 32;
    cfg.thetaN = 16;
    cfg.phiN = 32;
    cfg.maxGrid = 64;
    cfg.initialData = 2;
    cfg.useGpu = true;
    cfg.liveGpuDiagnostics = true;
    cfg.readbackInterval = 1;
    cfg.substeps = 1;
    cfg.dt = 0.0005f;

    int frames = 1;
    std::string outPath = "gpu_cpu_parity.md";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc) {
            frames = std::stoi(argv[++i]);
        } else if (arg == "--grid" && i + 1 < argc) {
            const int n = std::stoi(argv[++i]);
            cfg.radialN = std::clamp(n, 32, 64);
            cfg.thetaN = std::clamp(n, 16, 64);
            cfg.phiN = std::clamp(n, 32, 64);
        } else if (arg == "--out" && i + 1 < argc) {
            outPath = argv[++i];
        } else if (arg == "--dt" && i + 1 < argc) {
            cfg.dt = std::stof(argv[++i]);
        } else if (arg == "--high-order") {
            cfg.highOrder = true;
        }
    }
    cfg.clamp();

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL init warning; continuing to WebGPU adapter request\n";
    }
    SDL_Window* window = SDL_CreateWindow("HARM GPU parity", 64, 64, SDL_WINDOW_HIDDEN);
    if (!window) {
        std::cerr << "SDL window creation failed\n";
        return 2;
    }
    wgfx::init(wgfx::getSurface(window));

    harm::Grid initialGrid{};
    const harm::Diagnostics initial = harm::InitialDataBuilder::build(cfg, initialGrid);

    harm::Grid cpuGrid = initialGrid;
    harm::Diagnostics cpuDiagnostics = initial;
    float cpuTime = 0.0f;
    for (int i = 0; i < frames; ++i) {
        harm::CpuSolver::stepWithIntegrator(cfg, cpuGrid, cpuDiagnostics, cpuTime, harm::CpuSolver::Integrator::Midpoint);
    }

    harm::GpuCompute gpu{};
    gpu.init(cfg);
    if (!gpu.valid()) {
        std::ofstream out(outPath);
        out << "# GPU/CPU HARM Parity Report\n\n";
        out << "GPU shader or compute pipeline validation failed.\n\n```text\n"
            << gpu.validationError() << "\n```\n\nOverall: **FAIL**\n";
        std::cerr << "GPU shader/pipeline validation failed: " << gpu.validationError() << "\n";
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 2;
    }
    gpu.uploadInitial(cfg, initialGrid);
    harm::Grid gpuGrid = initialGrid;
    harm::Diagnostics gpuDiagnostics = initial;
    float gpuTime = 0.0f;
    wgfx::ComputePass computePass;
    for (int i = 0; i < frames; ++i) {
        wgfx::start();
        computePass.prepare();
        gpu.dispatch(computePass, cfg, gpuTime);
        computePass.end();
        gpu.queueReadback(cfg);
        wgpu::CommandBuffer cmd = wgfx::encoder.finish();
        wgfx::queue.submit(1, &cmd);
        wgfx::encoder.release();
        wgfx::encoder = nullptr;
        gpu.consumeReadback(cfg, gpuGrid, gpuDiagnostics);
    }
    if (frames == 0) {
        // Exercise the actual upload/copy/map path even when no evolution is
        // requested. Falling back to initialGrid here would make this gate a
        // CPU self-comparison and could hide GPU storage-layout defects.
        wgfx::start();
        gpu.queueReadback(cfg);
        wgpu::CommandBuffer cmd = wgfx::encoder.finish();
        wgfx::queue.submit(1, &cmd);
        wgfx::encoder.release();
        wgfx::encoder = nullptr;
        if (!gpu.consumeReadback(cfg, gpuGrid, gpuDiagnostics)) {
            std::cerr << "GPU upload readback did not complete\n";
            SDL_DestroyWindow(window);
            SDL_Quit();
            return 2;
        }
    }

    if (gpuGrid.readback.empty()) {
        std::cerr << "GPU parity has no device readback to compare\n";
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 2;
    }
    const std::vector<float>& gpuState = gpuGrid.readback;
    const harm::StateNorms norms = harm::StateNormSampler::compare(cfg, cpuGrid.packed, gpuState);
    const bool uploadOnly = frames == 0;
    const bool pass = uploadOnly ?
        (norms.rhoLinf < 1.0e-7f && norms.uL1 < 1.0e-7f && norms.velocityL1 < 1.0e-7f && norms.magneticL1 < 1.0e-7f) :
        (norms.rhoL1 < 0.02f && norms.uL1 < 0.05f && norms.velocityL1 < 0.02f &&
         norms.magneticL1 < 0.02f && relDelta(cpuDiagnostics.mass, gpuDiagnostics.mass) < 0.01f &&
         relDelta(cpuDiagnostics.internalEnergy, gpuDiagnostics.internalEnergy) < 0.02f &&
         relDelta(cpuDiagnostics.magneticEnergy, gpuDiagnostics.magneticEnergy) < 0.02f &&
         std::abs(gpuDiagnostics.failFrac - cpuDiagnostics.failFrac) < 0.004f &&
         std::abs(gpuDiagnostics.floorMassFrac - cpuDiagnostics.floorMassFrac) < 0.002f);

    std::ofstream out(outPath);
    out << "# GPU/CPU HARM Parity Report\n\n";
    out << "| Setting | Value |\n|---|---:|\n";
    out << "| grid | " << cfg.radialN << " x " << cfg.thetaN << " x " << cfg.phiN << " |\n";
    out << "| frames | " << frames << " |\n";
    out << "| high-order | " << (cfg.highOrder ? "enabled" : "disabled") << " |\n";
    out << "| dt ceiling | " << cfg.dt << " |\n";
    writeNorms(norms, out);
    writeDiagnosticDeltas(cpuDiagnostics, gpuDiagnostics, out);
    writeDiagnostics("initial", initial, out);
    writeDiagnostics("CPU", cpuDiagnostics, out);
    writeDiagnostics("GPU", gpuDiagnostics, out);
    out << "\nOverall: **" << (pass ? "PASS" : "FAIL") << "**\n";

    std::cout << "Wrote " << outPath << "\n";
    std::cout << "GPU parity " << (pass ? "PASS" : "FAIL") << "\n";
    SDL_DestroyWindow(window);
    SDL_Quit();
    return pass ? 0 : 2;
}
