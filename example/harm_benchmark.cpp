#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "harm_amr_evolution.h"
#include "harm_config.h"
#include "harm_cpu_solver.h"
#include "harm_grid.h"
#include "harm_initial_data.h"
#include "harm_kerr_schild.h"
#include "harm_validation.h"

int main(int argc, char** argv) {
    harm::Config cfg{};
    cfg.spin = 0.9375f;
    cfg.rin = harm::KerrSchild::horizonRadius(cfg.spin) * 1.001f;
    cfg.rout = 50.0f;
    cfg.radialN = 64;
    cfg.thetaN = 32;
    cfg.phiN = 64;
    cfg.maxGrid = harm::Config::kSingleBufferMaxGrid;
    cfg.initialData = 2;
    cfg.useGpu = false;
    cfg.liveGpuDiagnostics = false;
    cfg.substeps = 1;
    cfg.dt = 0.0005f;

    int frames = 0;
    std::string outPath = "fishbone_benchmark.md";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc) {
            frames = std::stoi(argv[++i]);
        } else if (arg == "--grid" && i + 1 < argc) {
            const int n = std::stoi(argv[++i]);
            cfg.radialN = std::clamp(n, 32, harm::Config::kSingleBufferMaxGrid);
            cfg.thetaN = std::clamp(n / 2, 16, harm::Config::kSingleBufferMaxGrid);
            cfg.phiN = std::clamp(n, 32, harm::Config::kSingleBufferMaxGrid);
        } else if (arg == "--out" && i + 1 < argc) {
            outPath = argv[++i];
        } else if (arg == "--dt" && i + 1 < argc) {
            cfg.dt = std::stof(argv[++i]);
        } else if (arg == "--substeps" && i + 1 < argc) {
            cfg.substeps = std::stoi(argv[++i]);
        } else if (arg == "--high-order") {
            cfg.highOrder = true;
        }
    }
    cfg.clamp();

    harm::Grid grid{};
    harm::Diagnostics initial = harm::InitialDataBuilder::build(cfg, grid);
    harm::Grid initialGrid = grid;
    harm::Diagnostics evolved = initial;
    std::vector<harm::Diagnostics> samples;
    float time = 0.0f;
    for (int i = 0; i < frames; ++i) {
        harm::CpuSolver::step(cfg, grid, evolved, time);
        samples.push_back(evolved);
    }

    harm::Grid adaptiveGrid{};
    harm::AdaptiveEvolutionReport adaptive = harm::AdaptiveBlockStepper::evolve(cfg, initialGrid, grid, frames, adaptiveGrid);

    harm::FishboneReport report = harm::FishboneValidator::analyze(cfg, initialGrid, grid, initial, evolved, frames, adaptive);
    harm::FishboneValidator::attachSamples(report, std::move(samples));
    std::ofstream out(outPath);
    harm::FishboneValidator::writeMarkdown(report, out);
    std::cout << "Wrote " << outPath << "\n";
    std::cout << "Fishbone benchmark " << (report.passed() ? "PASS" : "FAIL") << "\n";
    return report.passed() ? 0 : 2;
}
