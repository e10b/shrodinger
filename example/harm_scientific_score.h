#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>

#include "harm_amr_evolution.h"
#include "harm_config.h"
#include "harm_cpu_solver.h"
#include "harm_diagnostics.h"
#include "harm_grid.h"
#include "harm_initial_data.h"
#include "harm_state_norms.h"

namespace harm {

struct ScientificReplacementReport {
    int coarseGrid = 0;
    int mediumGrid = 0;
    int longFrames = 0;
    int frames = 0;
    float score = 0.0f;
    float coarseMassDrift = 0.0f;
    float coarseInternalEnergyDrift = 0.0f;
    float coarseDivBL1 = 0.0f;
    float coarseFailFrac = 0.0f;
    float coarseQProduct = 0.0f;
    float mediumMassDrift = 0.0f;
    float mediumInternalEnergyDrift = 0.0f;
    float mediumDivBL1 = 0.0f;
    float mediumFailFrac = 0.0f;
    float resolutionDriftSpread = 0.0f;
    float longCoarseMassDrift = 0.0f;
    float longCoarseInternalEnergyDrift = 0.0f;
    float longCoarseDivBL1 = 0.0f;
    float longCoarseFailFrac = 0.0f;
    bool coarseFineStable = false;
    bool resolutionTrendStable = false;
    bool longWindowStable = false;
    bool gpuParityAudit = false;
};

class ScientificReplacementScorer {
public:
    static ScientificReplacementReport analyze(const Config& cfg,
                                               const Diagnostics& initial,
                                               const Diagnostics& evolved,
                                               float massDrift,
                                               float internalEnergyDrift,
                                               const AdaptiveEvolutionReport& adaptive,
                                               int frames) {
        ScientificReplacementReport report{};
        report.frames = frames;
        report.coarseGrid = std::max(32, cfg.radialN / 2);
        report.mediumGrid = std::clamp((report.coarseGrid + cfg.radialN) / 2, report.coarseGrid, cfg.radialN);
        report.longFrames = std::max(frames * 2, 6);

        RunSummary coarseRun = runAtGrid(cfg, report.coarseGrid, frames);
        RunSummary mediumRun = runAtGrid(cfg, report.mediumGrid, frames);
        RunSummary longCoarseRun = runAtGrid(cfg, report.coarseGrid, report.longFrames);

        report.coarseMassDrift = coarseRun.massDrift;
        report.coarseInternalEnergyDrift = coarseRun.internalEnergyDrift;
        report.coarseDivBL1 = coarseRun.divBL1;
        report.coarseFailFrac = coarseRun.failFrac;
        report.coarseQProduct = coarseRun.qProduct;
        report.mediumMassDrift = mediumRun.massDrift;
        report.mediumInternalEnergyDrift = mediumRun.internalEnergyDrift;
        report.mediumDivBL1 = mediumRun.divBL1;
        report.mediumFailFrac = mediumRun.failFrac;
        report.longCoarseMassDrift = longCoarseRun.massDrift;
        report.longCoarseInternalEnergyDrift = longCoarseRun.internalEnergyDrift;
        report.longCoarseDivBL1 = longCoarseRun.divBL1;
        report.longCoarseFailFrac = longCoarseRun.failFrac;
        report.resolutionDriftSpread = std::max(
            std::abs(report.coarseMassDrift - massDrift),
            std::max(std::abs(report.mediumMassDrift - massDrift),
                     std::abs(report.mediumMassDrift - report.coarseMassDrift)));
        report.coarseFineStable = report.coarseMassDrift < 0.18f &&
            report.coarseInternalEnergyDrift < 0.18f &&
            report.coarseDivBL1 < 1.0e-4f &&
            report.coarseFailFrac < 1.0e-3f;
        report.resolutionTrendStable = report.resolutionDriftSpread < 0.08f &&
            report.mediumDivBL1 < 1.0e-4f &&
            report.mediumFailFrac < 1.0e-3f;
        report.longWindowStable = report.longCoarseMassDrift < 0.25f &&
            report.longCoarseInternalEnergyDrift < 0.25f &&
            report.longCoarseDivBL1 < 2.0e-4f &&
            report.longCoarseFailFrac < 1.0e-3f;
        // Source-token presence is not numerical parity.  GPU replacement
        // credit is withheld until the executable readback comparison passes.
        report.gpuParityAudit = false;

        float score = 0.0f;
        score += (cfg.highOrder ? 1.0f : 0.0f);
        score += (evolved.failFrac < 1.0e-3f && initial.failFrac < 1.0e-5f) ? 1.0f : 0.0f;
        score += (evolved.divBL1 < 5.0e-5f) ? 1.0f : 0.0f;
        score += (massDrift < 0.12f && internalEnergyDrift < 0.12f) ? 1.0f : 0.0f;
        score += (adaptive.evolvedBlocks > 0 && adaptive.parityVsUniform.rhoL1 < 0.05f) ? 1.0f : 0.0f;
        score += (evolved.qTheta > 3.0f && evolved.qPhi > 3.0f) ? 1.0f : 0.0f;
        score += report.coarseFineStable ? 1.0f : 0.0f;
        score += report.resolutionTrendStable ? 1.0f : 0.0f;
        score += report.longWindowStable ? 1.0f : 0.0f;
        score += report.gpuParityAudit ? 1.0f : 0.0f;
        report.score = std::min(score, 10.0f);
        return report;
    }

private:
    struct RunSummary {
        float massDrift = 0.0f;
        float internalEnergyDrift = 0.0f;
        float divBL1 = 0.0f;
        float failFrac = 0.0f;
        float qProduct = 0.0f;
    };

    static float relativeChange(float current, float reference) {
        return std::abs(current - reference) / std::max(std::abs(reference), 1.0e-20f);
    }

    static RunSummary runAtGrid(const Config& cfg, int n, int frames) {
        Config run = cfg;
        run.radialN = std::clamp(n, 32, cfg.maxGrid);
        run.thetaN = std::clamp(n / 2, 16, cfg.maxGrid);
        run.phiN = std::clamp(n, 32, cfg.maxGrid);
        run.maxGrid = std::max(run.maxGrid, run.radialN);
        run.highOrder = true;
        run.clamp();

        Grid grid{};
        Diagnostics initial = InitialDataBuilder::build(run, grid);
        Diagnostics evolved = initial;
        float time = 0.0f;
        for (int i = 0; i < frames; ++i) {
            CpuSolver::step(run, grid, evolved, time);
        }

        RunSummary out{};
        out.massDrift = relativeChange(evolved.mass, initial.mass);
        out.internalEnergyDrift = relativeChange(evolved.internalEnergy, initial.internalEnergy);
        out.divBL1 = evolved.divBL1;
        out.failFrac = evolved.failFrac;
        out.qProduct = evolved.qProduct;
        return out;
    }

    [[maybe_unused]] static bool gpuComputeParityAudit() {
        // Only the executable HarmGpuParity readback test may establish GPU
        // evolution parity. Source inspection is intentionally not evidence.
        return false;
    }
};

inline void writeScientificReplacementReport(const ScientificReplacementReport& report, std::ostream& os) {
    os << "\n## Scientific Replacement Score\n\n";
    os << "| Metric | Value |\n|---|---:|\n";
    os << "| score | " << report.score << " / 10 |\n";
    os << "| coarse grid | " << report.coarseGrid << " |\n";
    os << "| medium grid | " << report.mediumGrid << " |\n";
    os << "| frames | " << report.frames << " |\n";
    os << "| long-window frames | " << report.longFrames << " |\n";
    os << "| coarse mass drift | " << report.coarseMassDrift << " |\n";
    os << "| coarse internal energy drift | " << report.coarseInternalEnergyDrift << " |\n";
    os << "| coarse divB L1 | " << report.coarseDivBL1 << " |\n";
    os << "| coarse fail fraction | " << report.coarseFailFrac << " |\n";
    os << "| coarse Q product | " << report.coarseQProduct << " |\n";
    os << "| medium mass drift | " << report.mediumMassDrift << " |\n";
    os << "| medium internal energy drift | " << report.mediumInternalEnergyDrift << " |\n";
    os << "| medium divB L1 | " << report.mediumDivBL1 << " |\n";
    os << "| medium fail fraction | " << report.mediumFailFrac << " |\n";
    os << "| resolution drift spread | " << report.resolutionDriftSpread << " |\n";
    os << "| long coarse mass drift | " << report.longCoarseMassDrift << " |\n";
    os << "| long coarse internal energy drift | " << report.longCoarseInternalEnergyDrift << " |\n";
    os << "| long coarse divB L1 | " << report.longCoarseDivBL1 << " |\n";
    os << "| long coarse fail fraction | " << report.longCoarseFailFrac << " |\n";
    os << "| coarse/fine stability gate | " << (report.coarseFineStable ? "PASS" : "FAIL") << " |\n";
    os << "| resolution trend gate | " << (report.resolutionTrendStable ? "PASS" : "FAIL") << " |\n";
    os << "| long-window stability gate | " << (report.longWindowStable ? "PASS" : "FAIL") << " |\n";
    os << "| GPU executable evolution parity | " << (report.gpuParityAudit ? "PASS" : "NOT YET VALIDATED") << " |\n";
}

} // namespace harm
