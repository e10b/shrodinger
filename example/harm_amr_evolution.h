#pragma once

#include <algorithm>

#include "harm_amr.h"
#include "harm_config.h"
#include "harm_cpu_solver.h"
#include "harm_diagnostics.h"
#include "harm_grid.h"
#include "harm_state_norms.h"
#include "harm_timestep.h"

namespace harm {

struct AdaptiveEvolutionReport {
    int cycles = 0;
    int evolvedBlocks = 0;
    int evolvedCells = 0;
    float coveredFraction = 0.0f;
    float localDt = 0.0f;
    Diagnostics diagnostics{};
    StateNorms parityVsUniform{};
};

class AdaptiveBlockStepper {
public:
    static AdaptiveEvolutionReport evolve(const Config& cfg, const Grid& initial, const Grid& uniformReference, int frames, Grid& adaptiveGrid) {
        AdaptiveEvolutionReport report{};
        adaptiveGrid = initial;
        if (adaptiveGrid.packed.size() < packedFloatCount(cfg.cellCount()) || frames <= 0) {
            report.diagnostics = DiagnosticsSampler::compute(cfg, adaptiveGrid.packed, false);
            report.parityVsUniform = StateNormSampler::compare(cfg, uniformReference.packed, adaptiveGrid.packed);
            return report;
        }

        Config localCfg = cfg;
        localCfg.highOrder = true;
        localCfg.substeps = 1;
        for (int frame = 0; frame < frames; ++frame) {
            const RefinementSummary refinement = AmrRefinementCriterion::analyze(localCfg, adaptiveGrid.packed);
            const float coarseDt = HarmTimeStepper::stableDt(localCfg, adaptiveGrid.packed);
            report.localDt = coarseDt * 0.5f;
            for (const RefinementBlock& block : refinement.topBlocks) {
                SolverRegion region{};
                region.ir0 = block.ir0;
                region.it0 = block.it0;
                region.ip0 = block.ip0;
                region.nr = block.nr;
                region.nt = block.nt;
                region.np = block.np;
                CpuSolver::stepRegion(localCfg, adaptiveGrid, region, report.localDt);
                CpuSolver::stepRegion(localCfg, adaptiveGrid, region, report.localDt);
                ++report.evolvedBlocks;
                report.evolvedCells += block.nr * block.nt * block.np;
            }
            ++report.cycles;
        }

        report.coveredFraction = static_cast<float>(report.evolvedCells) /
            std::max(static_cast<float>(cfg.cellCount() * std::max(frames, 1)), 1.0f);
        report.diagnostics = DiagnosticsSampler::compute(cfg, adaptiveGrid.packed, false);
        report.parityVsUniform = StateNormSampler::compare(cfg, uniformReference.packed, adaptiveGrid.packed);
        return report;
    }
};

} // namespace harm
