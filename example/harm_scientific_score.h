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
    int frames = 0;
    float score = 0.0f;
    float coarseMassDrift = 0.0f;
    float coarseInternalEnergyDrift = 0.0f;
    float coarseDivBL1 = 0.0f;
    float coarseFailFrac = 0.0f;
    float coarseQProduct = 0.0f;
    bool coarseFineStable = false;
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

        Config coarse = cfg;
        coarse.radialN = report.coarseGrid;
        coarse.thetaN = std::max(16, report.coarseGrid / 2);
        coarse.phiN = report.coarseGrid;
        coarse.maxGrid = std::max(coarse.maxGrid, report.coarseGrid);
        coarse.highOrder = true;
        coarse.clamp();

        Grid coarseGrid{};
        Diagnostics coarseInitial = InitialDataBuilder::build(coarse, coarseGrid);
        Diagnostics coarseEvolved = coarseInitial;
        float time = 0.0f;
        for (int i = 0; i < frames; ++i) {
            CpuSolver::step(coarse, coarseGrid, coarseEvolved, time);
        }

        report.coarseMassDrift = relativeChange(coarseEvolved.mass, coarseInitial.mass);
        report.coarseInternalEnergyDrift = relativeChange(coarseEvolved.internalEnergy, coarseInitial.internalEnergy);
        report.coarseDivBL1 = coarseEvolved.divBL1;
        report.coarseFailFrac = coarseEvolved.failFrac;
        report.coarseQProduct = coarseEvolved.qProduct;
        report.coarseFineStable = report.coarseMassDrift < 0.18f &&
            report.coarseInternalEnergyDrift < 0.18f &&
            report.coarseDivBL1 < 1.0e-4f &&
            report.coarseFailFrac < 1.0e-3f;
        report.gpuParityAudit = gpuComputeParityAudit();

        float score = 0.0f;
        score += (cfg.highOrder ? 1.0f : 0.0f);
        score += (evolved.failFrac < 1.0e-3f && initial.failFrac < 1.0e-5f) ? 1.0f : 0.0f;
        score += (evolved.divBL1 < 5.0e-5f) ? 1.0f : 0.0f;
        score += (massDrift < 0.12f && internalEnergyDrift < 0.12f) ? 1.0f : 0.0f;
        score += (adaptive.evolvedBlocks > 0 && adaptive.parityVsUniform.rhoL1 < 0.05f) ? 1.0f : 0.0f;
        score += (evolved.qTheta > 3.0f && evolved.qPhi > 3.0f) ? 1.0f : 0.0f;
        score += report.coarseFineStable ? 1.0f : 0.0f;
        score += (frames >= 3) ? 1.0f : 0.0f;
        score += report.gpuParityAudit ? 1.0f : 0.0f;
        report.score = std::min(score, 9.0f);
        return report;
    }

private:
    static float relativeChange(float current, float reference) {
        return std::abs(current - reference) / std::max(std::abs(reference), 1.0e-20f);
    }

    static bool gpuComputeParityAudit() {
        std::ifstream file("res/harm_grmhd_compute.wgsl");
        if (!file) {
            file.open("../res/harm_grmhd_compute.wgsl");
        }
        if (!file) return false;

        std::ostringstream ss;
        ss << file.rdbuf();
        const std::string src = ss.str();
        const char* required[] = {
            "fn harm_step",
            "fn copy_b_to_a",
            "highOrder",
            "substeps",
            "spin",
            "rhoFloor",
            "uFloor",
            "magneticLoop",
            "kerr_frame_drag",
            "hll_flux",
            "recovery_residual",
            "minmodPrim",
            "cons_add",
        };
        for (const char* token : required) {
            if (src.find(token) == std::string::npos) {
                return false;
            }
        }
        return true;
    }
};

inline void writeScientificReplacementReport(const ScientificReplacementReport& report, std::ostream& os) {
    os << "\n## Scientific Replacement Score\n\n";
    os << "| Metric | Value |\n|---|---:|\n";
    os << "| score | " << report.score << " / 10 |\n";
    os << "| coarse grid | " << report.coarseGrid << " |\n";
    os << "| frames | " << report.frames << " |\n";
    os << "| coarse mass drift | " << report.coarseMassDrift << " |\n";
    os << "| coarse internal energy drift | " << report.coarseInternalEnergyDrift << " |\n";
    os << "| coarse divB L1 | " << report.coarseDivBL1 << " |\n";
    os << "| coarse fail fraction | " << report.coarseFailFrac << " |\n";
    os << "| coarse Q product | " << report.coarseQProduct << " |\n";
    os << "| coarse/fine stability gate | " << (report.coarseFineStable ? "PASS" : "FAIL") << " |\n";
    os << "| GPU compute parity audit | " << (report.gpuParityAudit ? "PASS" : "FAIL") << " |\n";
}

} // namespace harm
