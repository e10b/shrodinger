#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "harm_config.h"
#include "harm_diagnostics.h"
#include "harm_grid.h"
#include "harm_initial_data.h"

namespace harm {

struct FishboneCheck {
    std::string name;
    float value = 0.0f;
    float target = 0.0f;
    float tolerance = 0.0f;
    bool pass = false;
};

struct FishboneReport {
    Config cfg{};
    Diagnostics initial{};
    Diagnostics evolved{};
    float peakRadius = 0.0f;
    float innerEdgeRadius = 0.0f;
    float betaMin = 0.0f;
    float pressureMax = 0.0f;
    float b2Max = 0.0f;
    float massDrift = 0.0f;
    float internalEnergyDrift = 0.0f;
    float magneticEnergyDrift = 0.0f;
    int frames = 0;
    std::vector<FishboneCheck> checks;
    std::vector<Diagnostics> samples;

    bool passed() const {
        return std::all_of(checks.begin(), checks.end(), [](const FishboneCheck& c) { return c.pass; });
    }
};

class FishboneValidator {
public:
    static FishboneReport analyze(const Config& cfg, const Grid& grid, const Diagnostics& initial, const Diagnostics& evolved, int frames) {
        FishboneReport report{};
        report.cfg = cfg;
        report.initial = initial;
        report.evolved = evolved;
        report.frames = frames;
        if (grid.packed.size() < packedFloatCount(cfg.cellCount())) {
            return report;
        }

        const float logRange = std::max(std::log(cfg.rout) - std::log(cfg.rin), 1.0e-6f);
        float rhoMax = 0.0f;
        float firstTorusRadius = cfg.rout;
        for (int ip = 0; ip < cfg.phiN; ++ip) {
            for (int it = 0; it < cfg.thetaN; ++it) {
                for (int ir = 0; ir < cfg.radialN; ++ir) {
                    const size_t base = ((static_cast<size_t>(ip) * static_cast<size_t>(cfg.thetaN) + static_cast<size_t>(it)) *
                        static_cast<size_t>(cfg.radialN) + static_cast<size_t>(ir)) * 12;
                    const float rho = grid.packed[base + 0];
                    const float uu = grid.packed[base + 1];
                    const float b2 = grid.packed[base + 5] * grid.packed[base + 5] +
                        grid.packed[base + 6] * grid.packed[base + 6] +
                        grid.packed[base + 7] * grid.packed[base + 7];
                    const float r = radiusAt(cfg, ir, logRange);
                    if (rho > rhoMax) {
                        rhoMax = rho;
                        report.peakRadius = r;
                    }
                    if (rho > 10.0f * cfg.rhoFloor) {
                        firstTorusRadius = std::min(firstTorusRadius, r);
                    }
                    report.pressureMax = std::max(report.pressureMax, uu / 3.0f);
                    report.b2Max = std::max(report.b2Max, b2);
                }
            }
        }
        report.innerEdgeRadius = firstTorusRadius;
        report.betaMin = report.pressureMax / std::max(report.b2Max, 1.0e-20f);
        report.massDrift = relativeChange(evolved.mass, initial.mass);
        report.internalEnergyDrift = relativeChange(evolved.internalEnergy, initial.internalEnergy);
        report.magneticEnergyDrift = relativeChange(evolved.magneticEnergy, initial.magneticEnergy);
        addCheck(report, "spin a", cfg.spin, 0.9375f, 1.0e-4f);
        addCheck(report, "density peak radius", report.peakRadius, 12.0f, 1.25f);
        addCheck(report, "torus inner edge radius", report.innerEdgeRadius, 6.0f, 1.25f);
        addCheck(report, "weak-loop beta pmax/B2max", report.betaMin, 100.0f, 35.0f);
        addCheck(report, "initial divB L1", initial.divBL1, 0.0f, 5.0e-3f);
        addCheck(report, "initial fail fraction", initial.failFrac, 0.0f, 1.0e-5f);
        if (frames > 0) {
            addCheck(report, "evolved fail fraction", evolved.failFrac, 0.0f, 5.0e-2f);
            addCheck(report, "evolved CFL", evolved.cfl, 0.0f, 1.25f);
            addCheck(report, "short-run mass drift", report.massDrift, 0.0f, 0.12f);
            addCheck(report, "short-run internal energy drift", report.internalEnergyDrift, 0.0f, 0.12f);
            addCheck(report, "short-run magnetic energy drift", report.magneticEnergyDrift, 0.0f, 0.35f);
        }
        return report;
    }

    static void attachSamples(FishboneReport& report, std::vector<Diagnostics> samples) {
        report.samples = std::move(samples);
    }

    static void writeMarkdown(const FishboneReport& report, std::ostream& os) {
        os << "# Fishbone-Moncrief Benchmark Report\n\n";
        os << "This is a quick Porth-style initial-data and short-evolution check for the HARM GRMHD engine.\n\n";
        os << "## Setup\n\n";
        os << "| Setting | Value |\n|---|---:|\n";
        os << "| spin `a` | " << report.cfg.spin << " |\n";
        os << "| grid | " << report.cfg.radialN << " x " << report.cfg.thetaN << " x " << report.cfg.phiN << " |\n";
        os << "| radial domain | " << report.cfg.rin << " to " << report.cfg.rout << " |\n";
        os << "| frames evolved | " << report.frames << " |\n\n";

        os << "## Checks\n\n";
        os << "| Check | Value | Target | Tolerance | Result |\n|---|---:|---:|---:|---|\n";
        for (const FishboneCheck& c : report.checks) {
            os << "| " << c.name << " | " << c.value << " | " << c.target << " | " << c.tolerance << " | " << (c.pass ? "PASS" : "FAIL") << " |\n";
        }

        os << "\n## Diagnostics\n\n";
        os << "| Drift | Relative change |\n|---|---:|\n";
        os << "| mass | " << report.massDrift << " |\n";
        os << "| internal energy | " << report.internalEnergyDrift << " |\n";
        os << "| magnetic energy | " << report.magneticEnergyDrift << " |\n";
        writeDiagnostics("initial", report.initial, os);
        if (report.frames > 0) {
            writeDiagnostics("evolved", report.evolved, os);
            writeTable2Style(report.samples, os);
        }
        os << "\nOverall: **" << (report.passed() ? "PASS" : "FAIL") << "**\n";
    }

private:
    static void addCheck(FishboneReport& report, const std::string& name, float value, float target, float tolerance) {
        FishboneCheck c{};
        c.name = name;
        c.value = value;
        c.target = target;
        c.tolerance = tolerance;
        c.pass = std::abs(value - target) <= tolerance;
        report.checks.push_back(c);
    }

    static float radiusAt(const Config& cfg, int ir, float logRange) {
        const float x = (static_cast<float>(ir) + 0.5f) / static_cast<float>(std::max(cfg.radialN, 1));
        return std::exp(std::log(cfg.rin) + x * logRange);
    }

    static void writeDiagnostics(const char* label, const Diagnostics& d, std::ostream& os) {
        os << "\n### " << label << "\n\n";
        os << "| Metric | Value |\n|---|---:|\n";
        os << "| mass | " << d.mass << " |\n";
        os << "| mdot | " << d.mdot << " |\n";
        os << "| phiBH | " << d.phiBH << " |\n";
        os << "| Ldot / Mdot | " << ratio(d.ldot, d.mdot) << " |\n";
        os << "| Edot / Mdot | " << ratio(d.edot, d.mdot) << " |\n";
        os << "| beta min | " << d.betaMin << " |\n";
        os << "| beta mean | " << d.betaMean << " |\n";
        os << "| sigma max | " << d.sigmaMax << " |\n";
        os << "| divB L1 | " << d.divBL1 << " |\n";
        os << "| divB max | " << d.divBMax << " |\n";
        os << "| floor mass fraction | " << d.floorMassFrac << " |\n";
        os << "| fail fraction | " << d.failFrac << " |\n";
        os << "| max Lorentz | " << d.maxLorentz << " |\n";
        os << "| CFL | " << d.cfl << " |\n";
        os << "| Qtheta | " << d.qTheta << " |\n";
        os << "| Qphi | " << d.qPhi << " |\n";
        os << "| Q product | " << d.qProduct << " |\n";
    }

    static float ratio(float numerator, float denominator) {
        return numerator / std::max(denominator, 1.0e-20f);
    }

    static float relativeChange(float current, float reference) {
        return std::abs(current - reference) / std::max(std::abs(reference), 1.0e-20f);
    }

    static void writeTable2Style(const std::vector<Diagnostics>& samples, std::ostream& os) {
        if (samples.empty()) return;
        auto value = [](const Diagnostics& d, int column) -> float {
            switch (column) {
                case 0: return d.mdot;
                case 1: return d.phiBH;
                case 2: return ratio(d.ldot, d.mdot);
                case 3: return ratio(d.edot, d.mdot);
                case 4: return d.divBL1;
                case 5: return d.failFrac;
                case 6: return d.qTheta;
                case 7: return d.qPhi;
                default: return 0.0f;
            }
        };
        const char* names[] = {
            "Mdot",
            "PhiBH/sqrt(Mdot)",
            "Ldot/Mdot",
            "Edot/Mdot",
            "divB L1",
            "fail fraction",
            "Qtheta",
            "Qphi",
        };
        os << "\n## Table-2-Style Time Series\n\n";
        os << "| Quantity | Peak | Mean | Stddev |\n|---|---:|---:|---:|\n";
        for (int col = 0; col < 8; ++col) {
            float peak = -std::numeric_limits<float>::infinity();
            float sum = 0.0f;
            for (const Diagnostics& d : samples) {
                const float v = value(d, col);
                peak = std::max(peak, v);
                sum += v;
            }
            const float mean = sum / static_cast<float>(samples.size());
            float var = 0.0f;
            for (const Diagnostics& d : samples) {
                const float dv = value(d, col) - mean;
                var += dv * dv;
            }
            const float stddev = std::sqrt(var / static_cast<float>(samples.size()));
            os << "| " << names[col] << " | " << peak << " | " << mean << " | " << stddev << " |\n";
        }
    }
};

} // namespace harm
