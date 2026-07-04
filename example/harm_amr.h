#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "harm_config.h"
#include "harm_geometry.h"
#include "harm_grid.h"
#include "harm_types.h"

namespace harm {

struct RefinementBlock {
    int ir0 = 0;
    int it0 = 0;
    int ip0 = 0;
    int nr = 0;
    int nt = 0;
    int np = 0;
    float score = 0.0f;
    float densityContrast = 0.0f;
    float sigmaMax = 0.0f;
    float floorFraction = 0.0f;
    std::string reason;
};

struct RefinementSummary {
    int blockSize = 8;
    int candidateBlocks = 0;
    int totalBlocks = 0;
    int coveredCells = 0;
    float coveredFraction = 0.0f;
    float maxScore = 0.0f;
    std::vector<RefinementBlock> topBlocks;
};

class AmrRefinementCriterion {
public:
    static RefinementSummary analyze(const Config& cfg, const std::vector<float>& packed, int blockSize = 8) {
        RefinementSummary summary{};
        summary.blockSize = std::clamp(blockSize, 4, 16);
        if (packed.size() < packedFloatCount(cfg.cellCount())) {
            return summary;
        }

        for (int ip0 = 0; ip0 < cfg.phiN; ip0 += summary.blockSize) {
            for (int it0 = 0; it0 < cfg.thetaN; it0 += summary.blockSize) {
                for (int ir0 = 0; ir0 < cfg.radialN; ir0 += summary.blockSize) {
                    RefinementBlock block = scoreBlock(cfg, packed, ir0, it0, ip0, summary.blockSize);
                    ++summary.totalBlocks;
                    summary.maxScore = std::max(summary.maxScore, block.score);
                    if (block.score >= 1.0f) {
                        ++summary.candidateBlocks;
                        summary.coveredCells += block.nr * block.nt * block.np;
                        insertTopBlock(summary.topBlocks, block);
                    }
                }
            }
        }

        summary.coveredFraction = static_cast<float>(summary.coveredCells) /
            std::max(static_cast<float>(cfg.cellCount()), 1.0f);
        return summary;
    }

private:
    static RefinementBlock scoreBlock(const Config& cfg, const std::vector<float>& packed, int ir0, int it0, int ip0, int blockSize) {
        RefinementBlock block{};
        block.ir0 = ir0;
        block.it0 = it0;
        block.ip0 = ip0;
        block.nr = std::min(blockSize, cfg.radialN - ir0);
        block.nt = std::min(blockSize, cfg.thetaN - it0);
        block.np = std::min(blockSize, cfg.phiN - ip0);

        float rhoMin = 1.0e30f;
        float rhoMax = 0.0f;
        float floorCells = 0.0f;
        float cells = 0.0f;
        for (int ip = ip0; ip < ip0 + block.np; ++ip) {
            for (int it = it0; it < it0 + block.nt; ++it) {
                for (int ir = ir0; ir < ir0 + block.nr; ++ir) {
                    const size_t idx = (static_cast<size_t>(ip) * static_cast<size_t>(cfg.thetaN) + static_cast<size_t>(it)) *
                        static_cast<size_t>(cfg.radialN) + static_cast<size_t>(ir);
                    const size_t base = idx * 12;
                    const float rho = std::max(packed[base + 0], cfg.rhoFloor);
                    const float b2 = packed[base + 5] * packed[base + 5] +
                        packed[base + 6] * packed[base + 6] +
                        packed[base + 7] * packed[base + 7];
                    rhoMin = std::min(rhoMin, rho);
                    rhoMax = std::max(rhoMax, rho);
                    block.sigmaMax = std::max(block.sigmaMax, b2 / std::max(rho, cfg.rhoFloor));
                    if (rho <= 1.01f * cfg.rhoFloor || packed[base + 1] <= 1.01f * cfg.uFloor) {
                        floorCells += 1.0f;
                    }
                    cells += 1.0f;
                }
            }
        }

        block.densityContrast = (rhoMax - rhoMin) / std::max(rhoMax + rhoMin, cfg.rhoFloor);
        block.floorFraction = floorCells / std::max(cells, 1.0f);

        const bool floorInterface = block.floorFraction > 0.02f && block.floorFraction < 0.85f;
        const float gradientScore = block.densityContrast / 0.35f;
        const float sigmaScore = block.sigmaMax / 10.0f;
        const float floorScore = floorInterface ? (block.floorFraction / 0.08f) : 0.0f;
        block.score = std::max(gradientScore, std::max(sigmaScore, floorScore));
        if (floorScore >= gradientScore && floorScore >= sigmaScore) {
            block.reason = "floor/funnel";
        } else if (sigmaScore >= gradientScore) {
            block.reason = "magnetization";
        } else {
            block.reason = "density gradient";
        }
        return block;
    }

    static void insertTopBlock(std::vector<RefinementBlock>& top, const RefinementBlock& block) {
        top.push_back(block);
        std::sort(top.begin(), top.end(), [](const RefinementBlock& a, const RefinementBlock& b) {
            return a.score > b.score;
        });
        if (top.size() > 8) {
            top.resize(8);
        }
    }
};

} // namespace harm
