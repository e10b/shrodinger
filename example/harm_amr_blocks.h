#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "harm_config.h"
#include "harm_grid.h"
#include "harm_types.h"

namespace harm {

struct AmrBlock {
    int ir0 = 0;
    int it0 = 0;
    int ip0 = 0;
    int nr = 0;
    int nt = 0;
    int np = 0;
    int ghost = 2;
    std::vector<float> packed;

    int strideR() const {
        return nr + 2 * ghost;
    }

    int strideT() const {
        return nt + 2 * ghost;
    }

    int strideP() const {
        return np + 2 * ghost;
    }

    size_t cellCount() const {
        return static_cast<size_t>(strideR()) * static_cast<size_t>(strideT()) * static_cast<size_t>(strideP());
    }

    size_t localIndex(int ir, int it, int ip) const {
        const int r = std::clamp(ir + ghost, 0, strideR() - 1);
        const int t = std::clamp(it + ghost, 0, strideT() - 1);
        const int p = std::clamp(ip + ghost, 0, strideP() - 1);
        return (static_cast<size_t>(p) * static_cast<size_t>(strideT()) + static_cast<size_t>(t)) *
            static_cast<size_t>(strideR()) + static_cast<size_t>(r);
    }

    float* cell(int ir, int it, int ip) {
        return packed.data() + localIndex(ir, it, ip) * 12;
    }

    const float* cell(int ir, int it, int ip) const {
        return packed.data() + localIndex(ir, it, ip) * 12;
    }
};

class AmrBlockOps {
public:
    static AmrBlock loadWithGhosts(const Config& cfg, const Grid& grid, int ir0, int it0, int ip0, int nr, int nt, int np, int ghost = 2) {
        AmrBlock block{};
        block.ir0 = std::clamp(ir0, 0, cfg.radialN - 1);
        block.it0 = std::clamp(it0, 0, cfg.thetaN - 1);
        block.ip0 = std::clamp(ip0, 0, cfg.phiN - 1);
        block.nr = std::clamp(nr, 1, cfg.radialN - block.ir0);
        block.nt = std::clamp(nt, 1, cfg.thetaN - block.it0);
        block.np = std::clamp(np, 1, cfg.phiN - block.ip0);
        block.ghost = std::clamp(ghost, 1, 4);
        block.packed.assign(packedFloatCount(block.cellCount()), 0.0f);

        for (int ip = -block.ghost; ip < block.np + block.ghost; ++ip) {
            for (int it = -block.ghost; it < block.nt + block.ghost; ++it) {
                for (int ir = -block.ghost; ir < block.nr + block.ghost; ++ir) {
                    const size_t global = grid.index(cfg, block.ir0 + ir, block.it0 + it, block.ip0 + ip);
                    const float* src = grid.cell(global);
                    float* dst = block.cell(ir, it, ip);
                    for (int c = 0; c < 12; ++c) {
                        dst[c] = src[c];
                    }
                }
            }
        }
        return block;
    }

    static void injectInterior(const Config& cfg, const AmrBlock& block, Grid& grid) {
        for (int ip = 0; ip < block.np; ++ip) {
            for (int it = 0; it < block.nt; ++it) {
                for (int ir = 0; ir < block.nr; ++ir) {
                    const size_t global = grid.index(cfg, block.ir0 + ir, block.it0 + it, block.ip0 + ip);
                    const float* src = block.cell(ir, it, ip);
                    float* dst = grid.cell(global);
                    for (int c = 0; c < 12; ++c) {
                        dst[c] = src[c];
                    }
                }
            }
        }
    }

    static AmrBlock restrict2x(const AmrBlock& fine) {
        AmrBlock coarse{};
        coarse.ir0 = fine.ir0;
        coarse.it0 = fine.it0;
        coarse.ip0 = fine.ip0;
        coarse.nr = std::max(fine.nr / 2, 1);
        coarse.nt = std::max(fine.nt / 2, 1);
        coarse.np = std::max(fine.np / 2, 1);
        coarse.ghost = fine.ghost;
        coarse.packed.assign(packedFloatCount(coarse.cellCount()), 0.0f);

        for (int ip = 0; ip < coarse.np; ++ip) {
            for (int it = 0; it < coarse.nt; ++it) {
                for (int ir = 0; ir < coarse.nr; ++ir) {
                    float* dst = coarse.cell(ir, it, ip);
                    for (int fp = 0; fp < 2; ++fp) {
                        for (int ft = 0; ft < 2; ++ft) {
                            for (int fr = 0; fr < 2; ++fr) {
                                const float* src = fine.cell(2 * ir + fr, 2 * it + ft, 2 * ip + fp);
                                for (int c = 0; c < 12; ++c) {
                                    dst[c] += 0.125f * src[c];
                                }
                            }
                        }
                    }
                }
            }
        }
        return coarse;
    }

    static AmrBlock prolong2xNearest(const AmrBlock& coarse) {
        AmrBlock fine{};
        fine.ir0 = coarse.ir0;
        fine.it0 = coarse.it0;
        fine.ip0 = coarse.ip0;
        fine.nr = std::max(coarse.nr * 2, 1);
        fine.nt = std::max(coarse.nt * 2, 1);
        fine.np = std::max(coarse.np * 2, 1);
        fine.ghost = coarse.ghost;
        fine.packed.assign(packedFloatCount(fine.cellCount()), 0.0f);

        for (int ip = 0; ip < fine.np; ++ip) {
            for (int it = 0; it < fine.nt; ++it) {
                for (int ir = 0; ir < fine.nr; ++ir) {
                    const float* src = coarse.cell(ir / 2, it / 2, ip / 2);
                    float* dst = fine.cell(ir, it, ip);
                    for (int c = 0; c < 12; ++c) {
                        dst[c] = src[c];
                    }
                }
            }
        }
        return fine;
    }

    static float interiorRelativeL1(const AmrBlock& a, const AmrBlock& b, int component) {
        const int nr = std::min(a.nr, b.nr);
        const int nt = std::min(a.nt, b.nt);
        const int np = std::min(a.np, b.np);
        float sum = 0.0f;
        int count = 0;
        for (int ip = 0; ip < np; ++ip) {
            for (int it = 0; it < nt; ++it) {
                for (int ir = 0; ir < nr; ++ir) {
                    const float av = a.cell(ir, it, ip)[component];
                    const float bv = b.cell(ir, it, ip)[component];
                    sum += std::abs(av - bv) / std::max(std::abs(av), 1.0e-12f);
                    ++count;
                }
            }
        }
        return sum / std::max(static_cast<float>(count), 1.0f);
    }

    static float interiorNormalizedL1(const AmrBlock& a, const AmrBlock& b, int component) {
        const int nr = std::min(a.nr, b.nr);
        const int nt = std::min(a.nt, b.nt);
        const int np = std::min(a.np, b.np);
        float sum = 0.0f;
        float scale = 0.0f;
        int count = 0;
        for (int ip = 0; ip < np; ++ip) {
            for (int it = 0; it < nt; ++it) {
                for (int ir = 0; ir < nr; ++ir) {
                    const float av = a.cell(ir, it, ip)[component];
                    const float bv = b.cell(ir, it, ip)[component];
                    sum += std::abs(av - bv);
                    scale = std::max(scale, std::abs(av));
                    ++count;
                }
            }
        }
        return (sum / std::max(static_cast<float>(count), 1.0f)) / std::max(scale, 1.0e-12f);
    }
};

} // namespace harm
