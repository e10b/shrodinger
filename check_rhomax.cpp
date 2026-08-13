#include <iostream>
#include "example/harm_config.h"
#include "example/harm_grid.h"
#include "example/harm_initial_data.h"

int main() {
    harm::Config cfg;
    harm::Grid grid;

    cfg.setMaxGrid(64);
    harm::Diagnostics d64 = harm::InitialDataBuilder::build(cfg, grid);
    std::cout << "64: rhoMax=" << d64.rhoMax << " mass=" << d64.mass << "\n";

    cfg.setMaxGrid(192);
    harm::Diagnostics d192 = harm::InitialDataBuilder::build(cfg, grid);
    std::cout << "192: rhoMax=" << d192.rhoMax << " mass=" << d192.mass << "\n";
    return 0;
}
