#pragma once

#include "harm_controller.h"
#include "wgfx.h"

class Quad {
public:
    static Quad& Instance() {
        static Quad instance;
        return instance;
    }

    wgfx::Pipeline* pipeline = nullptr;

    void setHarmMode(bool play) {
        harm_.setAnimationPlayback(play);
    }

    void setMaxGridSize(int size) {
        harm_.setMaxGridSize(size);
        pipeline = harm_.pipeline;
    }

    void dispatchCompute3d() {
        // Compatibility hook: the HARM controller owns all compute dispatch now.
    }

    void dispatchComputeHarm() {
        harm_.dispatchCompute();
    }

    void afterFrameSubmit() {
        harm_.afterFrameSubmit();
    }

    void render(float dt) {
        harm_.render(dt);
        pipeline = harm_.pipeline;
    }

    void drawImGuiPanel() {
        harm_.drawUi();
    }

private:
    Quad() : harm_() {
        pipeline = harm_.pipeline;
    }

    Quad(const Quad&) = delete;
    void operator=(const Quad&) = delete;

    harm::Controller harm_;
};
