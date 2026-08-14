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

    void setPaused(bool paused) {
        harm_.setPaused(paused);
    }

    void setTimeStep(float dt) {
        harm_.setTimeStep(dt);
    }

    void setSubsteps(int substeps) {
        harm_.setSubsteps(substeps);
    }

    void setViewMode(int viewMode) {
        harm_.setViewMode(viewMode);
    }

    void setLensingMode(int lensingMode) {
        harm_.setLensingMode(lensingMode);
    }

    void setGravityEnabled(bool enabled) {
        harm_.setGravityEnabled(enabled);
    }

    void setColorScale(float colorScale) {
        harm_.setColorScale(colorScale);
    }

    void setCameraInclination(float inclination) {
        harm_.setCameraInclination(inclination);
    }

    void applyChaosDemoPreset() {
        harm_.applyChaosDemoPreset();
        pipeline = harm_.pipeline;
    }

    void applyMadChaosDemoPreset() {
        harm_.applyMadChaosDemoPreset();
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
