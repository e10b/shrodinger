#pragma once

#include <algorithm>

#include "SDL3/SDL.h"
#include "context.h"
#include "imgui.h"
#include "wgfx.h"

#include "harm_camera.h"
#include "harm_config.h"
#include "harm_fullscreen_quad.h"
#include "harm_gpu_compute.h"
#include "harm_grid.h"
#include "harm_initial_data.h"
#include "harm_renderer.h"
#include "harm_types.h"

namespace harm {

class Controller {
public:
    wgfx::Pipeline* pipeline = nullptr;
    wgfx::ComputePass computePass;

    Controller() {
        cfg_.clamp();
        quad_.init();
        grid_.resize(cfg_);
        gpu_.init(cfg_);
        renderer_.init(quad_, gpu_);
        reset();
        pipeline = renderer_.pipeline;
    }

    void setAnimationPlayback(bool play) {
        camera_.playing = play;
    }

    void setMaxGridSize(int size) {
        cfg_.setMaxGrid(size);
        reset();
    }

    void dispatchCompute() {
        if (!cfg_.useGpu) return;
        ensureUploaded();
        computePass.prepare();
        gpu_.dispatch(computePass, cfg_, time_);
        computePass.end();
        gpu_.queueReadback(cfg_);
    }

    void afterFrameSubmit() {
        gpu_.consumeReadback(cfg_, grid_, diagnostics_);
    }

    void render(float) {
        if (!cfg_.useGpu) {
            ensureUploaded();
        }
        int width = 1280;
        int height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);
        camera_.process(width, height, aspect);
        camera_.applyAnimation(time_);
        renderer_.update(cfg_, camera_, time_, aspect);
        pipeline = renderer_.pipeline;
    }

    void drawUi() {
        ImGui::Begin("HARM GRMHD");
        int n = cfg_.radialN;
        int nTheta = cfg_.thetaN;
        int nPhi = cfg_.phiN;
        int substeps = cfg_.substeps;
        int viewMode = cfg_.viewMode;
        int initMode = cfg_.initialData;
        const float oldRout = cfg_.rout;
        const float oldSpin = cfg_.spin;
        const float oldLoop = cfg_.magneticLoop;
        const int oldInitMode = cfg_.initialData;

        ImGui::Combo("view##harm", &viewMode, "density\0magnetization\0plasma beta\0radial 4-velocity\0primitive fail\0shadow image\0vertical slice\0azimuth slice\0evolved div B\0evolved flux\0evolved accretion\0volume render\0");
        ImGui::Combo("initial data##harm", &initMode, "SANE torus\0MAD torus\0");
        ImGui::SliderInt("radial N##harm", &n, 32, cfg_.maxGrid);
        ImGui::SliderInt("theta N##harm", &nTheta, 16, cfg_.maxGrid);
        ImGui::SliderInt("phi N##harm", &nPhi, 32, cfg_.maxGrid);
        ImGui::SliderInt("substeps/frame##harm", &substeps, 1, 12);
        ImGui::SliderFloat("CFL dt##harm", &cfg_.dt, 0.0002f, 0.02f, "%.5f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("event horizon r_in##harm", &cfg_.rin, 0.1f, 5.0f, "%.2f");
        ImGui::SliderFloat("r out##harm", &cfg_.rout, 12.0f, 80.0f, "%.1f");
        ImGui::SliderFloat("spin a##harm", &cfg_.spin, -0.98f, 0.98f, "%.2f");
        ImGui::SliderFloat("magnetic loop##harm", &cfg_.magneticLoop, 0.0f, 0.18f, "%.3f");
        ImGui::SliderFloat("rho floor##harm", &cfg_.rhoFloor, 1e-6f, 1e-3f, "%.6f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("u floor##harm", &cfg_.uFloor, 1e-7f, 1e-3f, "%.7f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("color scale##harm", &cfg_.colorScale, 0.2f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("camera yaw##harm", &camera_.yaw, -kPi, kPi, "%.2f");
        ImGui::SliderFloat("camera inclination##harm", &camera_.inclination, 0.05f, 1.45f, "%.2f");
        ImGui::Checkbox("enable real gravity / event horizon##harm", &cfg_.enableGravity);
        ImGui::Text("Gravitational Lensing Mode");
        ImGui::RadioButton("1st-Order (Schwarzschild, Fast)##harm", &cfg_.lensingMode, 0);
        ImGui::RadioButton("2nd-Order RK2 (Kerr, Balanced)##harm", &cfg_.lensingMode, 1);
        ImGui::RadioButton("4th-Order RK4 (Kerr, Exact)##harm", &cfg_.lensingMode, 2);
        ImGui::Checkbox("high-order MUSCL##harm", &cfg_.highOrder);
        ImGui::Checkbox("GPU compute##harm", &cfg_.useGpu);
        ImGui::Checkbox("live GPU diagnostics##harm", &cfg_.liveGpuDiagnostics);
        ImGui::SliderInt("diagnostic readback interval##harm", &cfg_.readbackInterval, 1, 120);
        ImGui::Checkbox("paused##harm", &cfg_.paused);
        ImGui::SameLine();
        if (ImGui::Button("Reset torus##harm")) {
            reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("Raytraced view##harm")) {
            cfg_.viewMode = 5;
            viewMode = 5;
        }
        ImGui::SameLine();
        ImGui::Text("t = %.2f", time_);
        ImGui::Text("3D grid: %d x %d x %d", cfg_.radialN, cfg_.thetaN, cfg_.phiN);
        ImGui::Text("beta min %.2f  <beta> %.1f  phi_BH %.2f  sigma max %.2f",
            diagnostics_.betaMin, diagnostics_.betaMean, diagnostics_.phiBH, diagnostics_.sigmaMax);
        ImGui::Text("M %.4e  Mdot %.4e  U %.4e  EB %.4e",
            diagnostics_.mass, diagnostics_.mdot, diagnostics_.internalEnergy, diagnostics_.magneticEnergy);
        ImGui::Text("Lz %.4e  divB L1 %.3e  divB max %.3e  CFL %.3f",
            diagnostics_.angularMomentum, diagnostics_.divBL1, diagnostics_.divBMax, diagnostics_.cfl);
        ImGui::Text("floor mass %.2f%%  fail %.2f%%  gamma max %.2f",
            100.0f * diagnostics_.floorMassFrac, 100.0f * diagnostics_.failFrac, diagnostics_.maxLorentz);
        if (cfg_.useGpu && !cfg_.liveGpuDiagnostics) {
            ImGui::TextWrapped("Diagnostics describe the initial/uploaded field. Enable live GPU diagnostics to sample the evolved GPU state.");
        }
        ImGui::Text("Left-drag orbit, right/middle-drag pan, wheel zoom");

        ImGui::Separator();
        ImGui::Text("Camera Keyframes");
        if (ImGui::Button("Add Keyframe")) {
            camera_.keyframes.push_back({ time_, camera_.yaw, camera_.inclination, camera_.zoom });
            std::sort(camera_.keyframes.begin(), camera_.keyframes.end(), [](const CameraKeyframe& a, const CameraKeyframe& b) {
                return a.time < b.time;
            });
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            camera_.keyframes.clear();
        }
        ImGui::SameLine();
        ImGui::Checkbox("Play Anim", &camera_.playing);
        if (!camera_.keyframes.empty()) {
            ImGui::Text("%zu keyframes (%.1f to %.1f)", camera_.keyframes.size(), camera_.keyframes.front().time, camera_.keyframes.back().time);
        }

        cfg_.radialN = std::clamp(n, 32, cfg_.maxGrid);
        cfg_.thetaN = std::clamp(nTheta, 16, cfg_.maxGrid);
        cfg_.phiN = std::clamp(nPhi, 16, cfg_.maxGrid);
        cfg_.substeps = std::clamp(substeps, 1, 12);
        cfg_.viewMode = std::clamp(viewMode, 0, 11);
        cfg_.initialData = std::clamp(initMode, 0, 1);
        cfg_.clamp();

        if (cfg_.radialN != previousRadialN_ || cfg_.thetaN != previousThetaN_ || cfg_.phiN != previousPhiN_ ||
            oldRout != cfg_.rout || oldSpin != cfg_.spin || oldLoop != cfg_.magneticLoop || oldInitMode != cfg_.initialData) {
            reset();
        }
        ImGui::End();
    }

private:
    Config cfg_{};
    Grid grid_{};
    Diagnostics diagnostics_{};
    FullscreenQuad quad_{};
    GpuCompute gpu_{};
    Renderer renderer_{};
    CameraController camera_{};
    float time_ = 0.0f;
    bool uploaded_ = false;
    int previousRadialN_ = 64;
    int previousThetaN_ = 32;
    int previousPhiN_ = 64;

    void reset() {
        cfg_.clamp();
        grid_.resize(cfg_);
        diagnostics_ = InitialDataBuilder::build(cfg_, grid_);
        time_ = 0.0f;
        uploaded_ = false;
        previousRadialN_ = cfg_.radialN;
        previousThetaN_ = cfg_.thetaN;
        previousPhiN_ = cfg_.phiN;
    }

    void ensureUploaded() {
        if (uploaded_) return;
        gpu_.uploadInitial(grid_);
        uploaded_ = true;
    }
};

} // namespace harm
