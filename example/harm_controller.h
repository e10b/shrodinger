#pragma once

#include <algorithm>

#include "SDL3/SDL.h"
#include "context.h"
#include "imgui.h"
#include "wgfx.h"

#include "harm_camera.h"
#include "harm_config.h"
#include "harm_cpu_solver.h"
#include "harm_fullscreen_quad.h"
#include "harm_gpu_compute.h"
#include "harm_grid.h"
#include "harm_initial_data.h"
#include "harm_flux.h"
#include "harm_primitive_recovery.h"
#include "harm_constrained_transport.h"
#include "harm_renderer.h"
#include "harm_types.h"

namespace harm {

class Controller {
public:
    wgfx::Pipeline* pipeline = nullptr;
    wgfx::ComputePass computePass;

    static void setStartupMaxGridSize(int size) {
        startupMaxGridSize_ = size;
    }

    static void setStartupChaosDemo(bool enabled) {
        startupDemoMode_ = enabled ? 1 : 0;
    }

    static void setStartupMadChaosDemo(bool enabled) {
        startupDemoMode_ = enabled ? 2 : 0;
    }

    Controller() {
        if (startupMaxGridSize_ > 0) {
            cfg_.setMaxGrid(startupMaxGridSize_);
        }
        if (startupDemoMode_ > 0) {
            configureChaosDemo(startupDemoMode_ == 2);
        }
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
        const int oldMaxGrid = cfg_.maxGrid;
        cfg_.setMaxGrid(size);
        if (cfg_.maxGrid != oldMaxGrid) {
            gpu_.init(cfg_);
            renderer_.init(quad_, gpu_);
            pipeline = renderer_.pipeline;
        }
        reset();
    }

    void setPaused(bool paused) {
        cfg_.paused = paused;
    }

    void setTimeStep(float dt) {
        cfg_.dt = dt;
        cfg_.clamp();
    }

    void setSubsteps(int substeps) {
        cfg_.substeps = substeps;
        cfg_.clamp();
    }

    void setViewMode(int viewMode) {
        cfg_.viewMode = viewMode;
        cfg_.clamp();
    }

    void setLensingMode(int lensingMode) {
        cfg_.lensingMode = lensingMode;
        cfg_.clamp();
    }

    void setGravityEnabled(bool enabled) {
        cfg_.enableGravity = enabled;
    }

    void setColorScale(float colorScale) {
        cfg_.colorScale = std::max(colorScale, 0.001f);
    }

    void setCameraInclination(float inclination) {
        camera_.inclination = std::clamp(inclination, 0.0f, 0.5f * kPi);
    }

    void applyChaosDemoPreset() {
        // Keep the production/Porth defaults intact, but make a deliberately
        // coarse movie configuration that advances enough physical time per
        // displayed frame for orbital motion to be obvious on consumer GPUs.
        configureChaosDemo(false);
        reset();
    }

    void applyMadChaosDemoPreset() {
        configureChaosDemo(true);
        reset();
    }

private:
    void configureChaosDemo(bool mad) {
        cfg_.radialN = 32;
        cfg_.thetaN = 32;
        cfg_.phiN = 32;
        cfg_.initialData = mad ? 1 : 0;
        cfg_.substeps = 32;
        cfg_.dt = 0.03f;
        cfg_.viewMode = 5;
        cfg_.lensingMode = 0;
        cfg_.enableGravity = true;
        cfg_.highOrder = true;
        cfg_.useGpu = true;
        cfg_.liveGpuDiagnostics = false;
        cfg_.paused = false;
        cfg_.magneticLoop = mad ? 0.12f : 0.055f;
        cfg_.colorScale = mad ? 1.85f : 1.45f;
        camera_.yaw = mad ? -0.62f : -0.35f;
        camera_.inclination = (mad ? 64.0f : 58.0f) * (kPi / 180.0f);
        camera_.zoom = 0.070f;
        presentationMode_ = mad ? 2 : 1;
    }

public:

    void dispatchCompute() {
        if (!cfg_.useGpu) {
            CpuSolver::step(cfg_, grid_, diagnostics_, time_);
            uploaded_ = false;
            return;
        }
        ensureUploaded();
        computePass.prepare();
        gpu_.dispatch(computePass, cfg_, time_);
        computePass.end();
        gpu_.queueReadback(cfg_);
    }

    void afterFrameSubmit() {
        if (gpu_.consumeReadback(cfg_, grid_, diagnostics_)) {
            time_ = gpu_.simulatedTime();
        }
    }

    void render(float) {
        if (!cfg_.useGpu) {
            ensureUploaded();
        }
        int width = 1280;
        int height = 720;
        Context& context = Context::Instance();
        if (!context.headless && context.window) {
            SDL_GetWindowSize(context.window, &width, &height);
        } else {
            width = std::max(wgfx::width, 1);
            height = std::max(wgfx::height, 1);
        }
        const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);
        if (!context.headless && context.window) {
            camera_.process(width, height, aspect);
        }
        camera_.applyAnimation(time_);
        renderer_.update(cfg_, camera_, time_, aspect, presentationMode_);
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

        if (ImGui::Button("Fast chaos demo##harm")) {
            applyChaosDemoPreset();
            n = cfg_.radialN;
            nTheta = cfg_.thetaN;
            nPhi = cfg_.phiN;
            substeps = cfg_.substeps;
            viewMode = cfg_.viewMode;
            initMode = cfg_.initialData;
        }
        ImGui::SameLine();
        if (ImGui::Button("MAD CHAOS##harm")) {
            applyMadChaosDemoPreset();
            n = cfg_.radialN;
            nTheta = cfg_.thetaN;
            nPhi = cfg_.phiN;
            substeps = cfg_.substeps;
            viewMode = cfg_.viewMode;
            initMode = cfg_.initialData;
        }
        if (presentationMode_ == 1) {
            ImGui::TextWrapped("FAST DEMO: real CFL-safe GPU evolution plus 60x velocity-advected flow tracers so orbital motion is visible. Tracer speed is illustrative; this is not a Porth validation run.");
        } else if (presentationMode_ == 2) {
            ImGui::TextWrapped("MAD CHAOS: strongly magnetized MAD initial data with asymmetric fluid perturbations, plus accelerated magnetic knots/streams for immediate drama. Cinematic presentation, not validation evidence.");
        }

        ImGui::Combo("view##harm", &viewMode, "density\0magnetization\0plasma beta\0radial 4-velocity\0primitive fail\0shadow image\0vertical slice\0azimuth slice\0evolved div B\0evolved flux\0evolved accretion\0volume render\0");
        ImGui::Combo("initial data##harm", &initMode, "SANE torus\0MAD torus\0Porth 2019 common SANE setup\0");
        if (initMode == 2) {
            int cubicN = n;
            ImGui::SliderInt("cubic N (Porth)##harm", &cubicN, 32, cfg_.maxGrid);
            n = cubicN;
            nTheta = cubicN;
            nPhi = cubicN;
        } else {
            ImGui::SliderInt("radial N##harm", &n, 32, cfg_.maxGrid);
            ImGui::SliderInt("theta N##harm", &nTheta, 16, cfg_.maxGrid);
            ImGui::SliderInt("phi N##harm", &nPhi, 32, cfg_.maxGrid);
        }
        ImGui::SliderInt("substeps/frame##harm", &substeps, 1, Config::kMaxSubstepsPerFrame);
        ImGui::SliderFloat("requested dt ceiling##harm", &cfg_.dt, 0.0002f, 0.03f, "%.5f", ImGuiSliderFlags_Logarithmic);
        if (cfg_.useGpu) {
            ImGui::Text("actual CFL-limited dt: %.6f", gpu_.actualTimeStep());
        }
        ImGui::SliderFloat("event horizon r_in##harm", &cfg_.rin, 0.1f, 5.0f, "%.2f");
        ImGui::SliderFloat("r out##harm", &cfg_.rout, 12.0f, 80.0f, "%.1f");
        ImGui::SliderFloat("spin a##harm", &cfg_.spin, -0.98f, 0.98f, "%.2f");
        ImGui::SliderFloat("magnetic loop##harm", &cfg_.magneticLoop, 0.0f, 0.18f, "%.3f");
        ImGui::SliderFloat("rho floor##harm", &cfg_.rhoFloor, 1e-6f, 1e-3f, "%.6f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("u floor##harm", &cfg_.uFloor, 1e-7f, 1e-3f, "%.7f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("color scale##harm", &cfg_.colorScale, 0.2f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("camera yaw##harm", &camera_.yaw, -kPi, kPi, "%.2f");
        float inclinationDegrees = camera_.inclination * (180.0f / kPi);
        if (ImGui::SliderFloat("camera inclination (0=face-on)##harm", &inclinationDegrees, 0.0f, 90.0f, "%.1f deg")) {
            camera_.inclination = inclinationDegrees * (kPi / 180.0f);
        }
        ImGui::Checkbox("enable real gravity / event horizon##harm", &cfg_.enableGravity);
        ImGui::Text("Gravitational Lensing Mode");
        ImGui::RadioButton("Schwarzschild midpoint (Fast)##harm", &cfg_.lensingMode, 0);
        ImGui::RadioButton("Kerr-like RK2 (Experimental)##harm", &cfg_.lensingMode, 1);
        ImGui::RadioButton("Kerr-like RK4 (Experimental)##harm", &cfg_.lensingMode, 2);
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
        if (cfg_.initialData == 2) {
            ImGui::TextWrapped("Porth comparison evolution setup; the shadow image is a diagnostic fast-light emissivity view, not validated GRRT.");
        }
        ImGui::Text("3D grid: %d x %d x %d", cfg_.radialN, cfg_.thetaN, cfg_.phiN);
        ImGui::Text("beta min %.2f  <beta> %.1f  phi_BH %.2f  sigma max %.2f",
            diagnostics_.betaMin, diagnostics_.betaMean, diagnostics_.phiBH, diagnostics_.sigmaMax);
        ImGui::Text("M %.4e  Mdot %.4e  U %.4e  EB %.4e",
            diagnostics_.mass, diagnostics_.mdot, diagnostics_.internalEnergy, diagnostics_.magneticEnergy);
        ImGui::Text("Lz %.4e  divB L1 %.3e  divB max %.3e  CFL %.3f",
            diagnostics_.angularMomentum, diagnostics_.divBL1, diagnostics_.divBMax, diagnostics_.cfl);
        ImGui::Text("floor mass %.2f%%  fail %.2f%%  gamma max %.2f",
            100.0f * diagnostics_.floorMassFrac, 100.0f * diagnostics_.failFrac, diagnostics_.maxLorentz);
        ImGui::Text("MRI Qtheta %.2f  Qphi %.2f  product %.1f",
            diagnostics_.qTheta, diagnostics_.qPhi, diagnostics_.qProduct);
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
        cfg_.substeps = std::clamp(substeps, 1, Config::kMaxSubstepsPerFrame);
        cfg_.viewMode = std::clamp(viewMode, 0, 11);
        cfg_.initialData = std::clamp(initMode, 0, 2);
        cfg_.clamp();

        if (cfg_.radialN != 32 || cfg_.thetaN != 32 || cfg_.phiN != 32 ||
            ((presentationMode_ == 1 && cfg_.initialData != 0) ||
             (presentationMode_ == 2 && cfg_.initialData != 1))) {
            presentationMode_ = 0;
        }

        if (cfg_.radialN != previousRadialN_ || cfg_.thetaN != previousThetaN_ || cfg_.phiN != previousPhiN_ ||
            oldRout != cfg_.rout || oldSpin != cfg_.spin || oldLoop != cfg_.magneticLoop || oldInitMode != cfg_.initialData) {
            reset();
        }
        ImGui::End();
    }

private:
    inline static int startupMaxGridSize_ = 0;
    inline static int startupDemoMode_ = 0;
    Config cfg_{};
    Grid grid_{};
    Diagnostics diagnostics_{};
    FullscreenQuad quad_{};
    GpuCompute gpu_{};
    Renderer renderer_{};
    CameraController camera_{};
    float time_ = 0.0f;
    bool uploaded_ = false;
    int previousRadialN_ = 96;
    int previousThetaN_ = 96;
    int previousPhiN_ = 96;
    int presentationMode_ = 0;

    void reset() {
        cfg_.clamp();
        if (!gpu_.matchesConfig(cfg_)) {
            gpu_.init(cfg_);
            renderer_.init(quad_, gpu_);
            pipeline = renderer_.pipeline;
        }
        grid_.resize(cfg_);
        diagnostics_ = InitialDataBuilder::build(cfg_, grid_);
        time_ = 0.0f;
        uploaded_ = false;
        previousRadialN_ = cfg_.radialN;
        previousThetaN_ = cfg_.thetaN;
        previousPhiN_ = cfg_.phiN;
    }

    void ensureUploaded() {
        if (!gpu_.matchesConfig(cfg_)) {
            gpu_.init(cfg_);
            renderer_.init(quad_, gpu_);
            pipeline = renderer_.pipeline;
            uploaded_ = false;
        }
        if (uploaded_) return;
        gpu_.uploadInitial(cfg_, grid_);
        uploaded_ = gpu_.matchesConfig(cfg_);
    }
};

} // namespace harm
