#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifdef ATOMS_ENABLE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

#ifdef ATOMS_ENABLE_TORCHSCRIPT
#include <torch/script.h>
#endif

#include "context.h"
#include "imgui.h"
#include "slater_vmc.h"
#include "wgfx.h"

struct QuantumState {
    int n = 9;
    int l = 3;
    int m = 1;
    int sampleCount = 100000;

    void clamp() {
        n = std::clamp(n, 1, 30);
        l = std::clamp(l, 0, n - 1);
        if (l == 14) {
            l = 13;
        }
        m = std::clamp(m, -l, l);
    }
};

struct ClipState {
    glm::vec3 origin = glm::vec3(0.0f);
    int removedOctant = 1;
};

class OrbitCamera {
public:
    glm::vec3 target = glm::vec3(-40.050f, 1.792f, 21.270f);
    float radius = 548.800f;
    float azimuth = 1.051212f;   // 60.23 deg
    float elevation = 1.378810f; // 79.00 deg
    float orbitSpeed = 0.01f;
    float zoomSpeed = 10.0f;

    // Reset to sensible defaults for a given domain half-extent
    void resetForDomain(float domainHalf) {
        target    = glm::vec3(0.0f);
        radius    = domainHalf * 3.0f;
        azimuth   = 0.8f;
        elevation = 1.1f;
        dragging_ = false;
    }

    glm::vec3 position() const {
        float e = glm::clamp(elevation, 0.01f, 3.14159265358979323846f - 0.01f);
        return target + glm::vec3(
            radius * std::sin(e) * std::cos(azimuth),
            radius * std::cos(e),
            radius * std::sin(e) * std::sin(azimuth)
        );
    }

    void process(float dt, bool allowMouseCapture, float wheelDelta) {
        float mx = 0.0f;
        float my = 0.0f;

        if (allowMouseCapture) {
            Uint32 mask = SDL_GetMouseState(&mx, &my);
            bool draggingNow = (mask & SDL_BUTTON_LMASK) != 0 || (mask & SDL_BUTTON_MMASK) != 0;
            if (draggingNow) {
                if (!dragging_) {
                    dragging_ = true;
                    lastX_ = mx;
                    lastY_ = my;
                } else {
                    float dx = mx - lastX_;
                    float dy = my - lastY_;
                    azimuth += dx * orbitSpeed;
                    elevation -= dy * orbitSpeed;
                    elevation = glm::clamp(elevation, 0.01f, 3.14159265358979323846f - 0.01f);
                    lastX_ = mx;
                    lastY_ = my;
                }
            } else {
                dragging_ = false;
            }
        }

        if (wheelDelta != 0.0f) {
            radius -= wheelDelta * zoomSpeed;
        }

        const Uint8* keys = SDL_GetKeyboardState(nullptr);
        if (keys[SDL_SCANCODE_EQUALS]) radius -= zoomSpeed * dt * 20.0f;
        if (keys[SDL_SCANCODE_MINUS]) radius += zoomSpeed * dt * 20.0f;
        radius = std::max(radius, 1.0f);
    }

private:
    bool dragging_ = false;
    float lastX_ = 0.0f;
    float lastY_ = 0.0f;
};

class Quad {
public:
    static Quad& Instance() {
        static Quad instance;
        return instance;
    }

    wgfx::Pipeline* pipeline = nullptr;

    // Called from main.cpp before the render pass when in 3D TDSE mode.
    // The ComputePass must be begun/ended around this call.
    wgfx::ComputePass computePass3d;
    wgfx::ComputePass computePassHarm;

    void dispatchCompute3d() {
        if (renderPath_ != RenderPath::Path3D) return;
        computePass3d.prepare();
        dispatchCompute(computePass3d);
        computePass3d.end();
    }

    void dispatchComputeHarm() {
        if (renderPath_ != RenderPath::PathHarmGRMHD || !harmUseGpu_) return;
        computePassHarm.prepare();
        dispatchHarmCompute(computePassHarm);
        computePassHarm.end();
    }

    void render(float dt) {
        if (renderPath_ == RenderPath::Path2D) {
            render2d(dt);
            return;
        }
        if (renderPath_ == RenderPath::Path3D) {
            render3d(dt);
            return;
        }
        if (renderPath_ == RenderPath::PathGRMHD) {
            renderGrmhd(dt);
            return;
        }
        if (renderPath_ == RenderPath::PathHarmGRMHD) {
            renderHarmGrmhd(dt);
            return;
        }

        processShortcuts();
        advanceSimulation(dt);

        const bool vmcActive = vmcMode_ && vmcConfigured_;

        if (vmcActive) {
            vmc_.setWalkerCount(vmcWalkerCount_);
            vmc_.setParameters(vmcStepSize_, vmcZetaScale_, vmcJastrowBeta_);
            vmc_.setMaxCloudPoints(static_cast<size_t>(quantum_.sampleCount));
            vmc_.step(vmcSweepsPerFrame_, vmcThermalizationSweeps_, vmcMeasureEvery_);
            rebuildVmcPointBuffer();
            updateCameraForVmc();
        } else {
            if (orbitalBufferDirty_) {
                rebuildOrbitalAttributeBuffer();
                orbitalBufferDirty_ = false;
            }
        }

        ImGuiIO& io = ImGui::GetIO();
        float wheel = Context::Instance().consumeWheelDelta();
        camera_.process(dt, !io.WantCaptureMouse, wheel);

        int width = 1280;
        int height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 2000.0f);
        glm::mat4 view = glm::lookAt(camera_.position(), camera_.target, glm::vec3(0, 1, 0));
        gpuState_.viewProj = projection * view;

        gpuState_.clipOrigin = glm::vec4(clip_.origin, intensityRange_);
        gpuState_.quantum = glm::vec4(
            vmcActive ? -1.0f : static_cast<float>(quantum_.n),
            static_cast<float>(quantum_.l),
            static_cast<float>(quantum_.m),
            static_cast<float>(colorMode_)
        );
        gpuState_.render = glm::vec4(
            static_cast<float>(clip_.removedOctant),
            intensityScale_,
            simulationTime_,
            sampleSeed_
        );

        writeRenderUniform(pipeline, reinterpret_cast<const float*>(&gpuState_));

        if (vmcActive) {
            ibo_->indexCount = static_cast<uint32_t>(std::clamp(vmcDrawCount_, 1, kMaxParticles));
        } else {
            ibo_->indexCount = static_cast<uint32_t>(quantum_.sampleCount);
        }
        pipeline->setVertexBuffer(vbo_.get());
        pipeline->setIndexBuffer(ibo_.get());
    }

    void drawImGuiPanel() {
        ImGui::Begin("Orbital Controls");
        int pathIndex = static_cast<int>(renderPath_);
        if (ImGui::Combo("path", &pathIndex, "orbital\0"
                                               "2d\0"
                                               "3d TDSE\0"
                                               "GRMHD demo\0"
                                               "HARM GRMHD\0")) {
            renderPath_ = static_cast<RenderPath>(std::clamp(pathIndex, 0, 4));
            if (renderPath_ == RenderPath::Path2D) {
                pipeline = pipeline2d_;
                pipeline->setVertexBuffer(vbo2d_.get());
                pipeline->setIndexBuffer(ibo2d_.get());
            } else if (renderPath_ == RenderPath::Path3D) {
                pipeline = pipeline3d_;
                pipeline->setVertexBuffer(vbo3d_.get());
                pipeline->setIndexBuffer(ibo3d_.get());
                // Snap the 3D camera to fit the current domain on first switch
                camera3d_.resetForDomain(tdse3dDomainHalf_);
            } else if (renderPath_ == RenderPath::PathGRMHD) {
                pipeline = pipelineGrmhd_;
                pipeline->setVertexBuffer(vbo2d_.get());
                pipeline->setIndexBuffer(ibo2d_.get());
            } else if (renderPath_ == RenderPath::PathHarmGRMHD) {
                pipeline = pipelineHarmGrmhd_;
                pipeline->setVertexBuffer(vbo2d_.get());
                pipeline->setIndexBuffer(ibo2d_.get());
                harmViewMode_ = 5;
                twoDZoom_ = 0.070f;
                twoDPan_ = glm::vec2(0.0f);
            } else {
                pipeline = pipelineOrbital_;
                pipeline->setVertexBuffer(vbo_.get());
                pipeline->setIndexBuffer(ibo_.get());
            }
        }

        if (renderPath_ == RenderPath::Path3D) {
            ImGui::Separator();
            ImGui::Text("3D TDSE FDTD");

            int gridSize     = tdse3dGridSize_;
            int substeps     = tdse3dSubsteps_;
            int integrator   = tdse3dIntegrator_;
            int potType      = static_cast<int>(tdse3dPotType_);
            int colorMode    = tdse3dColorMode_;
            int sliceAxis    = tdse3dSliceAxis_ + 1; // 0->volume, 1->x, 2->y, 3->z
            int marchSteps   = static_cast<int>(tdse3dMarchSteps_);

            const int oldGrid  = tdse3dGridSize_;
            const Potential3dType oldPot = tdse3dPotType_;
            const float oldStrength  = tdse3dPotStrength_;
            const float oldRadius    = tdse3dPotRadius_;
            const float oldDomain    = tdse3dDomainHalf_;
            const glm::vec3 oldPos   = tdse3dPacketPos_;
            const glm::vec3 oldMom   = tdse3dPacketMom_;
            const float oldSigma     = tdse3dSigma_;
            const bool oldAbsorb     = tdse3dUseAbsorbing_;
            const float oldAbsW      = tdse3dAbsorbWidth_;
            const float oldAbsS      = tdse3dAbsorbStrength_;

            ImGui::Combo("integrator##3d", &integrator, "Euler\0Crank-Nicolson\0");
            ImGui::SliderInt("grid N##3d", &gridSize, 8, kMaxTdse3dGrid);
            ImGui::SliderInt("substeps##3d", &substeps, 1, 16);
            ImGui::SliderFloat("dt##3d", &tdse3dDt_, 1e-4f, 0.5f, "%.5f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("domain half##3d", &tdse3dDomainHalf_, 4.0f, 30.0f, "%.2f");
            ImGui::Separator();
            ImGui::Text("Initial wavepacket");
            ImGui::SliderFloat3("packet pos##3d",  glm::value_ptr(tdse3dPacketPos_), -10.0f, 10.0f, "%.2f");
            ImGui::SliderFloat3("packet mom##3d",  glm::value_ptr(tdse3dPacketMom_), -6.0f,  6.0f,  "%.2f");
            ImGui::SliderFloat("sigma##3d", &tdse3dSigma_, 0.3f, 6.0f, "%.2f");
            ImGui::Separator();
            ImGui::Text("Potential");
            ImGui::Combo("potential##3d", &potType, "Free\0Harmonic well\0Coulomb well\0Spherical barrier\0Double well\0");
            ImGui::SliderFloat("strength##3d", &tdse3dPotStrength_, 0.0f, 4.0f, "%.3f");
            ImGui::SliderFloat("radius##3d",   &tdse3dPotRadius_,   0.5f, 12.0f, "%.2f");
            ImGui::Checkbox("show potential##3d", &tdse3dShowPotential_);
            ImGui::Separator();
            ImGui::Text("Absorbing boundary");
            ImGui::Checkbox("absorbing##3d", &tdse3dUseAbsorbing_);
            if (tdse3dUseAbsorbing_) {
                ImGui::SliderFloat("absorb width##3d",    &tdse3dAbsorbWidth_,    0.5f, 10.0f, "%.2f");
                ImGui::SliderFloat("absorb strength##3d", &tdse3dAbsorbStrength_, 0.5f, 40.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            }
            ImGui::Separator();
            ImGui::Text("Visualization");
            ImGui::Combo("color##3d", &colorMode, "Inferno\0Magma\0Plasma\0Viridis\0\0\0Gray\0\0\0Phase\0");
            ImGui::Combo("view##3d", &sliceAxis, "Volume (ray-march)\0Slice X\0Slice Y\0Slice Z\0");
            if (sliceAxis > 0) {
                ImGui::SliderFloat("slice pos##3d", &tdse3dSlicePos_, -tdse3dDomainHalf_, tdse3dDomainHalf_, "%.2f");
            } else {
                ImGui::SliderInt("march steps##3d", &marchSteps, 16, 256);
                ImGui::SliderFloat("alpha scale##3d", &tdse3dAlphaScale_, 0.001f, 2.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
            }
            ImGui::SliderFloat("intensity scale##3d", &tdse3dIntensityScale_, 0.01f, 20.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("intensity range##3d", &tdse3dIntensityRange_, 0.01f, 4.0f,  "%.4f", ImGuiSliderFlags_Logarithmic);

            ImGui::Separator();
            ImGui::Text("ONNX rollout");
#ifdef ATOMS_ENABLE_ONNX
            ImGui::InputText("model path##onnx3d", tdse3dOnnxModelPath_, sizeof(tdse3dOnnxModelPath_));
            if (ImGui::Button("Load ONNX model##3d")) {
                loadOnnxModel3d(std::string(tdse3dOnnxModelPath_));
            }
            ImGui::SameLine();
            if (ImGui::Button("Run ONNX rollout##3d")) {
                tdse3dOnnxRunRequested_ = true;
                tdse3dUseOnnx_ = true;
            }
            ImGui::Checkbox("use ONNX mode##3d", &tdse3dUseOnnx_);
            ImGui::TextWrapped("%s", tdse3dOnnxStatus_.c_str());
#else
            ImGui::TextWrapped("ONNX support is disabled. Reconfigure with -DATOMS_ENABLE_ONNX=ON and set ONNXRUNTIME_ROOT.");
#endif

            ImGui::Separator();
            ImGui::Text("TorchScript rollout");
#ifdef ATOMS_ENABLE_TORCHSCRIPT
            ImGui::InputText("model path##torchscript3d", tdse3dTorchscriptModelPath_, sizeof(tdse3dTorchscriptModelPath_));
            ImGui::Text("TorchScript expected grid: %d", tdse3dTorchscriptExpectedGrid_);
            if (ImGui::Button("Load TorchScript model##3d")) {
                loadTorchscriptModel3d(std::string(tdse3dTorchscriptModelPath_));
            }
            ImGui::SameLine();
            if (ImGui::Button("Run TorchScript rollout##3d")) {
                tdse3dTorchscriptRunRequested_ = true;
                tdse3dUseTorchscript_ = true;
            }
            ImGui::Checkbox("use TorchScript mode##3d", &tdse3dUseTorchscript_);
            ImGui::Checkbox("autoplay TorchScript##3d", &tdse3dTorchscriptAutoplay_);
            ImGui::SliderInt("TorchScript steps/frame##3d", &tdse3dTorchscriptStepsPerFrame_, 1, 8);
            ImGui::TextWrapped("%s", tdse3dTorchscriptStatus_.c_str());
#else
            ImGui::TextWrapped("TorchScript support is disabled. Reconfigure with -DATOMS_ENABLE_TORCHSCRIPT=ON and set TORCH_ROOT.");
#endif

            if (ImGui::Button("Reset 3D wavefunction")) {
                tdse3dNeedsReset_ = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Fit camera")) {
                camera3d_.resetForDomain(tdse3dDomainHalf_);
            }
            ImGui::SameLine();
            ImGui::Text("t = %.3f", tdse3dTime_);

            if (ImGui::CollapsingHeader("3D Camera")) {
                const glm::vec3 cp3 = camera3d_.position();
                ImGui::Text("pos: (%.2f, %.2f, %.2f)", cp3.x, cp3.y, cp3.z);
                ImGui::Text("radius: %.2f", camera3d_.radius);
                ImGui::SliderFloat("orbit speed##3d", &camera3d_.orbitSpeed, 0.001f, 0.05f, "%.4f");
            }

            // Commit
            tdse3dIntegrator_ = integrator;
            tdse3dSubsteps_   = substeps;
            tdse3dColorMode_  = colorMode;
            tdse3dMarchSteps_ = static_cast<float>(std::clamp(marchSteps, 16, 256));
            tdse3dSliceAxis_  = sliceAxis - 1;
            const Potential3dType newPotType = static_cast<Potential3dType>(std::clamp(potType, 0, 4));

            if (gridSize != oldGrid) {
                tdse3dGridSize_ = gridSize;
                resize3dBuffers();
            }
            if (newPotType != oldPot ||
                tdse3dPotStrength_ != oldStrength ||
                tdse3dPotRadius_   != oldRadius   ||
                tdse3dDomainHalf_  != oldDomain) {
                tdse3dPotType_   = newPotType;
                tdse3dPotDirty_  = true;
                tdse3dNeedsReset_ = true;
            } else {
                tdse3dPotType_ = newPotType;
            }
            if (tdse3dPacketPos_ != oldPos || tdse3dPacketMom_ != oldMom || tdse3dSigma_ != oldSigma) {
                tdse3dNeedsReset_ = true;
            }
            if (tdse3dUseAbsorbing_ != oldAbsorb ||
                tdse3dAbsorbWidth_  != oldAbsW   ||
                tdse3dAbsorbStrength_ != oldAbsS) {
                tdse3dNeedsReset_ = true;
            }

            ImGui::End();
            return;
        }

        if (renderPath_ == RenderPath::PathGRMHD) {
            ImGui::Separator();
            ImGui::Text("GRMHD-inspired torus demo");
            int grid = grmhdGridSize_;
            int substeps = grmhdSubsteps_;
            int viewMode = grmhdViewMode_;
            const float oldDomain = grmhdDomainHalf_;
            const float oldHorizon = grmhdHorizonRadius_;
            const float oldSpin = grmhdSpin_;
            const float oldLoop = grmhdMagneticLoop_;
            ImGui::Combo("view##grmhd", &viewMode, "density\0magnetization\0plasma beta\0radial speed\0div B\0");
            ImGui::SliderInt("grid N##grmhd", &grid, 64, kMaxGrmhdGrid);
            ImGui::SliderInt("substeps/frame##grmhd", &substeps, 1, 16);
            ImGui::SliderFloat("dt##grmhd", &grmhdDt_, 0.0002f, 0.03f, "%.5f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("domain half##grmhd", &grmhdDomainHalf_, 8.0f, 36.0f, "%.1f");
            ImGui::SliderFloat("black-hole radius##grmhd", &grmhdHorizonRadius_, 0.8f, 4.0f, "%.2f");
            ImGui::SliderFloat("spin proxy##grmhd", &grmhdSpin_, -0.98f, 0.98f, "%.2f");
            ImGui::SliderFloat("magnetic loop##grmhd", &grmhdMagneticLoop_, 0.0f, 0.22f, "%.3f");
            ImGui::SliderFloat("pressure floor##grmhd", &grmhdPressureFloor_, 0.0001f, 0.04f, "%.5f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("diffusion##grmhd", &grmhdDiffusion_, 0.0f, 0.08f, "%.4f");
            ImGui::SliderFloat("color scale##grmhd", &grmhdColorScale_, 0.2f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::Checkbox("paused##grmhd", &grmhdPaused_);
            ImGui::SameLine();
            if (ImGui::Button("Reset torus##grmhd")) {
                grmhdNeedsReset_ = true;
            }
            ImGui::SameLine();
            ImGui::Text("t = %.2f", grmhdTime_);

            grmhdViewMode_ = std::clamp(viewMode, 0, 4);
            if (grid != grmhdGridSize_) {
                grmhdGridSize_ = std::clamp(grid, 64, kMaxGrmhdGrid);
                resizeGrmhdBuffers();
            }
            if (oldDomain != grmhdDomainHalf_ ||
                oldHorizon != grmhdHorizonRadius_ ||
                oldSpin != grmhdSpin_ ||
                oldLoop != grmhdMagneticLoop_) {
                grmhdConfigDirty_ = true;
            }
            grmhdSubsteps_ = std::clamp(substeps, 1, 16);
            ImGui::End();
            return;
        }

        if (renderPath_ == RenderPath::PathHarmGRMHD) {
            ImGui::Separator();
            ImGui::Text("HARM GRMHD");
            int n = harmGridSize_;
            int substeps = harmSubsteps_;
            int viewMode = harmViewMode_;
            int initMode = harmInitMode_;
            const float oldRout = harmRout_;
            const float oldA = harmSpin_;
            const float oldLoop = harmMagneticLoop_;
            const int oldInitMode = harmInitMode_;
            ImGui::Combo("view##harm", &viewMode, "density\0magnetization\0plasma beta\0radial 4-velocity\0primitive fail\0shadow image\0vertical slice\0azimuth slice\0evolved div B\0evolved flux\0evolved accretion\0");
            ImGui::Combo("initial data##harm", &initMode, "SANE torus\0MAD torus\0");
            ImGui::SliderInt("radial N##harm", &n, 32, kMaxHarmGrid);
            ImGui::SliderInt("substeps/frame##harm", &substeps, 1, 12);
            ImGui::SliderFloat("CFL dt##harm", &harmDt_, 0.0002f, 0.02f, "%.5f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("r out##harm", &harmRout_, 12.0f, 80.0f, "%.1f");
            ImGui::SliderFloat("spin a##harm", &harmSpin_, -0.98f, 0.98f, "%.2f");
            ImGui::SliderFloat("magnetic loop##harm", &harmMagneticLoop_, 0.0f, 0.18f, "%.3f");
            ImGui::SliderFloat("rho floor##harm", &harmRhoFloor_, 1e-6f, 1e-3f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("u floor##harm", &harmUFloor_, 1e-7f, 1e-3f, "%.7f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("color scale##harm", &harmColorScale_, 0.2f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("camera yaw##harm", &harmCameraYaw_, -3.14159f, 3.14159f, "%.2f");
            ImGui::SliderFloat("camera inclination##harm", &harmCameraInclination_, 0.05f, 1.45f, "%.2f");
            ImGui::Checkbox("GPU compute##harm", &harmUseGpu_);
            ImGui::Checkbox("paused##harm", &harmPaused_);
            ImGui::SameLine();
            if (ImGui::Button("Reset torus##harm")) {
                harmNeedsReset_ = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Raytraced view##harm")) {
                harmViewMode_ = 5;
                viewMode = 5;
            }
            ImGui::SameLine();
            ImGui::Text("t = %.2f", harmTime_);
            ImGui::Text("3D grid: %d x %d x %d", harmGridSize_, harmThetaSize(), harmPhiSize());
            ImGui::Text("beta min %.2f  <beta> %.1f  phi_BH %.2f  sigma max %.2f",
                harmDiagBetaMin_, harmDiagBetaMean_, harmDiagPhiBH_, harmDiagSigmaMax_);
            ImGui::Text("Left-drag orbit, right/middle-drag pan, wheel zoom");

            harmInitMode_ = std::clamp(initMode, 0, 1);
            harmProblem_ = harmInitMode_;
            harmViewMode_ = std::clamp(viewMode, 0, 10);
            if (n != harmGridSize_) {
                harmGridSize_ = std::clamp(n, 32, kMaxHarmGrid);
                resizeHarmBuffers();
            }
            if (oldRout != harmRout_ || oldA != harmSpin_ || oldLoop != harmMagneticLoop_ || oldInitMode != harmInitMode_) {
                harmNeedsReset_ = true;
            }
            harmSubsteps_ = std::clamp(substeps, 1, 12);
            ImGui::End();
            return;
        }

        if (renderPath_ == RenderPath::Path2D) {
            ImGui::Text("2D Hydrogen orbital visualizer");
            int n = quantum_.n;
            int l = quantum_.l;
            int m = quantum_.m;
            int colorMode = colorMode_;
            int twoDMode = twoDUseTdse_ ? 1 : 0;
            int potentialType = static_cast<int>(tdsePotentialType_);
            int gridSize = tdseGridSize_;
            const int oldN = quantum_.n;
            const int oldL = quantum_.l;
            const int oldM = quantum_.m;
            const int oldGrid = tdseGridSize_;
            const Potential2dType oldPotentialType = tdsePotentialType_;
            const float oldPotentialStrength = tdsePotentialStrength_;
            const float oldSquareHalfWidth = tdseSquareHalfWidth_;
            const float oldDomain = tdseDomainHalfExtent_;
            const float oldBigRadius = tdseCircleRadiusBig_;
            const float oldSmallRadius = tdseCircleRadiusSmall_;
            const float oldAnnIn = tdseAnnulusInner_;
            const float oldAnnOut = tdseAnnulusOuter_;
            const float oldBarrierHalfWidth = tdseBarrierHalfWidth_;
            const float oldSlitCenter = tdseSlitCenterOffset_;
            const float oldSlitHalfH = tdseSlitHalfHeight_;
            const glm::vec2 oldPacketCenter = tdsePacketCenter_;
            const glm::vec2 oldPacketMomentum = tdsePacketMomentum_;
            const bool oldUseAbsorbing = tdseUseAbsorbingBoundary_;
            const float oldAbsorbWidth = tdseAbsorbWidth_;
            const float oldAbsorbStrength = tdseAbsorbStrength_;
            constexpr int kMaxN = 30;
            ImGui::Combo("2d mode", &twoDMode, "Analytic orbital\0TDSE FDTD\0");
            ImGui::SliderInt("n", &n, 1, kMaxN);
            ImGui::SliderInt("l", &l, 0, kMaxN - 1);
            if (l >= n) {
                n = std::min(kMaxN, l + 1);
                l = std::min(l, n - 1);
            }
            m = std::clamp(m, -l, l);
            ImGui::SliderInt("m", &m, -std::max(1, l), std::max(1, l));
            ImGui::Text("Constraint: l < n (n auto-increases when needed)");
            ImGui::Combo("colorspace", &colorMode, "Inferno\0Magma\0Plasma\0Viridis\0Cividis\0Turbo\0Gray\0Fire\0Cyan-Magenta\0Phase Velocity\0Stationary Phase\0");
            ImGui::SliderFloat("2d zoom", &twoDZoom_, 1e-6f, 1e6f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("2d thickness", &twoDThickness_, 0.1f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("2d slice z", &twoDSliceZ_, -8.0f, 8.0f, "%.2f");
            ImGui::Checkbox("2d integrate depth", &twoDIntegrateDepth_);
            ImGui::SliderFloat("2d intensity scale", &intensityScale_, 0.1f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("2d intensity range", &intensityRange_, 0.05f, 50.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("2d phase speed", &twoDPhaseSpeed_, 0.0f, 8.0f, "%.2f");

            twoDUseTdse_ = (twoDMode == 1);
            if (twoDUseTdse_) {
                ImGui::Separator();
                ImGui::Text("Time-dependent Schrodinger (explicit FDTD)");
                ImGui::Text("Use WASD to move the packet center");
                ImGui::Combo("potential", &potentialType, "Square well\0Circle well (big)\0Circle well (small)\0Double slit well\0Big circle small circle (annulus)\0");
                ImGui::SliderFloat("potential strength", &tdsePotentialStrength_, 0.0f, 16.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("square half-width", &tdseSquareHalfWidth_, 0.2f, 10.0f, "%.2f");
                ImGui::SliderFloat("big circle radius", &tdseCircleRadiusBig_, 0.3f, 20.0f, "%.2f");
                ImGui::SliderFloat("small circle radius", &tdseCircleRadiusSmall_, 0.2f, 10.0f, "%.2f");
                ImGui::SliderFloat("annulus inner radius", &tdseAnnulusInner_, 0.2f, 12.0f, "%.2f");
                ImGui::SliderFloat("annulus outer radius", &tdseAnnulusOuter_, 0.3f, 20.0f, "%.2f");
                ImGui::SliderFloat("double-slit barrier half-width", &tdseBarrierHalfWidth_, 0.05f, 1.2f, "%.3f");
                ImGui::SliderFloat("double-slit offset", &tdseSlitCenterOffset_, 0.0f, 6.0f, "%.2f");
                ImGui::SliderFloat("double-slit half-height", &tdseSlitHalfHeight_, 0.05f, 2.5f, "%.2f");
                ImGui::SliderFloat("domain half-size", &tdseDomainHalfExtent_, 4.0f, 40.0f, "%.2f");
                ImGui::SliderInt("fdtd grid", &gridSize, 64, 320);
                ImGui::Combo("integrator", &tdseIntegrator_, "Euler\0Crank-Nicolson\0");
                ImGui::SliderInt("substeps/frame", &tdseSubstepsPerFrame_, 1, 32);
                ImGui::SliderFloat("dt", &tdseDt_, 1e-6f, 1.0f, "%.6f", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat2("packet center", glm::value_ptr(tdsePacketCenter_), -40.0f, 40.0f, "%.2f");
                ImGui::SliderFloat2("packet momentum", glm::value_ptr(tdsePacketMomentum_), -8.0f, 8.0f, "%.2f");
                ImGui::SliderFloat("WASD move speed", &tdsePacketMoveSpeed_, 0.5f, 25.0f, "%.2f");
                ImGui::Checkbox("absorbing boundary", &tdseUseAbsorbingBoundary_);
                if (tdseUseAbsorbingBoundary_) {
                    ImGui::SliderFloat("absorb width", &tdseAbsorbWidth_, 0.4f, 16.0f, "%.2f");
                    ImGui::SliderFloat("absorb strength", &tdseAbsorbStrength_, 0.5f, 40.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                }
                ImGui::Checkbox("overlay potential", &tdseOverlayPotential_);
                if (ImGui::Button("Reset TDSE wavefunction")) {
                    tdseWaveNeedsReset_ = true;
                }
                tdsePotentialType_ = static_cast<Potential2dType>(std::clamp(potentialType, 0, 4));
                tdseGridSize_ = std::clamp(gridSize, 64, 320);
                if (tdseGridSize_ != oldGrid) {
                    resizeTdseBuffers();
                }
                if (tdsePotentialType_ != oldPotentialType ||
                    tdsePotentialStrength_ != oldPotentialStrength ||
                    tdseSquareHalfWidth_ != oldSquareHalfWidth ||
                    tdseDomainHalfExtent_ != oldDomain ||
                    tdseCircleRadiusBig_ != oldBigRadius ||
                    tdseCircleRadiusSmall_ != oldSmallRadius ||
                    tdseAnnulusInner_ != oldAnnIn ||
                    tdseAnnulusOuter_ != oldAnnOut ||
                    tdseBarrierHalfWidth_ != oldBarrierHalfWidth ||
                    tdseSlitCenterOffset_ != oldSlitCenter ||
                    tdseSlitHalfHeight_ != oldSlitHalfH) {
                    tdsePotentialDirty_ = true;
                    tdseWaveNeedsReset_ = true;
                }
                if (tdsePacketCenter_ != oldPacketCenter || tdsePacketMomentum_ != oldPacketMomentum) {
                    tdseWaveNeedsReset_ = true;
                }
                if (tdseUseAbsorbingBoundary_ != oldUseAbsorbing ||
                    tdseAbsorbWidth_ != oldAbsorbWidth ||
                    tdseAbsorbStrength_ != oldAbsorbStrength) {
                    tdseWaveNeedsReset_ = true;
                }
            }

            quantum_.n = n;
            quantum_.l = l;
            quantum_.m = m;
            quantum_.clamp();
            colorMode_ = std::clamp(colorMode, 0, 10);
            if (oldN != quantum_.n || oldL != quantum_.l || oldM != quantum_.m) {
                tdseWaveNeedsReset_ = true;
            }
            ImGui::End();
            return;
        }

        ImGui::Text("GPU orbital evaluation (instant n/l/m updates)");

        bool applyConfigFromEnter = ImGui::InputText(
            "electron config",
            configInput_,
            sizeof(configInput_),
            ImGuiInputTextFlags_EnterReturnsTrue
        );
        ImGui::SameLine();
        if (ImGui::Button("Apply config") || applyConfigFromEnter) {
            applyElectronConfiguration(configInput_);
        }
        ImGui::SameLine();
        if (ImGui::Button("Single orbital mode")) {
            vmcMode_ = false;
            useConfiguration_ = false;
            configMessage_ = "Single-orbital mode active";
            configError_.clear();
            orbitalBufferDirty_ = true;
        }

        ImGui::TextWrapped("Examples: 1s^1 (Hydrogen), 1s^2 2s^2 2p^2 (Carbon)");
        if (!configMessage_.empty()) {
            ImGui::TextWrapped("%s", configMessage_.c_str());
        }
        if (!configError_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", configError_.c_str());
        }
        ImGui::Separator();

        ImGui::Checkbox("Many-body Slater VMC mode", &vmcMode_);
        if (vmcMode_ && !vmcConfigured_) {
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.25f, 1.0f), "VMC is enabled but not configured. Click Apply config first.");
        }
        ImGui::SliderInt("VMC sweeps/frame", &vmcSweepsPerFrame_, 1, 48);
        ImGui::SliderInt("VMC thermalization", &vmcThermalizationSweeps_, 0, 16);
        ImGui::SliderInt("VMC measure every", &vmcMeasureEvery_, 1, 8);
        ImGui::SliderInt("VMC walkers", &vmcWalkerCount_, 1, 512);
        ImGui::SliderFloat("VMC step size", &vmcStepSize_, 0.03f, 1.5f, "%.3f");
        ImGui::SliderFloat("VMC zeta scale", &vmcZetaScale_, 0.2f, 3.0f, "%.3f");
        ImGui::SliderFloat("Jastrow beta", &vmcJastrowBeta_, 0.0f, 1.5f, "%.3f");
        ImGui::Checkbox("VMC camera auto-center", &vmcAutoCenterCamera_);
        ImGui::SameLine();
        if (ImGui::Button("Center camera now")) {
            centerCameraOnVmcCloud(true);
        }
        if (ImGui::Button("Reset walkers")) {
            vmc_.resetWalkers(static_cast<uint32_t>(sampleSeed_ * 7919.0f) ^ 0xa341316cu);
            vmc_.clearPointCloud();
            sampleSeed_ += 1.0f;
            centerCameraOnVmcCloud(true);
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear VMC cloud")) {
            vmc_.clearPointCloud();
            vmcDrawCount_ = 0;
        }
        const SlaterVMC::Stats vmcStats = vmc_.stats();
        ImGui::Text("VMC active walkers: %d", vmc_.walkerCount());
        ImGui::Text("VMC acceptance: %.2f%%", vmcStats.acceptance * 100.0);
        ImGui::Text("VMC <E_L>: %.6f Ha (%d measurements)", vmcStats.avgEnergy, vmcStats.measurements);
        ImGui::Separator();

        int n = quantum_.n;
        int l = quantum_.l;
        int m = quantum_.m;
        int count = quantum_.sampleCount;
        int colorMode = colorMode_;
        int octant = clip_.removedOctant;
        float origin[3] = { clip_.origin.x, clip_.origin.y, clip_.origin.z };
        int oldN = quantum_.n;
        int oldL = quantum_.l;
        int oldM = quantum_.m;
        float flowSpeed = flowSpeed_;
        float intensityRange = intensityRange_;

        constexpr int kMaxN = 30;
        ImGui::SliderInt("n", &n, 1, kMaxN);
        ImGui::SliderInt("l", &l, 0, kMaxN - 1);
        if (l >= n) {
            n = std::min(kMaxN, l + 1);
            l = std::min(l, n - 1);
        }
        m = std::clamp(m, -l, l);
        ImGui::SliderInt("m", &m, -std::max(1, l), std::max(1, l));
        ImGui::SliderInt("samples", &count, 1000, kMaxParticles);
        ImGui::Text("Constraint: l < n (n auto-increases when needed)");

        ImGui::Combo("colorspace", &colorMode, "Inferno\0Magma\0Plasma\0Viridis\0Cividis\0Turbo\0Gray\0Fire\0Cyan-Magenta\0Phase Velocity\0Stationary Phase\0");
        if (colorMode == 9) {
            ImGui::TextWrapped("Phase Velocity: Colors cycle through the spectrum; faster local color cycling indicates higher local momentum/energy.");
        }
        if (colorMode == 10) {
            ImGui::TextWrapped("Stationary Phase: The orbital phase rotates coherently over time (global e^{-iEt/hbar} style evolution).");
        }

        ImGui::Separator();
        ImGui::Text("Selected octant is removed");
        ImGui::SliderFloat3("origin", origin, -200.0f, 200.0f, "%.2f");
        ImGui::SliderInt("removed octant", &octant, 1, 8);

        ImGui::SliderFloat("intensity scale", &intensityScale_, 0.1f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("intensity range", &intensityRange, 0.05f, 50.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("flow speed", &flowSpeed, 0.0f, 8.0f, "%.2f");

        quantum_.n = n;
        quantum_.l = l;
        quantum_.m = m;
        quantum_.sampleCount = std::clamp(count, 1000, kMaxParticles);
        quantum_.clamp();
        colorMode_ = std::clamp(colorMode, 0, 10);
        flowSpeed_ = std::max(0.0f, flowSpeed);
        intensityRange_ = std::clamp(intensityRange, 0.01f, 50.0f);
        vmc_.setMaxCloudPoints(static_cast<size_t>(quantum_.sampleCount));

        if (!useConfiguration_) {
            orbitalBufferDirty_ = true;
        }

        if (oldN != quantum_.n || oldL != quantum_.l || oldM != quantum_.m) {
            sampleSeed_ += 1.0f;
            orbitalBufferDirty_ = true;
        }

        clip_.origin = glm::vec3(origin[0], origin[1], origin[2]);
        clip_.removedOctant = std::clamp(octant, 1, 8);

        if (ImGui::Button("Reshuffle sample set")) {
            sampleSeed_ += 1.0f;
            orbitalBufferDirty_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset motion phase")) {
            simulationTime_ = 0.0f;
        }

        if (ImGui::CollapsingHeader("Debug Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            const glm::vec3 camPos = camera_.position();
            const float azimuthDeg = camera_.azimuth * 57.29577951308232f;
            const float elevationDeg = camera_.elevation * 57.29577951308232f;
            ImGui::Text("position: (%.3f, %.3f, %.3f)", camPos.x, camPos.y, camPos.z);
            ImGui::Text("target:   (%.3f, %.3f, %.3f)", camera_.target.x, camera_.target.y, camera_.target.z);
            ImGui::Text("rotation: azimuth %.2f deg, elevation %.2f deg", azimuthDeg, elevationDeg);
            ImGui::Text("radius: %.3f", camera_.radius);
        }

        ImGui::Text("Default removes octant 1 at origin (0,0,0)");
        ImGui::End();
    }

private:
    static constexpr float kPi = 3.14159265358979323846f;
    static constexpr int kMaxTdseGrid = 320;
    static constexpr int kMaxGrmhdGrid = 256;
    static constexpr int kMaxHarmGrid = 96;
    static constexpr int kHarmGhost = 2;

    static void writeRenderUniform(wgfx::Pipeline* pipeline, const float* data) {
        if (!pipeline || pipeline->uniforms.uniforms.empty()) return;
        wgfx::Uniform* uniform = pipeline->uniforms.uniforms.at(0);
        wgfx::queue.writeBuffer(uniform->buffer, 0, data, uniform->minBindingSize);
        if (pipeline->uniforms.dynamicOffsets.empty()) {
            pipeline->uniforms.dynamicOffsets.resize(1, 0);
        }
        pipeline->uniforms.dynamicOffsets[0] = 0;
    }

    enum class Potential2dType : int {
        SquareWell = 0,
        CircleWellBig = 1,
        CircleWellSmall = 2,
        DoubleSlitWell = 3,
        AnnularWell = 4
    };

    enum class RenderPath : int {
        Orbital = 0,
        Path2D = 1,
        Path3D = 2,
        PathGRMHD = 3,
        PathHarmGRMHD = 4
    };

    struct alignas(16) GpuOrbitalState {
        glm::mat4 viewProj;
        glm::vec4 clipOrigin;
        glm::vec4 quantum;
        glm::vec4 render;
    };
    struct alignas(16) Gpu2dState {
        glm::vec4 orbital; // x:n, y:l, z:m, w:colorMode
        glm::vec4 tuning;  // x:intensityScale, y:intensityRange, z:zoom, w:thickness
        glm::vec4 render;  // x:time, y:aspect, z:phaseSpeed, w:mode(0:analytic,1:tdse)
        glm::vec4 pan;     // x:panX, y:panY, z:sliceZ, w:integrateDepth(0/1)
        glm::vec4 tdse;    // x:gridSize, y:domainHalfExtent, z:potentialOverlay, w:reserved
    };
    struct alignas(16) Gpu3dState {
        glm::mat4 invViewProj;
        glm::vec4 camPos;   // xyz=cam, w=time
        glm::vec4 params;   // x=gridSize, y=domainHalf, z=intensityScale, w=intensityRange
        glm::vec4 render;   // x=colorMode, y=aspect, z=showPotential, w=sliceAxis
        glm::vec4 march;    // x=stepCount, y=alphaScale, z=slicePos, w=reserved
    };

    static constexpr int kMaxParticles = 250000;

    std::unique_ptr<wgfx::VertexBuffer> vbo_;
    std::unique_ptr<wgfx::IndexBuffer> ibo_;
    wgfx::Uniform* stateUniform_ = nullptr;
    std::unique_ptr<wgfx::VertexBuffer> vbo2d_;
    std::unique_ptr<wgfx::IndexBuffer> ibo2d_;
    wgfx::Uniform* stateUniform2d_ = nullptr;
    wgfx::Uniform* tdseStorage_ = nullptr;
    wgfx::Pipeline* pipelineOrbital_ = nullptr;
    wgfx::Pipeline* pipeline2d_ = nullptr;
    wgfx::Pipeline* pipelineTdse2d_ = nullptr;
    wgfx::Pipeline* pipelineGrmhd_ = nullptr;
    wgfx::Pipeline* pipelineHarmGrmhd_ = nullptr;
    wgfx::Uniform* grmhdStorage_ = nullptr;
    wgfx::Uniform* harmStorage_ = nullptr;
    wgfx::Uniform* harmGpuA_ = nullptr;
    wgfx::Uniform* harmGpuB_ = nullptr;
    wgfx::Uniform* harmComputeParamsUni_ = nullptr;
    wgfx::Compute* harmComputeStep_ = nullptr;
    wgfx::Compute* harmComputeCopy_ = nullptr;

    std::unique_ptr<wgfx::VertexBuffer> vbo3d_;
    std::unique_ptr<wgfx::IndexBuffer> ibo3d_;
    wgfx::Uniform* stateUniform3d_ = nullptr;
    wgfx::Uniform* tdseStorage3d_ = nullptr;
    wgfx::Pipeline* pipeline3d_ = nullptr;

    RenderPath renderPath_ = RenderPath::Path2D;

    OrbitCamera camera_;   // Orbital / 2D camera
    OrbitCamera camera3d_; // 3D TDSE camera — tight defaults set in init3dCamera()
    QuantumState quantum_;
    ClipState clip_;

    int colorMode_ = 3;
    float intensityScale_ = 18.44f;
    float intensityRange_ = 0.337f;
    float sampleSeed_ = 1.0f;
    float simulationTime_ = 0.0f;
    float flowSpeed_ = 8.0f;

    bool useConfiguration_ = false;
    bool orbitalBufferDirty_ = true;
    std::vector<SpinOrbitalState> electronOrbitals_;
    std::vector<float> orbitalAttribData_;
    std::vector<float> vmcUploadData_;
    char configInput_[256] = "1s^1";
    std::string configMessage_;
    std::string configError_;

    SlaterVMC vmc_;
    bool vmcMode_ = true;
    bool vmcConfigured_ = false;
    int vmcSweepsPerFrame_ = 12;
    int vmcThermalizationSweeps_ = 2;
    int vmcMeasureEvery_ = 2;
    int vmcWalkerCount_ = 64;
    float vmcStepSize_ = 0.35f;
    float vmcZetaScale_ = 1.0f;
    float vmcJastrowBeta_ = 0.2f;
    int vmcDrawCount_ = 0;
    bool vmcAutoCenterCamera_ = true;

    GpuOrbitalState gpuState_{};
    Gpu2dState gpu2dState_{};
    float twoDTime_ = 0.0f;
    float twoDZoom_ = 0.043190f;
    float twoDThickness_ = 3.2f;
    float twoDPhaseSpeed_ = 1.0f;
    float twoDSliceZ_ = 0.0f;
    bool twoDIntegrateDepth_ = false;
    glm::vec2 twoDPan_ = glm::vec2(0.0f);
    bool twoDDragging_ = false;
    glm::vec2 twoDLastMouse_ = glm::vec2(0.0f);

    bool twoDUseTdse_ = true;
    int tdseGridSize_ = 192;
    float tdseDomainHalfExtent_ = 22.0f;
    float tdseDt_ = 0.1f;
    int tdseSubstepsPerFrame_ = 8;
    int tdseIntegrator_ = 1;
    Potential2dType tdsePotentialType_ = Potential2dType::SquareWell;
    float tdsePotentialStrength_ = 4.0f;
    float tdseSquareHalfWidth_ = 3.2f;
    float tdseCircleRadiusBig_ = 6.0f;
    float tdseCircleRadiusSmall_ = 2.6f;
    float tdseAnnulusInner_ = 2.4f;
    float tdseAnnulusOuter_ = 6.8f;
    float tdseBarrierHalfWidth_ = 0.32f;
    float tdseSlitCenterOffset_ = 1.7f;
    float tdseSlitHalfHeight_ = 0.55f;
    glm::vec2 tdsePacketCenter_ = glm::vec2(-4.8f, 0.0f);
    glm::vec2 tdsePacketMomentum_ = glm::vec2(3.2f, 0.0f);
    float tdsePacketMoveSpeed_ = 8.0f;
    bool tdseUseAbsorbingBoundary_ = true;
    float tdseAbsorbWidth_ = 5.0f;
    float tdseAbsorbStrength_ = 14.0f;
    bool tdseOverlayPotential_ = false;
    bool tdsePotentialDirty_ = true;
    bool tdseWaveNeedsReset_ = true;
    int tdseNormCounter_ = 0;

    std::vector<float> tdseReal_;
    std::vector<float> tdseImag_;
    std::vector<float> tdseNextReal_;
    std::vector<float> tdseNextImag_;
    std::vector<float> tdseRhsReal_;
    std::vector<float> tdseRhsImag_;
    std::vector<float> tdsePotential_;
    std::vector<float> tdseUpload_;

    int grmhdGridSize_ = 160;
    float grmhdDomainHalf_ = 22.0f;
    float grmhdDt_ = 0.006f;
    int grmhdSubsteps_ = 4;
    int grmhdViewMode_ = 0;
    float grmhdHorizonRadius_ = 1.7f;
    float grmhdSpin_ = 0.55f;
    float grmhdMagneticLoop_ = 0.075f;
    float grmhdPressureFloor_ = 0.001f;
    float grmhdDiffusion_ = 0.018f;
    float grmhdColorScale_ = 1.4f;
    float grmhdTime_ = 0.0f;
    bool grmhdPaused_ = false;
    bool grmhdNeedsReset_ = true;
    bool grmhdConfigDirty_ = true;

    struct GrmhdCell {
        float rho = 0.0f;
        float sx = 0.0f;
        float sy = 0.0f;
        float tau = 0.0f;
        float bx = 0.0f;
        float by = 0.0f;
        float divb = 0.0f;
        float aux = 0.0f;
    };

    std::vector<GrmhdCell> grmhd_;
    std::vector<GrmhdCell> grmhdNext_;
    std::vector<float> grmhdUpload_;

    int harmGridSize_ = 64;
    float harmRin_ = 1.85f;
    float harmRout_ = 42.0f;
    float harmDt_ = 0.0025f;
    int harmSubsteps_ = 3;
    int harmViewMode_ = 5;
    int harmProblem_ = 0;
    int harmInitMode_ = 0;
    float harmSpin_ = 0.7f;
    float harmMagneticLoop_ = 0.055f;
    float harmRhoFloor_ = 1e-5f;
    float harmUFloor_ = 1e-6f;
    float harmColorScale_ = 1.2f;
    float harmCameraYaw_ = 0.0f;
    float harmCameraInclination_ = 1.18f;
    float harmTime_ = 0.0f;
    bool harmPaused_ = false;
    bool harmUseGpu_ = true;
    bool harmNeedsReset_ = true;
    bool harmGpuNeedsUpload_ = true;
    float harmDiagBetaMin_ = 0.0f;
    float harmDiagBetaMean_ = 0.0f;
    float harmDiagPhiBH_ = 0.0f;
    float harmDiagSigmaMax_ = 0.0f;

    struct alignas(16) HarmComputeParams {
        uint32_t gridN = 0;
        uint32_t thetaN = 0;
        uint32_t phiN = 0;
        uint32_t substeps = 0;
        float dt = 0.0f;
        float rin = 0.0f;
        float rout = 0.0f;
        float spin = 0.0f;
        float rhoFloor = 0.0f;
        float uFloor = 0.0f;
        float magneticLoop = 0.0f;
        float time = 0.0f;
        uint32_t problem = 0;
        float pad1 = 0.0f;
        float pad2 = 0.0f;
        float pad3 = 0.0f;
    };
    HarmComputeParams harmComputeParams_{};

    struct HarmPrim {
        float rho = 0.0f;
        float u = 0.0f;
        float ur = 0.0f;
        float utheta = 0.0f;
        float uphi = 0.0f;
        float br = 0.0f;
        float btheta = 0.0f;
        float bphi = 0.0f;
        float fail = 0.0f;
        float pad0 = 0.0f;
        float pad1 = 0.0f;
        float pad2 = 0.0f;
    };

    struct HarmCons {
        float d = 0.0f;
        float sr = 0.0f;
        float sphi = 0.0f;
        float tau = 0.0f;
        float br = 0.0f;
        float bphi = 0.0f;
        float fail = 0.0f;
        float divb = 0.0f;
    };

    struct HarmGeom {
        float r = 1.0f;
        float phi = 0.0f;
        float alpha = 1.0f;
        float betaPhi = 0.0f;
        float sqrtg = 1.0f;
        float gcov[4][4]{};
        float gcon[4][4]{};
    };

    struct HarmState {
        float ucon[4]{};
        float ucov[4]{};
        float bcon[4]{};
        float bcov[4]{};
        float bsq = 0.0f;
    };

    std::vector<HarmPrim> harmP_;
    std::vector<HarmPrim> harmPNext_;
    std::vector<HarmCons> harmU_;
    std::vector<HarmCons> harmUNext_;
    std::vector<float> harmUpload_;

    bool prevW_ = false;
    bool prevS_ = false;
    bool prevE_ = false;
    bool prevD_ = false;
    bool prevR_ = false;
    bool prevF_ = false;
    bool prevT_ = false;
    bool prevG_ = false;

    // ---- 3D TDSE state ----
    static constexpr int kMaxTdse3dGrid = 128;
    Gpu3dState gpu3dState_{};
    int tdse3dGridSize_         = 80;
    float tdse3dDomainHalf_     = 12.0f;
    float tdse3dDt_             = 0.06f;
    int tdse3dSubsteps_         = 4;
    int tdse3dIntegrator_       = 1;  // 0=Euler, 1=CN
    float tdse3dIntensityScale_ = 1.0f;
    float tdse3dIntensityRange_ = 0.3f;
    int tdse3dColorMode_        = 9;  // phase by default
    int tdse3dSliceAxis_        = -1; // -1=volume, 0/1/2=x/y/z
    float tdse3dSlicePos_       = 0.0f;
    float tdse3dMarchSteps_     = 96.0f;
    float tdse3dAlphaScale_     = 0.08f;
    bool tdse3dShowPotential_   = false;
    bool tdse3dUseAbsorbing_    = true;
    float tdse3dAbsorbWidth_    = 3.0f;
    float tdse3dAbsorbStrength_ = 12.0f;
    float tdse3dTime_           = 0.0f;

    bool tdse3dUseOnnx_          = false;
    bool tdse3dOnnxLoaded_       = false;
    bool tdse3dOnnxRunRequested_ = false;
    char tdse3dOnnxModelPath_[512] = "./tdse_fno_runs_safe/tdse_fno3d.onnx";
    std::string tdse3dOnnxStatus_ = "ONNX model not loaded";
    std::vector<float> tdse3dOnnxInput_;

#ifdef ATOMS_ENABLE_ONNX
    std::unique_ptr<Ort::Env> onnxEnv_;
    std::unique_ptr<Ort::SessionOptions> onnxSessionOptions_;
    std::unique_ptr<Ort::Session> onnxSession_;
    std::vector<std::string> onnxInputNamesStorage_;
    std::vector<std::string> onnxOutputNamesStorage_;
    std::vector<const char*> onnxInputNames_;
    std::vector<const char*> onnxOutputNames_;
#endif

    bool tdse3dUseTorchscript_            = false;
    bool tdse3dTorchscriptLoaded_         = false;
    bool tdse3dTorchscriptRunRequested_   = false;
    bool tdse3dTorchscriptAutoplay_       = true;
    int tdse3dTorchscriptStepsPerFrame_   = 1;
    int tdse3dTorchscriptExpectedGrid_    = 128;
    char tdse3dTorchscriptModelPath_[512] = "/Users/ethan/code/atoms/tdse_fno_runs_safe/tdse_fno3d_torchscript.pt";
    std::string tdse3dTorchscriptStatus_  = "TorchScript model not loaded";

#ifdef ATOMS_ENABLE_TORCHSCRIPT
    std::unique_ptr<torch::jit::script::Module> torchscriptModule_;
#endif

    enum class Potential3dType : int {
        Free = 0,
        HarmonicWell = 1,
        CoulombWell = 2,
        SphericalBarrier = 3,
        DoubleWell = 4
    };
    Potential3dType tdse3dPotType_ = Potential3dType::Free;
    float tdse3dPotStrength_  = 0.5f;
    float tdse3dPotRadius_    = 4.0f;
    glm::vec3 tdse3dPacketPos_  = glm::vec3(-4.0f, 0.0f, 0.0f);
    glm::vec3 tdse3dPacketMom_  = glm::vec3(2.0f,  0.0f, 0.0f);
    float tdse3dSigma_          = 2.0f;

    bool tdse3dNeedsReset_    = true;
    bool tdse3dPotDirty_      = true;
    int  tdse3dNormCounter_   = 0;

    // CPU-side scratch for initial upload only
    std::vector<float> tdse3dReal_;
    std::vector<float> tdse3dImag_;
    std::vector<float> tdse3dPot_;

    // GPU compute pipeline objects
    struct alignas(16) GpuComputeParams {
        uint32_t gridN;
        uint32_t mode;         // 0=Euler, 1=CN
        float    dt;
        float    absorbWidth;
        float    absorbStr;
        float    dx;
        float    _pad0;
        float    _pad1;
    };
    GpuComputeParams gpuComputeParams_{};

    wgfx::Compute*  computeEuler_    = nullptr; // entry="euler"
    wgfx::Compute*  computeCnRhs_    = nullptr; // entry="cn_rhs"
    wgfx::Compute*  computeCnIter_   = nullptr; // entry="cn_iter"
    wgfx::Compute*  computeCnFinish_ = nullptr; // entry="cn_finish"

    // Shared storage buffers (allocated to kMaxTdse3dGrid³)
    wgfx::Uniform*  computeParamsUni_ = nullptr; // binding 0 — uniform
    wgfx::Uniform*  gpuWaveA_         = nullptr; // binding 1 — vec2f (re,im)
    wgfx::Uniform*  gpuWaveB_         = nullptr; // binding 2 — vec2f (re,im)  (also render binding 1)
    wgfx::Uniform*  gpuPot_           = nullptr; // binding 3 — f32   potential (also render binding 2)

    Quad() {
        pipelineOrbital_ = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "atoms_sphere.wgsl").c_str()));
        pipeline = pipelineOrbital_;

        stateUniform_ = wgfx::createUniform(0, sizeof(GpuOrbitalState), reinterpret_cast<const float*>(&gpuState_));
        pipelineOrbital_->uniforms.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        pipelineOrbital_->uniforms.setUniform(stateUniform_);

        pipelineOrbital_->targets = 1;
        pipelineOrbital_->useDepth = false;

        initBuffers();
        pipelineOrbital_->init(vbo_.get());

        pipeline2d_ = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "circle_2d.wgsl").c_str()));
        stateUniform2d_ = wgfx::createUniform(0, sizeof(Gpu2dState), reinterpret_cast<const float*>(&gpu2dState_));
        pipeline2d_->uniforms.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        pipeline2d_->uniforms.setUniform(stateUniform2d_);
        resizeTdseBuffers();
        tdseStorage_ = wgfx::createStorage(
            1,
            static_cast<size_t>(kMaxTdseGrid) * static_cast<size_t>(kMaxTdseGrid) * 4 * sizeof(float),
            nullptr,
            true);
        pipeline2d_->uniforms.setStorage(tdseStorage_);
        pipeline2d_->targets = 1;
        pipeline2d_->useDepth = false;
        init2dBuffers();
        pipeline2d_->init(vbo2d_.get());

        // Load TDSE 2D pipeline for time-dependent simulations
        pipelineTdse2d_ = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "tdse2d.wgsl").c_str()));
        pipelineTdse2d_->uniforms.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        pipelineTdse2d_->uniforms.setUniform(stateUniform2d_);  // Share same uniform as pipeline2d_
        pipelineTdse2d_->uniforms.setStorage(tdseStorage_);
        pipelineTdse2d_->targets = 1;
        pipelineTdse2d_->useDepth = false;
        pipelineTdse2d_->init(vbo2d_.get());

        pipelineGrmhd_ = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "grmhd_demo.wgsl").c_str()));
        pipelineGrmhd_->uniforms.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        pipelineGrmhd_->uniforms.setUniform(stateUniform2d_);
        grmhdStorage_ = wgfx::createStorage(
            1,
            static_cast<size_t>(kMaxGrmhdGrid) * static_cast<size_t>(kMaxGrmhdGrid) * 8 * sizeof(float),
            nullptr,
            true);
        pipelineGrmhd_->uniforms.setStorage(grmhdStorage_);
        pipelineGrmhd_->targets = 1;
        pipelineGrmhd_->useDepth = false;
        pipelineGrmhd_->init(vbo2d_.get());
        resizeGrmhdBuffers();

        pipelineHarmGrmhd_ = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "harm_grmhd.wgsl").c_str()));
        pipelineHarmGrmhd_->uniforms.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        pipelineHarmGrmhd_->uniforms.setUniform(stateUniform2d_);
        const size_t harmGpuBytes = static_cast<size_t>(kMaxHarmGrid)
            * static_cast<size_t>(kMaxHarmGrid / 2)
            * static_cast<size_t>(kMaxHarmGrid)
            * 12 * sizeof(float);
        harmGpuA_ = wgfx::createStorage(
            1,
            harmGpuBytes,
            nullptr,
            false);
        harmGpuB_ = wgfx::createStorage(
            2,
            harmGpuBytes,
            nullptr,
            false);
        harmStorage_ = new wgfx::Uniform();
        harmStorage_->isReadOnly = true;
        harmStorage_->binding = 1;
        harmStorage_->minBindingSize = harmGpuB_->minBindingSize;
        harmStorage_->buffer = harmGpuB_->buffer;
        harmStorage_->entry.binding = 1;
        harmStorage_->entry.buffer = harmGpuB_->buffer;
        harmStorage_->entry.offset = 0;
        harmStorage_->entry.size = static_cast<uint64_t>(harmGpuB_->minBindingSize);
        pipelineHarmGrmhd_->uniforms.setStorage(harmStorage_);
        pipelineHarmGrmhd_->targets = 1;
        pipelineHarmGrmhd_->useDepth = false;
        pipelineHarmGrmhd_->init(vbo2d_.get());

        const std::string harmComputeSrc = wgfx::loadFromFile(
            (std::string(RESOURCE_DIR) + "/" + "harm_grmhd_compute.wgsl").c_str());
        harmComputeParamsUni_ = wgfx::createUniform(0, sizeof(HarmComputeParams),
            reinterpret_cast<const float*>(&harmComputeParams_));
        auto makeHarmCompute = [&](wgfx::Compute*& c, const std::string& entry) {
            c = wgfx::loadCompute(harmComputeSrc);
            c->entryPoint = entry;
            c->uniforms.visibility = wgpu::ShaderStage::Compute;
            c->uniforms.setUniform(harmComputeParamsUni_);
            c->uniforms.setStorage(harmGpuA_);
            c->uniforms.setStorage(harmGpuB_);
            c->init();
        };
        makeHarmCompute(harmComputeStep_, "harm_step");
        makeHarmCompute(harmComputeCopy_, "copy_b_to_a");
        resizeHarmBuffers();

        pipeline3d_ = wgfx::loadPipeline(wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "tdse3d.wgsl").c_str()));
        stateUniform3d_ = wgfx::createUniform(0, sizeof(Gpu3dState), reinterpret_cast<const float*>(&gpu3dState_));
        // setUniform done below after storage buffers are ready

        // ---- Allocate shared GPU storage buffers (kMaxTdse3dGrid³) ----
        const size_t kMaxCells = static_cast<size_t>(kMaxTdse3dGrid)
                               * static_cast<size_t>(kMaxTdse3dGrid)
                               * static_cast<size_t>(kMaxTdse3dGrid);
        // waveA and waveB: u32 packed half2 each (read-write for compute)
        gpuWaveA_ = wgfx::createStorage(1, kMaxCells * sizeof(uint32_t), nullptr, false);
        gpuWaveB_ = wgfx::createStorage(2, kMaxCells * sizeof(uint32_t), nullptr, false);
        // pot: f32 each (read-only everywhere)
        gpuPot_   = wgfx::createStorage(3, kMaxCells     * sizeof(float), nullptr, true);

        // ---- Render pipeline: read-only, fragment-only storage bindings ----
        // Wrapping the same GPU buffers without ownership to avoid double-free.
        // Fragment-only visibility avoids VERTEX_WRITABLE_STORAGE requirement.
        auto makeRenderStorage = [](wgfx::Uniform* src, int binding, bool readOnly) -> wgfx::Uniform* {
            wgfx::Uniform* u = new wgfx::Uniform();
            u->isReadOnly    = readOnly;
            u->binding       = binding;
            u->minBindingSize = src->minBindingSize;
            u->buffer        = src->buffer;    // shared handle
            u->entry.binding = static_cast<uint32_t>(binding);
            u->entry.buffer  = src->buffer;    // same as createStorage: entry.buffer = buffer
            u->entry.offset  = 0;
            u->entry.size    = static_cast<uint64_t>(src->minBindingSize);
            return u;
        };
        wgfx::Uniform* renderWaveB = makeRenderStorage(gpuWaveB_, 1, true);
        wgfx::Uniform* renderPot   = makeRenderStorage(gpuPot_,   2, true);

        pipeline3d_->uniforms.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        pipeline3d_->uniforms.setUniform(stateUniform3d_);
        // Fragment-only visibility for storage so no VERTEX_WRITABLE_STORAGE needed
        pipeline3d_->uniforms.visibility = wgpu::ShaderStage::Fragment;
        pipeline3d_->uniforms.setStorage(renderWaveB);
        pipeline3d_->uniforms.setStorage(renderPot);
        pipeline3d_->targets = 1;
        pipeline3d_->useDepth = false;
        init3dBuffers();
        pipeline3d_->init(vbo3d_.get());

        // ---- Build compute pipelines (all share one WGSL, different entry points) ----
        const std::string computeSrc = wgfx::loadFromFile(
            (std::string(RESOURCE_DIR) + "/" + "tdse3d_compute.wgsl").c_str());

        computeParamsUni_ = wgfx::createUniform(0, sizeof(GpuComputeParams),
            reinterpret_cast<const float*>(&gpuComputeParams_));

        auto makeCompute = [&](wgfx::Compute*& c, const std::string& entry) {
            c = wgfx::loadCompute(computeSrc);
            c->entryPoint = entry;   // picked up by fixed init()
            c->uniforms.visibility = wgpu::ShaderStage::Compute;
            c->uniforms.setUniform(computeParamsUni_);
            c->uniforms.setStorage(gpuWaveA_);
            c->uniforms.setStorage(gpuWaveB_);
            c->uniforms.setStorage(gpuPot_);
            c->init();
        };
        makeCompute(computeEuler_,    "euler");
        makeCompute(computeCnRhs_,    "cn_rhs");
        makeCompute(computeCnIter_,   "cn_iter");
        makeCompute(computeCnFinish_, "cn_finish");


        // ---- Initial sim state ----
        resize3dBuffers();

        // Initialize the dedicated 3D TDSE camera
        camera3d_.resetForDomain(tdse3dDomainHalf_);

        if (renderPath_ == RenderPath::Path2D) {
            pipeline = twoDUseTdse_ ? pipelineTdse2d_ : pipeline2d_;
            pipeline->setVertexBuffer(vbo2d_.get());
            pipeline->setIndexBuffer(ibo2d_.get());
        }
    }

    Quad(const Quad&) = delete;
    void operator=(const Quad&) = delete;

    void initBuffers() {
        orbitalAttribData_.assign(static_cast<size_t>(kMaxParticles) * 3, 0.0f);
        fillSingleOrbitalAttribData();

        std::vector<uint32_t> indices(static_cast<size_t>(kMaxParticles));
        for (uint32_t i = 0; i < static_cast<uint32_t>(kMaxParticles); ++i) {
            indices[i] = i;
        }

        vbo_.reset(wgfx::createVertexBuffer(orbitalAttribData_));
        vbo_->setTopology(PrimitiveTopology::PointList);
        vbo_->setAttribute(0, wgfx::vec3f, 0);

        ibo_.reset(wgfx::createIndexBufferU32(indices));
        pipelineOrbital_->setVertexBuffer(vbo_.get());
        pipelineOrbital_->setIndexBuffer(ibo_.get());
    }

    void init2dBuffers() {
        const std::vector<float> vertices = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f
        };
        const std::vector<uint16_t> indices = { 0, 1, 2, 0, 2, 3 };

        vbo2d_.reset(wgfx::createVertexBuffer(vertices));
        vbo2d_->setTopology(PrimitiveTopology::TriangleList);
        vbo2d_->setAttribute(0, wgfx::vec3f, 0);

        ibo2d_.reset(wgfx::createIndexBuffer(indices));
        pipeline2d_->setVertexBuffer(vbo2d_.get());
        pipeline2d_->setIndexBuffer(ibo2d_.get());
    }

    void init3dBuffers() {
        const std::vector<float> vertices = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f
        };
        const std::vector<uint16_t> indices = { 0, 1, 2, 0, 2, 3 };
        vbo3d_.reset(wgfx::createVertexBuffer(vertices));
        vbo3d_->setTopology(PrimitiveTopology::TriangleList);
        vbo3d_->setAttribute(0, wgfx::vec3f, 0);
        ibo3d_.reset(wgfx::createIndexBuffer(indices));
        pipeline3d_->setVertexBuffer(vbo3d_.get());
        pipeline3d_->setIndexBuffer(ibo3d_.get());
    }

    void resize3dBuffers() {
        tdse3dGridSize_ = std::clamp(tdse3dGridSize_, 8, kMaxTdse3dGrid);
        const size_t N = static_cast<size_t>(tdse3dGridSize_);
        const size_t total = N * N * N;
        tdse3dReal_.assign(total, 0.0f);
        tdse3dImag_.assign(total, 0.0f);
        tdse3dPot_.assign(total, 0.0f);
        tdse3dPotDirty_ = true;
        tdse3dNeedsReset_ = true;
    }

    void rebuild3dPotential() {
        const int N = tdse3dGridSize_;
        const float h = std::max(tdse3dDomainHalf_, 1.0f);
        const float dx = (2.0f * h) / static_cast<float>(N - 1);
        const float strength = std::max(tdse3dPotStrength_, 0.0f);
        const float radius = std::max(tdse3dPotRadius_, 0.1f);

        for (int iz = 0; iz < N; ++iz) {
            const float z = -h + static_cast<float>(iz) * dx;
            for (int iy = 0; iy < N; ++iy) {
                const float y = -h + static_cast<float>(iy) * dx;
                for (int ix = 0; ix < N; ++ix) {
                    const float x = -h + static_cast<float>(ix) * dx;
                    const size_t idx = static_cast<size_t>(iz) * static_cast<size_t>(N) * static_cast<size_t>(N)
                                     + static_cast<size_t>(iy) * static_cast<size_t>(N)
                                     + static_cast<size_t>(ix);
                    const float r = std::sqrt(x*x + y*y + z*z);
                    float v = 0.0f;
                    switch (tdse3dPotType_) {
                    case Potential3dType::Free:
                        v = 0.0f;
                        break;
                    case Potential3dType::HarmonicWell:
                        v = 0.5f * strength * r * r;
                        break;
                    case Potential3dType::CoulombWell:
                        v = -strength / std::max(r, 0.3f);
                        break;
                    case Potential3dType::SphericalBarrier:
                        v = (r <= radius) ? 0.0f : strength;
                        break;
                    case Potential3dType::DoubleWell: {
                        const float r1 = std::sqrt((x - radius * 0.5f)*(x - radius * 0.5f) + y*y + z*z);
                        const float r2 = std::sqrt((x + radius * 0.5f)*(x + radius * 0.5f) + y*y + z*z);
                        v = -strength / std::max(r1, 0.3f) - strength / std::max(r2, 0.3f);
                        break;
                    }
                    }
                    tdse3dPot_[idx] = v;
                }
            }
        }
        tdse3dPotDirty_ = false;
        // Upload potential to GPU (read-only buffer)
        if (gpuPot_) {
            wgfx::queue.writeBuffer(gpuPot_->buffer, 0,
                tdse3dPot_.data(), tdse3dPot_.size() * sizeof(float));
        }
    }

    void reset3dWavefunction() {
        if (tdse3dPotDirty_) { rebuild3dPotential(); }
        const int N = tdse3dGridSize_;
        const float h = std::max(tdse3dDomainHalf_, 1.0f);
        const float dx = (2.0f * h) / static_cast<float>(N - 1);
        const float sig = std::max(tdse3dSigma_, 0.1f);
        const float inv2s2 = 1.0f / (2.0f * sig * sig);
        const glm::vec3 c = tdse3dPacketPos_;
        const glm::vec3 p = tdse3dPacketMom_;

        float norm = 0.0f;
        for (int iz = 0; iz < N; ++iz) {
            const float z = -h + static_cast<float>(iz) * dx;
            for (int iy = 0; iy < N; ++iy) {
                const float y = -h + static_cast<float>(iy) * dx;
                for (int ix = 0; ix < N; ++ix) {
                    const float x = -h + static_cast<float>(ix) * dx;
                    const size_t idx = static_cast<size_t>(iz)*static_cast<size_t>(N)*static_cast<size_t>(N)
                                     + static_cast<size_t>(iy)*static_cast<size_t>(N)
                                     + static_cast<size_t>(ix);
                    const float dx_ = x - c.x;
                    const float dy_ = y - c.y;
                    const float dz_ = z - c.z;
                    const float env = std::exp(-(dx_*dx_ + dy_*dy_ + dz_*dz_) * inv2s2);
                    const float phase = p.x * dx_ + p.y * dy_ + p.z * dz_;
                    tdse3dReal_[idx] = env * std::cos(phase);
                    tdse3dImag_[idx] = env * std::sin(phase);
                    norm += (tdse3dReal_[idx]*tdse3dReal_[idx] + tdse3dImag_[idx]*tdse3dImag_[idx]);
                }
            }
        }
        if (norm > 1e-12f) {
            const float inv = 1.0f / std::sqrt(norm);
            for (size_t i = 0; i < tdse3dReal_.size(); ++i) {
                tdse3dReal_[i] *= inv;
                tdse3dImag_[i] *= inv;
            }
        }
        tdse3dNeedsReset_ = false;
        tdse3dNormCounter_ = 0;
        tdse3dTime_ = 0.0f;
        // Pack (re, im) interleaved and upload to waveA on GPU
        if (gpuWaveA_) {
            std::vector<uint32_t> packed(tdse3dReal_.size());
            for (size_t i = 0; i < tdse3dReal_.size(); ++i) {
                packed[i] = glm::packHalf2x16(glm::vec2(tdse3dReal_[i], tdse3dImag_[i]));
            }
            wgfx::queue.writeBuffer(gpuWaveA_->buffer, 0, packed.data(), packed.size() * sizeof(uint32_t));
            wgfx::queue.writeBuffer(gpuWaveB_->buffer, 0, packed.data(), packed.size() * sizeof(uint32_t));
        }
    }

    void upload3dWaveToGpu() {
        if (!gpuWaveA_ || !gpuWaveB_ || tdse3dReal_.size() != tdse3dImag_.size()) {
            return;
        }
        std::vector<uint32_t> packed(tdse3dReal_.size());
        for (size_t i = 0; i < tdse3dReal_.size(); ++i) {
            packed[i] = glm::packHalf2x16(glm::vec2(tdse3dReal_[i], tdse3dImag_[i]));
        }
        wgfx::queue.writeBuffer(gpuWaveA_->buffer, 0, packed.data(), packed.size() * sizeof(uint32_t));
        wgfx::queue.writeBuffer(gpuWaveB_->buffer, 0, packed.data(), packed.size() * sizeof(uint32_t));
    }

#ifdef ATOMS_ENABLE_ONNX
    bool loadOnnxModel3d(const std::string& modelPath) {
        try {
            if (modelPath.empty()) {
                tdse3dOnnxStatus_ = "ONNX load failed: model path is empty";
                tdse3dOnnxLoaded_ = false;
                return false;
            }

            if (!std::filesystem::exists(modelPath)) {
                tdse3dOnnxStatus_ = "ONNX load failed: file not found";
                tdse3dOnnxLoaded_ = false;
                return false;
            }

            if (!onnxEnv_) {
                onnxEnv_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "atoms_tdse3d");
            }

            onnxSessionOptions_ = std::make_unique<Ort::SessionOptions>();
            onnxSessionOptions_->SetIntraOpNumThreads(1);
            onnxSessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            onnxSession_ = std::make_unique<Ort::Session>(*onnxEnv_, modelPath.c_str(), *onnxSessionOptions_);

            Ort::AllocatorWithDefaultOptions allocator;
            const size_t inputCount = onnxSession_->GetInputCount();
            const size_t outputCount = onnxSession_->GetOutputCount();

            onnxInputNamesStorage_.clear();
            onnxOutputNamesStorage_.clear();
            onnxInputNames_.clear();
            onnxOutputNames_.clear();

            for (size_t i = 0; i < inputCount; ++i) {
                auto name = onnxSession_->GetInputNameAllocated(i, allocator);
                onnxInputNamesStorage_.push_back(name.get() ? name.get() : "");
            }
            for (size_t i = 0; i < outputCount; ++i) {
                auto name = onnxSession_->GetOutputNameAllocated(i, allocator);
                onnxOutputNamesStorage_.push_back(name.get() ? name.get() : "");
            }
            for (const std::string& n : onnxInputNamesStorage_) {
                onnxInputNames_.push_back(n.c_str());
            }
            for (const std::string& n : onnxOutputNamesStorage_) {
                onnxOutputNames_.push_back(n.c_str());
            }

            if (onnxInputNames_.empty() || onnxOutputNames_.empty()) {
                tdse3dOnnxStatus_ = "ONNX load failed: model has no inputs or outputs";
                tdse3dOnnxLoaded_ = false;
                return false;
            }

            tdse3dOnnxLoaded_ = true;
            tdse3dOnnxStatus_ = "ONNX model loaded";
            return true;
        } catch (const std::exception& e) {
            tdse3dOnnxLoaded_ = false;
            tdse3dOnnxStatus_ = std::string("ONNX load failed: ") + e.what();
            return false;
        }
    }

    bool runOnnxRollout3d() {
        try {
            if (!tdse3dOnnxLoaded_ || !onnxSession_) {
                tdse3dOnnxStatus_ = "ONNX rollout skipped: no model loaded";
                return false;
            }

            if (tdse3dPotDirty_) {
                rebuild3dPotential();
            }
            if (tdse3dNeedsReset_) {
                reset3dWavefunction();
            }

            const int N = tdse3dGridSize_;
            const size_t cells = static_cast<size_t>(N) * static_cast<size_t>(N) * static_cast<size_t>(N);
            if (cells == 0 || tdse3dPot_.size() != cells || tdse3dReal_.size() != cells || tdse3dImag_.size() != cells) {
                tdse3dOnnxStatus_ = "ONNX rollout failed: invalid 3D buffer sizes";
                return false;
            }

            tdse3dOnnxInput_.resize(cells * 3);
            std::memcpy(tdse3dOnnxInput_.data() + 0 * cells, tdse3dPot_.data(),  cells * sizeof(float));
            std::memcpy(tdse3dOnnxInput_.data() + 1 * cells, tdse3dReal_.data(), cells * sizeof(float));
            std::memcpy(tdse3dOnnxInput_.data() + 2 * cells, tdse3dImag_.data(), cells * sizeof(float));

            const std::array<int64_t, 5> inputShape = {1, 3, N, N, N};
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                tdse3dOnnxInput_.data(),
                tdse3dOnnxInput_.size(),
                inputShape.data(),
                inputShape.size());

            std::vector<Ort::Value> outputs = onnxSession_->Run(
                Ort::RunOptions{nullptr},
                onnxInputNames_.data(),
                &inputTensor,
                1,
                onnxOutputNames_.data(),
                onnxOutputNames_.size());

            if (outputs.empty() || !outputs[0].IsTensor()) {
                tdse3dOnnxStatus_ = "ONNX rollout failed: no tensor output";
                return false;
            }

            Ort::TensorTypeAndShapeInfo outInfo = outputs[0].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> outShape = outInfo.GetShape();
            if (outShape.size() != 5 || outShape[1] != 2 || outShape[2] != N || outShape[3] != N || outShape[4] != N) {
                tdse3dOnnxStatus_ = "ONNX rollout failed: output shape mismatch";
                return false;
            }

            const float* outData = outputs[0].GetTensorData<float>();
            if (!outData) {
                tdse3dOnnxStatus_ = "ONNX rollout failed: null output data";
                return false;
            }

            for (size_t i = 0; i < cells; ++i) {
                tdse3dReal_[i] = outData[i];
                tdse3dImag_[i] = outData[cells + i];
            }
            upload3dWaveToGpu();
            tdse3dTime_ += std::max(tdse3dDt_, 0.0f);
            tdse3dOnnxStatus_ = "ONNX rollout complete";
            return true;
        } catch (const std::exception& e) {
            tdse3dOnnxStatus_ = std::string("ONNX rollout failed: ") + e.what();
            return false;
        }
    }
#else
    bool loadOnnxModel3d(const std::string&) {
        tdse3dOnnxLoaded_ = false;
        tdse3dOnnxStatus_ = "ONNX support disabled at compile time";
        return false;
    }

    bool runOnnxRollout3d() {
        tdse3dOnnxStatus_ = "ONNX support disabled at compile time";
        return false;
    }
#endif

#ifdef ATOMS_ENABLE_TORCHSCRIPT
    bool loadTorchscriptModel3d(const std::string& modelPath) {
        try {
            if (modelPath.empty()) {
                tdse3dTorchscriptStatus_ = "TorchScript load failed: model path is empty";
                tdse3dTorchscriptLoaded_ = false;
                return false;
            }
            if (!std::filesystem::exists(modelPath)) {
                tdse3dTorchscriptStatus_ = "TorchScript load failed: file not found";
                tdse3dTorchscriptLoaded_ = false;
                return false;
            }

            auto module = std::make_unique<torch::jit::script::Module>(torch::jit::load(modelPath));
            module->eval();
            torchscriptModule_ = std::move(module);

            tdse3dTorchscriptLoaded_ = true;
            tdse3dTorchscriptStatus_ = "TorchScript model loaded";
            return true;
        } catch (const std::exception& e) {
            tdse3dTorchscriptLoaded_ = false;
            tdse3dTorchscriptStatus_ = std::string("TorchScript load failed: ") + e.what();
            return false;
        }
    }

    bool runTorchscriptRollout3d() {
        try {
            if (!tdse3dTorchscriptLoaded_ || !torchscriptModule_) {
                tdse3dTorchscriptStatus_ = "TorchScript rollout skipped: no model loaded";
                return false;
            }

            if (tdse3dGridSize_ != tdse3dTorchscriptExpectedGrid_) {
                tdse3dTorchscriptStatus_ =
                    "TorchScript rollout blocked: grid mismatch. Set grid N to "
                    + std::to_string(tdse3dTorchscriptExpectedGrid_) +
                    " and reset 3D wavefunction.";
                return false;
            }

            if (tdse3dPotDirty_) {
                rebuild3dPotential();
            }
            if (tdse3dNeedsReset_) {
                reset3dWavefunction();
            }

            const int N = tdse3dGridSize_;
            const size_t cells = static_cast<size_t>(N) * static_cast<size_t>(N) * static_cast<size_t>(N);
            if (cells == 0 || tdse3dPot_.size() != cells || tdse3dReal_.size() != cells || tdse3dImag_.size() != cells) {
                tdse3dTorchscriptStatus_ = "TorchScript rollout failed: invalid 3D buffer sizes";
                return false;
            }

            std::vector<float> inputData(cells * 3);
            std::memcpy(inputData.data() + 0 * cells, tdse3dPot_.data(), cells * sizeof(float));
            std::memcpy(inputData.data() + 1 * cells, tdse3dReal_.data(), cells * sizeof(float));
            std::memcpy(inputData.data() + 2 * cells, tdse3dImag_.data(), cells * sizeof(float));

            torch::NoGradGuard noGrad;
            torch::Tensor input = torch::from_blob(inputData.data(), {1, 3, N, N, N}, torch::kFloat32).clone();
            torch::Tensor output = torchscriptModule_->forward({input}).toTensor().to(torch::kCPU).contiguous();

            if (output.dim() != 5 || output.size(0) != 1 || output.size(1) != 2 || output.size(2) != N || output.size(3) != N || output.size(4) != N) {
                tdse3dTorchscriptStatus_ = "TorchScript rollout failed: output shape mismatch";
                return false;
            }

            const float* outData = output.data_ptr<float>();
            if (!outData) {
                tdse3dTorchscriptStatus_ = "TorchScript rollout failed: null output data";
                return false;
            }

            for (size_t i = 0; i < cells; ++i) {
                tdse3dReal_[i] = outData[i];
                tdse3dImag_[i] = outData[cells + i];
            }
            upload3dWaveToGpu();
            tdse3dTime_ += std::max(tdse3dDt_, 0.0f);
            tdse3dTorchscriptStatus_ = "TorchScript rollout complete";
            return true;
        } catch (const std::exception& e) {
            tdse3dTorchscriptStatus_ = std::string("TorchScript rollout failed: ") + e.what();
            return false;
        }
    }
#else
    bool loadTorchscriptModel3d(const std::string&) {
        tdse3dTorchscriptLoaded_ = false;
        tdse3dTorchscriptStatus_ = "TorchScript support disabled at compile time";
        return false;
    }

    bool runTorchscriptRollout3d() {
        tdse3dTorchscriptStatus_ = "TorchScript support disabled at compile time";
        return false;
    }
#endif

    // Dispatch GPU compute for one FDTD substep.
    // Called by render3d(); the ComputePass is owned externally in main.cpp.
    // Returns the number of workgroups per axis.
    uint32_t computeWorkgroups() const {
        return (static_cast<uint32_t>(tdse3dGridSize_) + 3u) / 4u;
    }

    void dispatchCompute(wgfx::ComputePass& cp) {
        if (tdse3dUseTorchscript_) {
            int steps = 0;
            if (tdse3dTorchscriptAutoplay_) {
                steps = std::clamp(tdse3dTorchscriptStepsPerFrame_, 1, 32);
            }
            if (tdse3dTorchscriptRunRequested_) {
                steps = std::max(steps, 1);
                tdse3dTorchscriptRunRequested_ = false;
            }

            for (int i = 0; i < steps; ++i) {
                if (!runTorchscriptRollout3d()) {
                    break;
                }
            }
            return;
        }

        if (tdse3dUseOnnx_) {
            if (tdse3dOnnxRunRequested_) {
                runOnnxRollout3d();
                tdse3dOnnxRunRequested_ = false;
            }
            return;
        }

        if (tdse3dPotDirty_)   { rebuild3dPotential(); }
        if (tdse3dNeedsReset_) { reset3dWavefunction(); }
        if (!computeEuler_)    { return; }

        const float h  = std::max(tdse3dDomainHalf_, 1.0f);
        const float dx = (2.0f * h) / static_cast<float>(tdse3dGridSize_ - 1);
        const float absW = tdse3dUseAbsorbing_
            ? std::clamp(tdse3dAbsorbWidth_, 0.1f, h * 0.9f) : 0.0f;

        gpuComputeParams_.gridN       = static_cast<uint32_t>(tdse3dGridSize_);
        gpuComputeParams_.mode        = static_cast<uint32_t>(tdse3dIntegrator_);
        gpuComputeParams_.dt          = std::clamp(tdse3dDt_, 1e-6f, 2.0f);
        gpuComputeParams_.absorbWidth = absW;
        gpuComputeParams_.absorbStr   = std::max(tdse3dAbsorbStrength_, 0.0f);
        gpuComputeParams_.dx          = dx;

        // Write params ONCE at offset 0 — bypasses the accumulating dynamic
        // offset counter that would overflow after ~106 frames.
        wgfx::queue.writeBuffer(computeParamsUni_->buffer, 0,
            &gpuComputeParams_, sizeof(GpuComputeParams));

        // Pin every compute pipeline's dynamic offset for binding 0 to 0.
        // This must match the write above (offset 0 in the uniform buffer).
        auto pinOffset = [](wgfx::Compute* c) {
            if (!c) return;
            c->uniforms.dynamicOffsets.resize(1, 0);
            c->uniforms.dynamicOffsets[0] = 0;
            // Also reset the quantity so clear() won't accidentally re-advance it.
            if (!c->uniforms.uniforms.empty())
                c->uniforms.uniforms[0]->quantity = 0;
        };
        pinOffset(computeEuler_);
        pinOffset(computeCnRhs_);
        pinOffset(computeCnIter_);
        pinOffset(computeCnFinish_);

        const uint32_t wg = computeWorkgroups();
        const int substeps = std::clamp(tdse3dSubsteps_, 1, 32);
        const int CN_ITERS = 4;

        for (int s = 0; s < substeps; ++s) {
            if (tdse3dIntegrator_ == 0) {
                cp.drawXYZ(computeEuler_, wg, wg, wg);
            } else {
                cp.drawXYZ(computeCnRhs_, wg, wg, wg);
                for (int iter = 0; iter < CN_ITERS; ++iter) {
                    cp.drawXYZ(computeCnIter_, wg, wg, wg);
                }
                cp.drawXYZ(computeCnFinish_, wg, wg, wg);
            }
            std::swap(gpuWaveA_, gpuWaveB_);
            tdse3dTime_ += gpuComputeParams_.dt;
        }
    }


    void step3dSimulation() { /* replaced by dispatchCompute */ }

    void upload3dField() { /* no-op: GPU writes directly to waveB render buffer */ }

    // Kept for reference — no longer called:


    void render3d(float dt) {
        // Simulation is now dispatched via dispatchCompute3d() BEFORE the render pass.
        // Here we only update camera + render uniforms.
        int width = 1280, height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);

        ImGuiIO& io = ImGui::GetIO();
        const float wheel = Context::Instance().consumeWheelDelta();
        camera3d_.zoomSpeed = std::max(tdse3dDomainHalf_ * 0.05f, 0.5f);
        camera3d_.process(dt, !io.WantCaptureMouse, wheel);

        const float nearPlane = std::max(tdse3dDomainHalf_ * 0.01f, 0.01f);
        const float farPlane  = tdse3dDomainHalf_ * 20.0f;
        const glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, nearPlane, farPlane);
        const glm::mat4 view = glm::lookAt(camera3d_.position(), camera3d_.target, glm::vec3(0, 1, 0));
        const glm::mat4 vp   = proj * view;
        const glm::mat4 invVP = glm::inverse(vp);

        gpu3dState_.invViewProj = invVP;
        const glm::vec3 cp = camera3d_.position();
        gpu3dState_.camPos  = glm::vec4(cp, tdse3dTime_);
        gpu3dState_.params  = glm::vec4(
            static_cast<float>(tdse3dGridSize_),
            std::max(tdse3dDomainHalf_, 0.5f),
            std::max(tdse3dIntensityScale_, 0.0001f),
            std::max(tdse3dIntensityRange_, 0.0001f));
        gpu3dState_.render  = glm::vec4(
            static_cast<float>(tdse3dColorMode_),
            aspect,
            tdse3dShowPotential_ ? 1.0f : 0.0f,
            static_cast<float>(tdse3dSliceAxis_));
        gpu3dState_.march   = glm::vec4(
            tdse3dMarchSteps_,
            tdse3dAlphaScale_,
            tdse3dSlicePos_,
            0.0f);

        writeRenderUniform(pipeline3d_, reinterpret_cast<const float*>(&gpu3dState_));
        pipeline3d_->setVertexBuffer(vbo3d_.get());
        pipeline3d_->setIndexBuffer(ibo3d_.get());
        pipeline = pipeline3d_;
    }


    static float factorialIntCpu(int v) {
        if (v <= 1) {
            return 1.0f;
        }
        float out = 1.0f;
        for (int i = 2; i <= v; ++i) {
            out *= static_cast<float>(i);
        }
        return out;
    }

    static float associatedLaguerreCpu(int k, int alpha, float x) {
        if (k <= 0) {
            return 1.0f;
        }
        float lm2 = 1.0f;
        float lm1 = 1.0f + static_cast<float>(alpha) - x;
        if (k == 1) {
            return lm1;
        }
        float l = lm1;
        for (int j = 2; j <= k; ++j) {
            l = ((static_cast<float>(2 * j - 1 + alpha) - x) * lm1 - static_cast<float>(j - 1 + alpha) * lm2) / static_cast<float>(j);
            lm2 = lm1;
            lm1 = l;
        }
        return l;
    }

    static float associatedLegendreCpu(int l, int mAbs, float x) {
        float pmm = 1.0f;
        if (mAbs > 0) {
            const float somx2 = std::sqrt(std::max(0.0f, (1.0f - x) * (1.0f + x)));
            float fact = 1.0f;
            for (int j = 1; j <= mAbs; ++j) {
                pmm = pmm * (-fact) * somx2;
                fact += 2.0f;
            }
        }
        if (l == mAbs) {
            return pmm;
        }
        float pm1m = x * static_cast<float>(2 * mAbs + 1) * pmm;
        if (l == mAbs + 1) {
            return pm1m;
        }
        float pll = pm1m;
        for (int ll = mAbs + 2; ll <= l; ++ll) {
            pll = ((static_cast<float>(2 * ll - 1) * x * pm1m) - (static_cast<float>(ll + mAbs - 1) * pmm)) / static_cast<float>(ll - mAbs);
            pmm = pm1m;
            pm1m = pll;
        }
        return pll;
    }

    float orbitalWavefunctionSliceReal(float x, float y, int n, int l, int m) const {
        const float r = std::sqrt(x * x + y * y);
        const float theta = std::acos(std::clamp(0.0f / std::max(r, 1e-6f), -1.0f, 1.0f));
        const float rho = 2.0f * r / std::max(static_cast<float>(n), 1.0f);
        const int k = n - l - 1;
        const int alpha = 2 * l + 1;
        const float laguerre = associatedLaguerreCpu(std::max(k, 0), alpha, rho);
        const float nn = std::max(static_cast<float>(n), 1.0f);
        const float num = factorialIntCpu(std::max(n - l - 1, 0));
        const float den = std::max(factorialIntCpu(std::max(n + l, 0)), 1e-8f);
        const float norm = std::pow(2.0f / nn, 3.0f) * num / (2.0f * nn * den);
        const float radial = std::sqrt(std::max(norm, 0.0f)) * std::exp(-rho * 0.5f) * std::pow(std::max(rho, 1e-6f), static_cast<float>(l)) * laguerre;

        const float xLeg = std::cos(theta);
        const int mAbs = std::abs(m);
        const float plm = associatedLegendreCpu(l, mAbs, xLeg);
        const float angNum = factorialIntCpu(std::max(l - mAbs, 0));
        const float angDen = std::max(factorialIntCpu(std::max(l + mAbs, 0)), 1e-8f);
        const float yNorm = (static_cast<float>(2 * l + 1) / (4.0f * kPi)) * (angNum / angDen);
        const float angularAmp = std::sqrt(std::max(yNorm, 0.0f)) * plm;

        const float phi = std::atan2(y, x);
        const float phase = static_cast<float>(m) * phi;
        return radial * angularAmp * std::cos(phase);
    }

    void resizeTdseBuffers() {
        tdseGridSize_ = std::clamp(tdseGridSize_, 64, kMaxTdseGrid);
        const size_t cellCount = static_cast<size_t>(tdseGridSize_) * static_cast<size_t>(tdseGridSize_);
        tdseReal_.assign(cellCount, 0.0f);
        tdseImag_.assign(cellCount, 0.0f);
        tdseNextReal_.assign(cellCount, 0.0f);
        tdseNextImag_.assign(cellCount, 0.0f);
        tdseRhsReal_.assign(cellCount, 0.0f);
        tdseRhsImag_.assign(cellCount, 0.0f);
        tdsePotential_.assign(cellCount, 0.0f);
        tdseUpload_.assign(cellCount * 4, 0.0f);
        tdsePotentialDirty_ = true;
        tdseWaveNeedsReset_ = true;
    }

    void rebuildTdsePotential() {
        if (tdsePotential_.empty()) {
            return;
        }

        const int n = tdseGridSize_;
        const float domain = std::max(tdseDomainHalfExtent_, 1.0f);
        const float dx = (2.0f * domain) / static_cast<float>(n - 1);
        const float strength = std::max(tdsePotentialStrength_, 0.0f);
        const float halfWidth = std::clamp(tdseSquareHalfWidth_, 0.1f, domain * 0.95f);
        const float radiusBig = std::clamp(tdseCircleRadiusBig_, 0.2f, domain * 0.95f);
        const float radiusSmall = std::clamp(tdseCircleRadiusSmall_, 0.2f, domain * 0.95f);
        const float annIn = std::clamp(tdseAnnulusInner_, 0.1f, domain * 0.9f);
        const float annOut = std::clamp(tdseAnnulusOuter_, annIn + 0.1f, domain * 0.95f);
        const float barrierHalfWidth = std::clamp(tdseBarrierHalfWidth_, 0.06f, domain * 0.2f);
        const float slitCenter = std::clamp(tdseSlitCenterOffset_, 0.0f, domain * 0.8f);
        const float slitHalfH = std::clamp(tdseSlitHalfHeight_, 0.05f, domain * 0.4f);

        for (int iy = 0; iy < n; ++iy) {
            const float y = -domain + static_cast<float>(iy) * dx;
            for (int ix = 0; ix < n; ++ix) {
                const float x = -domain + static_cast<float>(ix) * dx;
                float v = 0.0f;
                const float r = std::sqrt(x * x + y * y);

                switch (tdsePotentialType_) {
                case Potential2dType::SquareWell:
                    if (std::abs(x) <= halfWidth && std::abs(y) <= halfWidth) {
                        v = -strength;
                    }
                    break;
                case Potential2dType::CircleWellBig:
                    if (r <= radiusBig) {
                        v = -strength;
                    }
                    break;
                case Potential2dType::CircleWellSmall:
                    if (r <= radiusSmall) {
                        v = -strength;
                    }
                    break;
                case Potential2dType::DoubleSlitWell: {
                    const bool inBarrier = (std::abs(x) <= barrierHalfWidth);
                    const bool inSlitA = std::abs(y - slitCenter) <= slitHalfH;
                    const bool inSlitB = std::abs(y + slitCenter) <= slitHalfH;
                    if (inBarrier && !(inSlitA || inSlitB)) {
                        v = 2.0f * strength;
                    }
                    break;
                }
                case Potential2dType::AnnularWell: {
                    if (r >= annIn && r <= annOut) {
                        v = -strength;
                    }
                    break;
                }
                }

                tdsePotential_[static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix)] = v;
            }
        }

        tdsePotentialDirty_ = false;
    }

    void resetTdseWavefunction() {
        if (tdsePotentialDirty_) {
            rebuildTdsePotential();
        }

        const int n = tdseGridSize_;
        const float domain = std::max(tdseDomainHalfExtent_, 1.0f);
        const float dx = (2.0f * domain) / static_cast<float>(n - 1);
        const glm::vec2 center(
            std::clamp(tdsePacketCenter_.x, -domain * 0.95f, domain * 0.95f),
            std::clamp(tdsePacketCenter_.y, -domain * 0.95f, domain * 0.95f));
        tdsePacketCenter_ = center;
        const glm::vec2 momentum = tdsePacketMomentum_;
        const float sigma = std::max(domain * 0.12f, 0.2f);
        const float inv2Sigma2 = 1.0f / std::max(2.0f * sigma * sigma, 1e-6f);

        float norm = 0.0f;
        for (int iy = 0; iy < n; ++iy) {
            const float y = -domain + static_cast<float>(iy) * dx;
            for (int ix = 0; ix < n; ++ix) {
                const float x = -domain + static_cast<float>(ix) * dx;
                const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix);
                const float dxp = x - center.x;
                const float dyp = y - center.y;
                const float env = std::exp(-(dxp * dxp + dyp * dyp) * inv2Sigma2);
                const float phase = momentum.x * dxp + momentum.y * dyp;
                tdseReal_[idx] = env * std::cos(phase);
                tdseImag_[idx] = env * std::sin(phase);
                norm += (tdseReal_[idx] * tdseReal_[idx] + tdseImag_[idx] * tdseImag_[idx]) * dx * dx;
            }
        }

        if (norm > 1e-12f) {
            const float inv = 1.0f / std::sqrt(norm);
            for (size_t i = 0; i < tdseReal_.size(); ++i) {
                tdseReal_[i] *= inv;
                tdseImag_[i] *= inv;
            }
        }

        tdseWaveNeedsReset_ = false;
        tdseNormCounter_ = 0;
        twoDTime_ = 0.0f;
    }

void stepTdseSimulation() {
        if (!twoDUseTdse_) {
            return;
        }
        if (tdsePotentialDirty_) {
            rebuildTdsePotential();
        }
        if (tdseWaveNeedsReset_) {
            resetTdseWavefunction();
        }

        const int n = tdseGridSize_;
        if (n < 8 || tdseReal_.size() != static_cast<size_t>(n) * static_cast<size_t>(n)) {
            return;
        }

        const float domain = std::max(tdseDomainHalfExtent_, 1.0f);
        const float dx = (2.0f * domain) / static_cast<float>(n - 1);
        // Mikaberidze's implementation assumes dx = 1.0 (grid units) for the Laplacian.
        // We set invDx2 = 1.0f here to perfectly match the stability and wave dynamics of the JS version.
        const float invDx2 = 1.0f; // previously: 1.0f / std::max(dx * dx, 1e-8f);
        const float dt = std::clamp(tdseDt_, 1e-6f, 1.0f);
        const int steps = std::clamp(tdseSubstepsPerFrame_, 1, 32);
        const float absorbWidth = std::clamp(tdseAbsorbWidth_, 0.1f, domain * 0.9f);
        const float absorbStrength = std::max(tdseAbsorbStrength_, 0.0f);

        for (int step = 0; step < steps; ++step) {
            if (tdseIntegrator_ == 0) {
                // Euler (Explicit)
                for (int iy = 1; iy < n - 1; ++iy) {
                    for (int ix = 1; ix < n - 1; ++ix) {
                        const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix);
                        const size_t left = idx - 1;
                        const size_t right = idx + 1;
                        const size_t down = idx - static_cast<size_t>(n);
                        const size_t up = idx + static_cast<size_t>(n);

                        const float lapI = (tdseImag_[left] + tdseImag_[right] + tdseImag_[down] + tdseImag_[up] - 4.0f * tdseImag_[idx]) * invDx2;
                        const float lapR = (tdseReal_[left] + tdseReal_[right] + tdseReal_[down] + tdseReal_[up] - 4.0f * tdseReal_[idx]) * invDx2;
                        const float v = tdsePotential_[idx];

                        tdseNextReal_[idx] = tdseReal_[idx] + dt * (-0.5f * lapI + v * tdseImag_[idx]);
                        tdseNextImag_[idx] = tdseImag_[idx] + dt * (0.5f * lapR - v * tdseReal_[idx]);
                    }
                }
            } else {
                // Crank-Nicolson
                // 1) Explicit RHS
                for (int iy = 1; iy < n - 1; ++iy) {
                    for (int ix = 1; ix < n - 1; ++ix) {
                        const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix);
                        const size_t left = idx - 1;
                        const size_t right = idx + 1;
                        const size_t down = idx - static_cast<size_t>(n);
                        const size_t up = idx + static_cast<size_t>(n);

                        const float lapI = (tdseImag_[left] + tdseImag_[right] + tdseImag_[down] + tdseImag_[up] - 4.0f * tdseImag_[idx]) * invDx2;
                        const float lapR = (tdseReal_[left] + tdseReal_[right] + tdseReal_[down] + tdseReal_[up] - 4.0f * tdseReal_[idx]) * invDx2;
                        const float v = tdsePotential_[idx];

                        tdseRhsReal_[idx] = tdseReal_[idx] + 0.5f * dt * (-0.5f * lapI + v * tdseImag_[idx]);
                        tdseRhsImag_[idx] = tdseImag_[idx] + 0.5f * dt * (0.5f * lapR - v * tdseReal_[idx]);
                    }
                }

                // 2) Fixed-point iteration
                const int CN_ITERS = 4;
                for (int iter = 0; iter < CN_ITERS; ++iter) {
                    for (int iy = 1; iy < n - 1; ++iy) {
                        for (int ix = 1; ix < n - 1; ++ix) {
                            const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix);
                            const size_t left = idx - 1;
                            const size_t right = idx + 1;
                            const size_t down = idx - static_cast<size_t>(n);
                            const size_t up = idx + static_cast<size_t>(n);

                            const float lapI = (tdseImag_[left] + tdseImag_[right] + tdseImag_[down] + tdseImag_[up] - 4.0f * tdseImag_[idx]) * invDx2;
                            const float lapR = (tdseReal_[left] + tdseReal_[right] + tdseReal_[down] + tdseReal_[up] - 4.0f * tdseReal_[idx]) * invDx2;
                            const float v = tdsePotential_[idx];

                            tdseNextReal_[idx] = tdseRhsReal_[idx] + 0.5f * dt * (-0.5f * lapI + v * tdseImag_[idx]);
                            tdseNextImag_[idx] = tdseRhsImag_[idx] + 0.5f * dt * (0.5f * lapR - v * tdseReal_[idx]);
                        }
                    }

                    if (iter < CN_ITERS - 1) {
                        tdseReal_.swap(tdseNextReal_);
                        tdseImag_.swap(tdseNextImag_);
                    }
                }
            }

            // Boundary zeroing
            for (int i = 0; i < n; ++i) {
                const size_t top = static_cast<size_t>(i);
                const size_t bottom = static_cast<size_t>(n - 1) * static_cast<size_t>(n) + static_cast<size_t>(i);
                const size_t left = static_cast<size_t>(i) * static_cast<size_t>(n);
                const size_t right = left + static_cast<size_t>(n - 1);
                tdseNextReal_[top] = tdseNextReal_[top + static_cast<size_t>(n)];
                tdseNextImag_[top] = tdseNextImag_[top + static_cast<size_t>(n)];
                tdseNextReal_[bottom] = tdseNextReal_[bottom - static_cast<size_t>(n)];
                tdseNextImag_[bottom] = tdseNextImag_[bottom - static_cast<size_t>(n)];
                tdseNextReal_[left] = tdseNextReal_[left + 1];
                tdseNextImag_[left] = tdseNextImag_[left + 1];
                tdseNextReal_[right] = tdseNextReal_[right - 1];
                tdseNextImag_[right] = tdseNextImag_[right - 1];
            }

            if (tdseUseAbsorbingBoundary_) {
                for (int iy = 0; iy < n; ++iy) {
                    for (int ix = 0; ix < n; ++ix) {
                        const float edgeCells = static_cast<float>(std::min(std::min(ix, n - 1 - ix), std::min(iy, n - 1 - iy)));
                        const float edgeDist = edgeCells * dx;
                        if (edgeDist < absorbWidth) {
                            const float s = 1.0f - (edgeDist / absorbWidth);
                            const float damping = std::exp(-absorbStrength * s * s * dt);
                            const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix);
                            tdseNextReal_[idx] *= damping;
                            tdseNextImag_[idx] *= damping;
                        }
                    }
                }
            }

            tdseReal_.swap(tdseNextReal_);
            tdseImag_.swap(tdseNextImag_);
        }

        ++tdseNormCounter_;
        if (tdseNormCounter_ >= 10) {
            float norm = 0.0f;
            for (size_t i = 0; i < tdseReal_.size(); ++i) {
                // Mikaberidze integrates without dx*dx scaling
                norm += (tdseReal_[i] * tdseReal_[i] + tdseImag_[i] * tdseImag_[i]); // previously: * dx * dx
            }
            if (norm > 1e-12f) {
                const float inv = 1.0f / std::sqrt(norm);
                for (size_t i = 0; i < tdseReal_.size(); ++i) {
                    tdseReal_[i] *= inv;
                    tdseImag_[i] *= inv;
                }
            }
            tdseNormCounter_ = 0;
        }
    }

    void uploadTdseField() {
        if (!tdseStorage_ || tdseUpload_.empty() || tdsePotential_.empty()) {
            return;
        }
        float maxAbsPotential = 0.0f;
        for (float v : tdsePotential_) {
            maxAbsPotential = std::max(maxAbsPotential, std::abs(v));
        }
        const float invMaxAbsPotential = (maxAbsPotential > 1e-6f) ? (1.0f / maxAbsPotential) : 0.0f;

        for (size_t i = 0; i < tdseReal_.size(); ++i) {
            const float re = tdseReal_[i];
            const float im = tdseImag_[i];
            const float rho = re * re + im * im;
            float phase01 = std::atan2(im, re) / (2.0f * kPi);
            if (phase01 < 0.0f) {
                phase01 += 1.0f;
            }
            const float pot01 = 0.5f + 0.5f * tdsePotential_[i] * invMaxAbsPotential;

            const size_t base = i * 4;
            tdseUpload_[base + 0] = rho;
            tdseUpload_[base + 1] = phase01;
            tdseUpload_[base + 2] = pot01;
            tdseUpload_[base + 3] = 1.0f;
        }

        pipeline2d_->uniforms.updateStorageBuffer(tdseStorage_, tdseUpload_.data(), tdseUpload_.size() * sizeof(float));
    }

    void resizeGrmhdBuffers() {
        grmhdGridSize_ = std::clamp(grmhdGridSize_, 64, kMaxGrmhdGrid);
        const size_t cellCount = static_cast<size_t>(grmhdGridSize_) * static_cast<size_t>(grmhdGridSize_);
        grmhd_.assign(cellCount, GrmhdCell{});
        grmhdNext_.assign(cellCount, GrmhdCell{});
        grmhdUpload_.assign(cellCount * 8, 0.0f);
        grmhdNeedsReset_ = true;
    }

    GrmhdCell grmhdPrimitiveFlux(const GrmhdCell& u, int axis) const {
        constexpr float gammaGas = 4.0f / 3.0f;
        const float rho = std::max(u.rho, 1e-6f);
        const float invRho = 1.0f / rho;
        const float vx = std::clamp(u.sx * invRho, -0.92f, 0.92f);
        const float vy = std::clamp(u.sy * invRho, -0.92f, 0.92f);
        const float b2 = u.bx * u.bx + u.by * u.by;
        const float kinetic = 0.5f * rho * (vx * vx + vy * vy);
        const float p = std::max((gammaGas - 1.0f) * (u.tau - kinetic - 0.5f * b2), grmhdPressureFloor_);
        const float pTot = p + 0.5f * b2;
        const float vDotB = vx * u.bx + vy * u.by;

        GrmhdCell f{};
        if (axis == 0) {
            f.rho = rho * vx;
            f.sx = u.sx * vx + pTot - u.bx * u.bx;
            f.sy = u.sy * vx - u.bx * u.by;
            f.tau = (u.tau + pTot) * vx - u.bx * vDotB;
            f.bx = 0.0f;
            f.by = u.by * vx - u.bx * vy;
        } else {
            f.rho = rho * vy;
            f.sx = u.sx * vy - u.by * u.bx;
            f.sy = u.sy * vy + pTot - u.by * u.by;
            f.tau = (u.tau + pTot) * vy - u.by * vDotB;
            f.bx = u.bx * vy - u.by * vx;
            f.by = 0.0f;
        }
        return f;
    }

    void resetGrmhdTorus() {
        const int n = grmhdGridSize_;
        const float domain = std::max(grmhdDomainHalf_, 8.0f);
        const float dx = (2.0f * domain) / static_cast<float>(n - 1);
        const float r0 = domain * 0.43f;
        const float widthR = domain * 0.16f;
        const float widthZ = domain * 0.085f;
        const float bh = std::max(grmhdHorizonRadius_, 0.2f);
        const float loop = std::max(grmhdMagneticLoop_, 0.0f);

        for (int iy = 0; iy < n; ++iy) {
            const float y = -domain + static_cast<float>(iy) * dx;
            for (int ix = 0; ix < n; ++ix) {
                const float x = -domain + static_cast<float>(ix) * dx;
                const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix);
                const float r = std::sqrt(x * x + y * y);
                const float torus = std::exp(-((r - r0) * (r - r0)) / (2.0f * widthR * widthR)
                                             -(y * y) / (2.0f * widthZ * widthZ));
                const float atmosphere = 0.0025f * std::exp(-0.12f * std::max(r - bh, 0.0f));
                const float rho = std::max(0.08f * torus + atmosphere, 1e-5f);
                const float kepler = std::sqrt(1.0f / std::max(r, bh + 0.6f));
                const float lapse = std::sqrt(std::max(1.0f - bh / std::max(r, bh + 0.02f), 0.08f));
                const float frameDrag = grmhdSpin_ * bh * bh / std::max(r * r * r, 1.0f);
                const float vphi = std::clamp(0.82f * kepler * lapse + frameDrag, -0.68f, 0.68f);
                const float invR = 1.0f / std::max(r, 1e-3f);
                const float pressure = std::max(0.055f * std::pow(rho, 4.0f / 3.0f), grmhdPressureFloor_);
                const float env = torus * loop;

                GrmhdCell c{};
                c.rho = rho;
                c.sx = -rho * vphi * y * invR;
                c.sy =  rho * vphi * x * invR;
                c.bx = -env * y / std::max(widthZ, 1e-3f);
                c.by =  env * (r - r0) / std::max(widthR, 1e-3f);
                const float v2 = (c.sx * c.sx + c.sy * c.sy) / std::max(rho * rho, 1e-8f);
                const float b2 = c.bx * c.bx + c.by * c.by;
                c.tau = pressure / (1.0f / 3.0f) + 0.5f * rho * v2 + 0.5f * b2;
                if (r < bh * 1.05f) {
                    c.rho = 1e-5f;
                    c.sx = c.sy = c.tau = c.bx = c.by = 0.0f;
                }
                grmhd_[idx] = c;
            }
        }

        grmhdNeedsReset_ = false;
        grmhdConfigDirty_ = false;
        grmhdTime_ = 0.0f;
    }

    void addGrmhdScaled(GrmhdCell& a, const GrmhdCell& b, float s) const {
        a.rho += b.rho * s;
        a.sx += b.sx * s;
        a.sy += b.sy * s;
        a.tau += b.tau * s;
        a.bx += b.bx * s;
        a.by += b.by * s;
    }

    void enforceGrmhdFloors(GrmhdCell& c) const {
        constexpr float gammaGas = 4.0f / 3.0f;
        c.rho = std::max(c.rho, 1e-5f);
        const float invRho = 1.0f / c.rho;
        float vx = std::clamp(c.sx * invRho, -0.92f, 0.92f);
        float vy = std::clamp(c.sy * invRho, -0.92f, 0.92f);
        c.sx = vx * c.rho;
        c.sy = vy * c.rho;
        const float b2 = c.bx * c.bx + c.by * c.by;
        const float kinetic = 0.5f * c.rho * (vx * vx + vy * vy);
        const float minTau = grmhdPressureFloor_ / (gammaGas - 1.0f) + kinetic + 0.5f * b2;
        c.tau = std::max(c.tau, minTau);
    }

    void stepGrmhdSimulation() {
        if (grmhdPaused_) {
            return;
        }
        if (grmhd_.empty() || grmhdNeedsReset_ || grmhdConfigDirty_) {
            resetGrmhdTorus();
        }

        const int n = grmhdGridSize_;
        const float domain = std::max(grmhdDomainHalf_, 8.0f);
        const float dx = (2.0f * domain) / static_cast<float>(n - 1);
        const float invDx = 1.0f / std::max(dx, 1e-6f);
        const float dt = std::clamp(grmhdDt_, 0.0001f, 0.04f);
        const int substeps = std::clamp(grmhdSubsteps_, 1, 16);
        const float bh = std::max(grmhdHorizonRadius_, 0.2f);
        const float diffusion = std::clamp(grmhdDiffusion_, 0.0f, 0.12f);

        auto cellAt = [&](int ix, int iy) -> const GrmhdCell& {
            ix = std::clamp(ix, 0, n - 1);
            iy = std::clamp(iy, 0, n - 1);
            return grmhd_[static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix)];
        };

        for (int step = 0; step < substeps; ++step) {
            grmhdNext_ = grmhd_;
            for (int iy = 1; iy < n - 1; ++iy) {
                const float y = -domain + static_cast<float>(iy) * dx;
                for (int ix = 1; ix < n - 1; ++ix) {
                    const float x = -domain + static_cast<float>(ix) * dx;
                    const size_t idx = static_cast<size_t>(iy) * static_cast<size_t>(n) + static_cast<size_t>(ix);

                    const GrmhdCell& c = cellAt(ix, iy);
                    GrmhdCell rhs{};
                    for (int axis = 0; axis < 2; ++axis) {
                        const GrmhdCell& lm = (axis == 0) ? cellAt(ix - 1, iy) : cellAt(ix, iy - 1);
                        const GrmhdCell& lp = c;
                        const GrmhdCell& rm = c;
                        const GrmhdCell& rp = (axis == 0) ? cellAt(ix + 1, iy) : cellAt(ix, iy + 1);
                        GrmhdCell fL = grmhdPrimitiveFlux(lp, axis);
                        GrmhdCell fLm = grmhdPrimitiveFlux(lm, axis);
                        GrmhdCell fR = grmhdPrimitiveFlux(rp, axis);
                        GrmhdCell fRm = grmhdPrimitiveFlux(rm, axis);
                        GrmhdCell fluxMinus = fL;
                        GrmhdCell fluxPlus = fR;
                        addGrmhdScaled(fluxMinus, fLm, 1.0f);
                        addGrmhdScaled(fluxMinus, lp, -0.95f);
                        addGrmhdScaled(fluxMinus, lm, 0.95f);
                        addGrmhdScaled(fluxPlus, fRm, 1.0f);
                        addGrmhdScaled(fluxPlus, rp, -0.95f);
                        addGrmhdScaled(fluxPlus, rm, 0.95f);
                        addGrmhdScaled(rhs, fluxPlus, -0.5f * invDx);
                        addGrmhdScaled(rhs, fluxMinus, 0.5f * invDx);
                    }

                    const float r2 = x * x + y * y + 0.35f * bh * bh;
                    const float r = std::sqrt(r2);
                    const float invR = 1.0f / std::max(r, 1e-3f);
                    const float grav = -0.75f / std::max(r2, 0.2f);
                    const float lapse = std::sqrt(std::max(1.0f - bh / std::max(r, bh + 0.02f), 0.08f));
                    const float drag = grmhdSpin_ * bh * bh / std::max(r2 * r, 1.0f);
                    const float ax = lapse * grav * x * invR - drag * c.sy / std::max(c.rho, 1e-5f);
                    const float ay = lapse * grav * y * invR + drag * c.sx / std::max(c.rho, 1e-5f);
                    rhs.sx += c.rho * ax;
                    rhs.sy += c.rho * ay;
                    rhs.tau += c.sx * ax + c.sy * ay;

                    GrmhdCell next = c;
                    addGrmhdScaled(next, rhs, dt);

                    const GrmhdCell& left = cellAt(ix - 1, iy);
                    const GrmhdCell& right = cellAt(ix + 1, iy);
                    const GrmhdCell& down = cellAt(ix, iy - 1);
                    const GrmhdCell& up = cellAt(ix, iy + 1);
                    addGrmhdScaled(next, left, diffusion * dt);
                    addGrmhdScaled(next, right, diffusion * dt);
                    addGrmhdScaled(next, down, diffusion * dt);
                    addGrmhdScaled(next, up, diffusion * dt);
                    addGrmhdScaled(next, c, -4.0f * diffusion * dt);

                    next.divb = ((right.bx - left.bx) + (up.by - down.by)) * 0.5f * invDx;
                    next.bx -= dt * 0.12f * next.divb;
                    next.by -= dt * 0.12f * next.divb;

                    const float edgeCells = static_cast<float>(std::min(std::min(ix, n - 1 - ix), std::min(iy, n - 1 - iy)));
                    const float edgeDamping = (edgeCells < 8.0f) ? std::exp(-0.045f * (8.0f - edgeCells)) : 1.0f;
                    if (r < bh * 1.12f) {
                        next.rho *= 0.38f;
                        next.sx *= 0.18f;
                        next.sy *= 0.18f;
                        next.tau *= 0.38f;
                        next.bx *= 0.25f;
                        next.by *= 0.25f;
                    } else {
                        next.rho *= edgeDamping;
                        next.sx *= edgeDamping;
                        next.sy *= edgeDamping;
                        next.tau *= edgeDamping;
                    }

                    enforceGrmhdFloors(next);
                    grmhdNext_[idx] = next;
                }
            }

            for (int i = 0; i < n; ++i) {
                grmhdNext_[static_cast<size_t>(i)] = grmhdNext_[static_cast<size_t>(n + i)];
                grmhdNext_[static_cast<size_t>(n - 1) * static_cast<size_t>(n) + static_cast<size_t>(i)] =
                    grmhdNext_[static_cast<size_t>(n - 2) * static_cast<size_t>(n) + static_cast<size_t>(i)];
                grmhdNext_[static_cast<size_t>(i) * static_cast<size_t>(n)] =
                    grmhdNext_[static_cast<size_t>(i) * static_cast<size_t>(n) + 1];
                grmhdNext_[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(n - 1)] =
                    grmhdNext_[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(n - 2)];
            }
            grmhd_.swap(grmhdNext_);
            grmhdTime_ += dt;
        }
    }

    void uploadGrmhdField() {
        if (!grmhdStorage_ || grmhd_.empty() || grmhdUpload_.empty()) {
            return;
        }
        float maxRho = 1e-6f;
        float maxDiv = 1e-6f;
        for (const GrmhdCell& c : grmhd_) {
            maxRho = std::max(maxRho, c.rho);
            maxDiv = std::max(maxDiv, std::abs(c.divb));
        }
        for (size_t i = 0; i < grmhd_.size(); ++i) {
            const GrmhdCell& c = grmhd_[i];
            const float rho = std::max(c.rho, 1e-8f);
            const float vx = c.sx / rho;
            const float vy = c.sy / rho;
            const float b2 = c.bx * c.bx + c.by * c.by;
            const float kinetic = 0.5f * rho * (vx * vx + vy * vy);
            const float p = std::max((1.0f / 3.0f) * (c.tau - kinetic - 0.5f * b2), grmhdPressureFloor_);
            const float sigma = b2 / rho;
            const float beta = p / std::max(0.5f * b2, 1e-6f);
            const int ix = static_cast<int>(i % static_cast<size_t>(grmhdGridSize_));
            const int iy = static_cast<int>(i / static_cast<size_t>(grmhdGridSize_));
            const float domain = std::max(grmhdDomainHalf_, 8.0f);
            const float dx = (2.0f * domain) / static_cast<float>(grmhdGridSize_ - 1);
            const float x = -domain + static_cast<float>(ix) * dx;
            const float y = -domain + static_cast<float>(iy) * dx;
            const float r = std::sqrt(x * x + y * y);
            const float radialSpeed = (r > 1e-4f) ? (vx * x + vy * y) / r : 0.0f;
            const size_t base = i * 8;
            grmhdUpload_[base + 0] = std::log1p(rho * grmhdColorScale_ * 16.0f) / std::log1p(maxRho * grmhdColorScale_ * 16.0f);
            grmhdUpload_[base + 1] = std::clamp(sigma * 4.0f, 0.0f, 1.0f);
            grmhdUpload_[base + 2] = std::clamp(std::log1p(beta) / 5.0f, 0.0f, 1.0f);
            grmhdUpload_[base + 3] = 1.0f;
            grmhdUpload_[base + 4] = std::clamp(0.5f + 0.5f * radialSpeed / 0.7f, 0.0f, 1.0f);
            grmhdUpload_[base + 5] = std::clamp(std::abs(c.divb) / maxDiv, 0.0f, 1.0f);
            grmhdUpload_[base + 6] = std::clamp(std::sqrt(vx * vx + vy * vy), 0.0f, 1.0f);
            grmhdUpload_[base + 7] = 0.0f;
        }
        pipelineGrmhd_->uniforms.updateStorageBuffer(grmhdStorage_, grmhdUpload_.data(), grmhdUpload_.size() * sizeof(float));
    }

    void renderGrmhd(float dt) {
        if (grmhd_.empty()) {
            resizeGrmhdBuffers();
        }
        stepGrmhdSimulation();
        uploadGrmhdField();

        int width = 1280;
        int height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);
        process2dNavigation(width, height, aspect, dt);

        gpu2dState_.orbital = glm::vec4(0.0f, 0.0f, 0.0f, static_cast<float>(grmhdViewMode_));
        gpu2dState_.tuning = glm::vec4(std::max(grmhdColorScale_, 0.001f), 1.0f, std::max(twoDZoom_, 1e-6f), grmhdSpin_);
        gpu2dState_.render = glm::vec4(grmhdTime_, aspect, 0.0f, 2.0f);
        gpu2dState_.pan = glm::vec4(twoDPan_.x, twoDPan_.y, grmhdHorizonRadius_, 0.0f);
        gpu2dState_.tdse = glm::vec4(static_cast<float>(grmhdGridSize_), std::max(grmhdDomainHalf_, 1.0f), grmhdMagneticLoop_, grmhdDiffusion_);

        writeRenderUniform(pipelineGrmhd_, reinterpret_cast<const float*>(&gpu2dState_));
        pipelineGrmhd_->setVertexBuffer(vbo2d_.get());
        pipelineGrmhd_->setIndexBuffer(ibo2d_.get());
        pipeline = pipelineGrmhd_;
    }

    int harmPitch() const {
        return harmGridSize_ + 2 * kHarmGhost;
    }

    size_t harmIndex(int ir, int ip) const {
        const int pitch = harmPitch();
        return static_cast<size_t>(ip + kHarmGhost) * static_cast<size_t>(pitch)
             + static_cast<size_t>(ir + kHarmGhost);
    }

    int harmThetaSize() const {
        return std::max(16, harmGridSize_ / 2);
    }

    int harmPhiSize() const {
        return harmGridSize_;
    }

    size_t harm3dCellCount() const {
        return static_cast<size_t>(harmGridSize_)
            * static_cast<size_t>(harmThetaSize())
            * static_cast<size_t>(harmPhiSize());
    }

    size_t harm3dIndex(int ir, int ith, int iph) const {
        const int n1 = harmGridSize_;
        const int n2 = harmThetaSize();
        const int n3 = harmPhiSize();
        const int rr = std::clamp(ir, 0, n1 - 1);
        const int tt = std::clamp(ith, 0, n2 - 1);
        int pp = iph % n3;
        if (pp < 0) pp += n3;
        return (static_cast<size_t>(pp) * static_cast<size_t>(n2) + static_cast<size_t>(tt))
             * static_cast<size_t>(n1) + static_cast<size_t>(rr);
    }

    static void invert4x4(const float in[4][4], float out[4][4]) {
        float a[4][8]{};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                a[i][j] = in[i][j];
            }
            a[i][i + 4] = 1.0f;
        }
        for (int c = 0; c < 4; ++c) {
            int pivot = c;
            float maxAbs = std::abs(a[c][c]);
            for (int r = c + 1; r < 4; ++r) {
                const float v = std::abs(a[r][c]);
                if (v > maxAbs) {
                    maxAbs = v;
                    pivot = r;
                }
            }
            if (pivot != c) {
                for (int j = 0; j < 8; ++j) {
                    std::swap(a[c][j], a[pivot][j]);
                }
            }
            const float invPivot = 1.0f / std::max(std::abs(a[c][c]), 1e-12f);
            const float sign = (a[c][c] < 0.0f) ? -1.0f : 1.0f;
            for (int j = 0; j < 8; ++j) {
                a[c][j] *= invPivot * sign;
            }
            for (int r = 0; r < 4; ++r) {
                if (r == c) continue;
                const float f = a[r][c];
                for (int j = 0; j < 8; ++j) {
                    a[r][j] -= f * a[c][j];
                }
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                out[i][j] = a[i][j + 4];
            }
        }
    }

    float harmKerrMetricUt(float rIn, float thetaIn, float ell) const {
        const float a = std::clamp(harmSpin_, -0.98f, 0.98f);
        const float r = std::max(rIn, 1.0f + std::sqrt(std::max(1.0f - a * a, 0.0f)) + 0.02f);
        const float th = std::clamp(thetaIn, 0.02f, kPi - 0.02f);
        const float sinTh = std::max(std::sin(th), 0.02f);
        const float cosTh = std::cos(th);
        const float sigma = r * r + a * a * cosTh * cosTh;
        const float gtt = -(1.0f - 2.0f * r / sigma);
        const float gtph = -2.0f * a * r * sinTh * sinTh / sigma;
        const float gpp = (r * r + a * a + 2.0f * a * a * r * sinTh * sinTh / sigma) * sinTh * sinTh;
        const float numer = std::max(gtph * gtph - gtt * gpp, 1e-10f);
        const float denom = std::max(gpp + 2.0f * ell * gtph + ell * ell * gtt, 1e-10f);
        return -std::sqrt(numer / denom);
    }

    float harmKeplerianEll(float rIn) const {
        const float a = std::clamp(harmSpin_, -0.98f, 0.98f);
        const float r = std::max(rIn, 2.2f);
        const float sr = std::sqrt(r);
        const float denom = std::max(sr * (r - 2.0f) + a, 1e-4f);
        return (r * r - 2.0f * a * sr + a * a) / denom;
    }

    HarmGeom harmMetricAt(float radius, float phi) const {
        HarmGeom g{};
        g.r = radius;
        g.phi = phi;
        const float a = std::clamp(harmSpin_, -0.98f, 0.98f);
        const float r = std::max(radius, 1.0f + std::sqrt(std::max(1.0f - a * a, 0.0f)) + 0.02f);
        const float r2 = r * r;
        const float twoOverR = 2.0f / r;
        g.gcov[0][0] = -1.0f + twoOverR;
        g.gcov[0][1] = twoOverR;
        g.gcov[1][0] = g.gcov[0][1];
        g.gcov[0][3] = -2.0f * a / r;
        g.gcov[3][0] = g.gcov[0][3];
        g.gcov[1][1] = 1.0f + twoOverR;
        g.gcov[1][3] = -a * (1.0f + twoOverR);
        g.gcov[3][1] = g.gcov[1][3];
        g.gcov[2][2] = r2;
        g.gcov[3][3] = r2 + a * a + 2.0f * a * a / r;
        invert4x4(g.gcov, g.gcon);
        g.alpha = 1.0f / std::sqrt(std::max(-g.gcon[0][0], 1e-8f));
        g.betaPhi = g.alpha * g.alpha * g.gcon[0][3];
        g.sqrtg = r2;
        return g;
    }

    HarmGeom harmGeom(int ir, int ip) const {
        const int n = harmGridSize_;
        const float rin = std::max(harmRin_, 1.05f);
        const float rout = std::max(harmRout_, rin + 4.0f);
        const float x = (static_cast<float>(ir) + 0.5f) / static_cast<float>(n);
        const float logr = std::log(rin) + x * (std::log(rout) - std::log(rin));
        const float r = std::exp(logr);
        const float phi = (static_cast<float>(ip) + 0.5f) * (2.0f * kPi / static_cast<float>(n));
        return harmMetricAt(r, phi);
    }

    void resizeHarmBuffers() {
        harmGridSize_ = std::clamp(harmGridSize_, 32, kMaxHarmGrid);
        const size_t total = static_cast<size_t>(harmPitch()) * static_cast<size_t>(harmPitch());
        harmP_.assign(total, HarmPrim{});
        harmPNext_.assign(total, HarmPrim{});
        harmU_.assign(total, HarmCons{});
        harmUNext_.assign(total, HarmCons{});
        harmUpload_.assign(harm3dCellCount() * 12, 0.0f);
        harmNeedsReset_ = true;
        harmGpuNeedsUpload_ = true;
    }

    void harmApplyFloors(HarmPrim& p) const {
        p.rho = std::max(p.rho, harmRhoFloor_);
        p.u = std::max(p.u, harmUFloor_);
        const float v2 = p.ur * p.ur + p.uphi * p.uphi;
        if (v2 > 0.92f * 0.92f) {
            const float s = 0.92f / std::sqrt(v2);
            p.ur *= s;
            p.uphi *= s;
            p.fail = 1.0f;
        }
    }

    HarmState harmMakeState(const HarmPrim& pin, const HarmGeom& g) const {
        HarmPrim p = pin;
        harmApplyFloors(p);
        HarmState s{};
        const float q1 = std::clamp(p.ur, -8.0f, 8.0f);
        const float q3 = std::clamp(p.uphi, -8.0f / std::max(g.r, 1.0f), 8.0f / std::max(g.r, 1.0f));
        const float qsq = std::max(
            g.gcov[1][1] * q1 * q1
            + 2.0f * g.gcov[1][3] * q1 * q3
            + g.gcov[3][3] * q3 * q3,
            0.0f);
        const float gamma = std::sqrt(1.0f + qsq);
        s.ucon[0] = gamma / std::max(g.alpha, 1e-8f);
        s.ucon[1] = q1 - gamma * g.alpha * g.gcon[0][1];
        s.ucon[2] = 0.0f;
        s.ucon[3] = q3 - gamma * g.alpha * g.gcon[0][3];
        for (int mu = 0; mu < 4; ++mu) {
            s.ucov[mu] = 0.0f;
            for (int nu = 0; nu < 4; ++nu) {
                s.ucov[mu] += g.gcov[mu][nu] * s.ucon[nu];
            }
        }

        const float bconSpatial[4] = {0.0f, p.br, 0.0f, p.bphi};
        s.bcon[0] = bconSpatial[1] * s.ucov[1] + bconSpatial[3] * s.ucov[3];
        s.bcon[1] = (bconSpatial[1] + s.bcon[0] * s.ucon[1]) / std::max(s.ucon[0], 1e-8f);
        s.bcon[2] = 0.0f;
        s.bcon[3] = (bconSpatial[3] + s.bcon[0] * s.ucon[3]) / std::max(s.ucon[0], 1e-8f);
        s.bsq = 0.0f;
        for (int mu = 0; mu < 4; ++mu) {
            s.bcov[mu] = 0.0f;
            for (int nu = 0; nu < 4; ++nu) {
                s.bcov[mu] += g.gcov[mu][nu] * s.bcon[nu];
            }
            s.bsq += s.bcon[mu] * s.bcov[mu];
        }
        s.bsq = std::max(s.bsq, 0.0f);
        return s;
    }

    float harmTmunu(const HarmPrim& p, const HarmGeom& g, const HarmState& s, int mu, int nuCov) const {
        constexpr float gammaGas = 4.0f / 3.0f;
        const float pg = (gammaGas - 1.0f) * std::max(p.u, harmUFloor_);
        const float eta = (mu == nuCov) ? 1.0f : 0.0f;
        const float wtot = std::max(p.rho, harmRhoFloor_) + std::max(p.u, harmUFloor_) + pg + s.bsq;
        return wtot * s.ucon[mu] * s.ucov[nuCov]
             + (pg + 0.5f * s.bsq) * eta
             - s.bcon[mu] * s.bcov[nuCov];
    }

    float harmTcon(const HarmPrim& p, const HarmGeom& g, const HarmState& s, int mu, int nu) const {
        constexpr float gammaGas = 4.0f / 3.0f;
        const float pg = (gammaGas - 1.0f) * std::max(p.u, harmUFloor_);
        const float wtot = std::max(p.rho, harmRhoFloor_) + std::max(p.u, harmUFloor_) + pg + s.bsq;
        return wtot * s.ucon[mu] * s.ucon[nu]
             + (pg + 0.5f * s.bsq) * g.gcon[mu][nu]
             - s.bcon[mu] * s.bcon[nu];
    }

    float harmMetricSourceR(const HarmPrim& p, const HarmGeom& g) const {
        const HarmState s = harmMakeState(p, g);
        const float dr = std::max(g.r * 1e-3f, 1e-4f);
        const HarmGeom gp = harmMetricAt(g.r + dr, g.phi);
        const HarmGeom gm = harmMetricAt(std::max(g.r - dr, harmRin_ * 0.9f), g.phi);
        float source = 0.0f;
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                const float dg = (gp.gcov[mu][nu] - gm.gcov[mu][nu]) / (gp.r - gm.r);
                source += harmTcon(p, g, s, mu, nu) * dg;
            }
        }
        return 0.5f * g.sqrtg * source;
    }

    HarmCons harmPrimToCons(const HarmPrim& pin, const HarmGeom& g) const {
        HarmPrim p = pin;
        harmApplyFloors(p);
        const HarmState s = harmMakeState(p, g);
        HarmCons u{};
        u.d = g.sqrtg * p.rho * s.ucon[0];
        u.tau = g.sqrtg * (harmTmunu(p, g, s, 0, 0) + p.rho * s.ucon[0]);
        u.sr = g.sqrtg * harmTmunu(p, g, s, 0, 1);
        u.sphi = g.sqrtg * harmTmunu(p, g, s, 0, 3);
        u.br = g.sqrtg * p.br;
        u.bphi = g.sqrtg * p.bphi;
        u.fail = p.fail;
        return u;
    }

    HarmPrim harmConsToPrim(const HarmCons& u, const HarmGeom& g) const {
        constexpr float gammaGas = 4.0f / 3.0f;
        HarmPrim p{};
        const float gdet = std::max(g.sqrtg, 1e-8f);
        const float lapse = std::max(g.alpha, 1e-8f);
        p.br = u.br / gdet;
        p.bphi = u.bphi / gdet;
        if (u.d <= 0.0f) {
            p.rho = harmRhoFloor_;
            p.u = harmUFloor_;
            p.fail = 1.0f;
            return p;
        }

        float Bcon[4] = {0.0f, u.br * lapse / gdet, 0.0f, u.bphi * lapse / gdet};
        float Bcov[4]{};
        float Qcov[4] = {
            (u.tau - u.d) * lapse / gdet,
            u.sr * lapse / gdet,
            0.0f,
            u.sphi * lapse / gdet
        };
        float Qcon[4]{};
        float ncon[4]{};
        float ncov[4] = {-lapse, 0.0f, 0.0f, 0.0f};
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                Bcov[mu] += g.gcov[mu][nu] * Bcon[nu];
                Qcon[mu] += g.gcon[mu][nu] * Qcov[nu];
                ncon[mu] += g.gcon[mu][nu] * ncov[nu];
            }
        }
        auto dot4 = [](const float a[4], const float b[4]) {
            float out = 0.0f;
            for (int i = 0; i < 4; ++i) out += a[i] * b[i];
            return out;
        };
        const float D = std::max(u.d * lapse / gdet, harmRhoFloor_);
        const float Bsq = std::max(dot4(Bcon, Bcov), 0.0f);
        const float QdB = dot4(Bcon, Qcov);
        const float Qdotn = dot4(Qcon, ncov);
        const float Qsq = dot4(Qcon, Qcov);
        float Qtcon[4]{};
        for (int mu = 0; mu < 4; ++mu) {
            Qtcon[mu] = Qcon[mu] + ncon[mu] * Qdotn;
        }
        const float Qtsq = std::max(Qsq + Qdotn * Qdotn, 0.0f);
        const float Ep = -Qdotn - D;

        auto pressureRhoW = [&](float rho0, float w) {
            return (gammaGas - 1.0f) * (w - rho0) / gammaGas;
        };
        auto gammaFromWp = [&](float Wp) {
            const float QdBsq = QdB * QdB;
            const float W = D + Wp;
            const float W2 = W * W;
            const float WB = W + Bsq;
            const float denom = QdBsq * (W + WB) + W2 * (Qtsq - WB * WB);
            const float utsq = -((W + WB) * QdBsq + W2 * Qtsq) / ((std::abs(denom) > 1e-12f) ? denom : -1e-12f);
            return std::sqrt(1.0f + std::abs(utsq));
        };
        auto errEqn = [&](float Wp) {
            const float W = D + Wp;
            const float gamma = gammaFromWp(Wp);
            const float w = W / std::max(gamma * gamma, 1e-8f);
            const float rho0 = D / std::max(gamma, 1e-8f);
            const float pres = pressureRhoW(rho0, w);
            const float WB = Bsq + W;
            return -Ep + Wp - pres + 0.5f * Bsq
                 + 0.5f * (Bsq * Qtsq - QdB * QdB) / std::max(WB * WB, 1e-10f);
        };

        float Wp = std::max(Ep - 0.5f * Bsq, harmUFloor_ + D * 0.05f);
        float err = errEqn(Wp);
        float Wprev = Wp * 0.95f;
        float eprev = errEqn(Wprev);
        bool converged = false;
        for (int it = 0; it < 16; ++it) {
            const float denom = err - eprev;
            float dW = (std::abs(denom) > 1e-12f) ? ((Wprev - Wp) * err / denom) : (-0.25f * Wp);
            dW = std::clamp(dW, -0.5f * Wp, 2.0f * Wp);
            Wprev = Wp;
            eprev = err;
            Wp = std::max(Wp + dW, harmUFloor_);
            err = errEqn(Wp);
            if (std::abs(dW / std::max(Wp, 1e-8f)) < 1e-6f || std::abs(err / std::max(Wp, 1e-8f)) < 1e-6f) {
                converged = true;
                break;
            }
        }

        const float gamma = gammaFromWp(Wp);
        const float W = Wp + D;
        const float rho0 = D / std::max(gamma, 1e-8f);
        const float w = W / std::max(gamma * gamma, 1e-8f);
        const float pres = pressureRhoW(rho0, w);
        p.rho = std::max(rho0, harmRhoFloor_);
        p.u = std::max(w - (rho0 + pres), harmUFloor_);
        p.ur = (gamma / std::max(W + Bsq, 1e-8f)) * (Qtcon[1] + QdB * Bcon[1] / std::max(W, 1e-8f));
        p.uphi = (gamma / std::max(W + Bsq, 1e-8f)) * (Qtcon[3] + QdB * Bcon[3] / std::max(W, 1e-8f));
        if (!converged || !std::isfinite(p.rho) || !std::isfinite(p.u) || !std::isfinite(p.ur) || !std::isfinite(p.uphi)) {
            p.rho = std::max(D, harmRhoFloor_);
            p.u = harmUFloor_;
            p.ur = 0.0f;
            p.uphi = 0.0f;
            p.fail = 1.0f;
        }
        harmApplyFloors(p);
        return p;
    }

    HarmCons harmFlux(const HarmPrim& pin, const HarmGeom& g, int dir) const {
        HarmPrim p = pin;
        harmApplyFloors(p);
        const HarmState s = harmMakeState(p, g);
        const int mu = (dir == 0) ? 1 : 3;
        const float vdir = s.ucon[mu] / std::max(s.ucon[0], 1e-8f);
        const float vr = s.ucon[1] / std::max(s.ucon[0], 1e-8f);
        const float vp = s.ucon[3] / std::max(s.ucon[0], 1e-8f);
        HarmCons f{};
        f.d = g.sqrtg * p.rho * s.ucon[mu];
        f.tau = g.sqrtg * (harmTmunu(p, g, s, mu, 0) + p.rho * s.ucon[mu]);
        f.sr = g.sqrtg * harmTmunu(p, g, s, mu, 1);
        f.sphi = g.sqrtg * harmTmunu(p, g, s, mu, 3);
        f.br = (dir == 0) ? 0.0f : g.sqrtg * (p.br * vp - p.bphi * vr);
        f.bphi = (dir == 1) ? 0.0f : g.sqrtg * (p.bphi * vr - p.br * vp);
        return f;
    }

    float harmFastSpeed(const HarmPrim& p) const {
        constexpr float gammaGas = 4.0f / 3.0f;
        const float pg = (gammaGas - 1.0f) * std::max(p.u, harmUFloor_);
        const float b2 = p.br * p.br + p.bphi * p.bphi;
        const float h = std::max(p.rho + p.u + pg + b2, 1e-6f);
        return std::sqrt(std::clamp((gammaGas * pg + b2) / h, 0.02f, 0.92f));
    }

    void harmAddScaled(HarmCons& a, const HarmCons& b, float s) const {
        a.d += b.d * s;
        a.sr += b.sr * s;
        a.sphi += b.sphi * s;
        a.tau += b.tau * s;
        a.br += b.br * s;
        a.bphi += b.bphi * s;
        a.fail += b.fail * s;
        a.divb += b.divb * s;
    }

    HarmCons harmHllFlux(const HarmPrim& l, const HarmPrim& r, const HarmGeom& g, int dir) const {
        const HarmCons ul = harmPrimToCons(l, g);
        const HarmCons ur = harmPrimToCons(r, g);
        const HarmCons fl = harmFlux(l, g, dir);
        const HarmCons fr = harmFlux(r, g, dir);
        const float vl = (dir == 0) ? l.ur : l.uphi;
        const float vr = (dir == 0) ? r.ur : r.uphi;
        const float cl = harmFastSpeed(l);
        const float cr = harmFastSpeed(r);
        const float cmin = std::min(0.0f, std::min(vl - cl, vr - cr));
        const float cmax = std::max(0.0f, std::max(vl + cl, vr + cr));
        if (cmax <= 0.0f) return fr;
        if (cmin >= 0.0f) return fl;
        HarmCons out{
            (cmax * fl.d - cmin * fr.d + cmax * cmin * (ur.d - ul.d)) / (cmax - cmin),
            (cmax * fl.sr - cmin * fr.sr + cmax * cmin * (ur.sr - ul.sr)) / (cmax - cmin),
            (cmax * fl.sphi - cmin * fr.sphi + cmax * cmin * (ur.sphi - ul.sphi)) / (cmax - cmin),
            (cmax * fl.tau - cmin * fr.tau + cmax * cmin * (ur.tau - ul.tau)) / (cmax - cmin),
            (cmax * fl.br - cmin * fr.br + cmax * cmin * (ur.br - ul.br)) / (cmax - cmin),
            (cmax * fl.bphi - cmin * fr.bphi + cmax * cmin * (ur.bphi - ul.bphi)) / (cmax - cmin),
            0.0f,
            0.0f
        };
        return out;
    }

    void harmFillGhosts() {
        const int n = harmGridSize_;
        for (int ip = 0; ip < n; ++ip) {
            for (int g = 1; g <= kHarmGhost; ++g) {
                harmP_[harmIndex(-g, ip)] = harmP_[harmIndex(0, ip)];
                harmP_[harmIndex(n - 1 + g, ip)] = harmP_[harmIndex(n - 1, ip)];
                harmP_[harmIndex(-g, ip)].ur = std::min(harmP_[harmIndex(-g, ip)].ur, 0.0f);
            }
        }
        for (int ir = -kHarmGhost; ir < n + kHarmGhost; ++ir) {
            for (int g = 1; g <= kHarmGhost; ++g) {
                harmP_[harmIndex(ir, -g)] = harmP_[harmIndex(ir, n - g)];
                harmP_[harmIndex(ir, n - 1 + g)] = harmP_[harmIndex(ir, g - 1)];
            }
        }
    }

    void resetHarmTorus() {
        const int n = harmGridSize_;
        const float rin = std::max(harmRin_, 1.05f);
        const float rout = std::max(harmRout_, rin + 4.0f);
        const float r0 = 0.34f * rout;
        const float sigma = 0.12f * rout;
        for (int ip = 0; ip < n; ++ip) {
            for (int ir = 0; ir < n; ++ir) {
                const HarmGeom g = harmGeom(ir, ip);
                const float torus = std::exp(-((g.r - r0) * (g.r - r0)) / std::max(2.0f * sigma * sigma, 1e-6f));
                const float rho = std::max(0.24f * torus + 1e-4f * std::pow(g.r / rin, -1.5f), harmRhoFloor_);
                const float vk = std::sqrt(1.0f / std::max(g.r, rin));
                const float uphi = std::clamp(0.74f * vk / (1.0f + harmSpin_ / std::pow(std::max(g.r, 1.0f), 1.5f)), -0.78f, 0.78f);
                const float pressure = 0.035f * std::pow(rho, 4.0f / 3.0f);
                const float loop = harmMagneticLoop_ * torus;
                const float logr = std::log(std::max(g.r, 1.0f));
                const float arm2 = std::sin(2.0f * g.phi - 3.6f * logr);
                const float arm3 = std::sin(3.0f * g.phi - 5.4f * logr + 0.7f);
                const float arm5 = std::sin(5.0f * g.phi + 1.7f * logr);
                const float perturb = std::clamp(1.0f + 0.10f * arm2 + 0.075f * arm3 + 0.035f * arm5, 0.72f, 1.35f);
                HarmPrim p{};
                p.rho = std::max(rho * perturb, harmRhoFloor_);
                p.u = std::max(pressure * perturb / (1.0f / 3.0f), harmUFloor_);
                p.ur = -0.002f * std::exp(-g.r / std::max(rout, 1.0f));
                p.uphi = uphi;
                p.br = loop * (std::sin(g.phi + 0.7f * logr) + 0.38f * std::sin(3.0f * g.phi - 2.0f * logr)) / std::max(g.r, 1.0f);
                p.bphi = loop * torus * (0.32f + 0.22f * arm2 + 0.14f * std::cos(4.0f * g.phi - 3.0f * logr));
                harmApplyFloors(p);
                harmP_[harmIndex(ir, ip)] = p;
                harmU_[harmIndex(ir, ip)] = harmPrimToCons(p, g);
            }
        }
        harmFillGhosts();
        harmTime_ = 0.0f;
        harmNeedsReset_ = false;
        harmGpuNeedsUpload_ = true;
    }

    void stepHarmSimulation() {
        if (harmPaused_) return;
        if (harmP_.empty() || harmNeedsReset_) {
            resetHarmTorus();
        }
        const int n = harmGridSize_;
        const float rin = std::max(harmRin_, 1.05f);
        const float rout = std::max(harmRout_, rin + 4.0f);
        const float dx1 = (std::log(rout) - std::log(rin)) / static_cast<float>(n);
        const float dx2 = 2.0f * kPi / static_cast<float>(n);
        const float dt = std::clamp(harmDt_, 0.00005f, 0.03f);
        const int substeps = std::clamp(harmSubsteps_, 1, 12);
        for (int step = 0; step < substeps; ++step) {
            harmFillGhosts();
            for (int ip = 0; ip < n; ++ip) {
                for (int ir = 0; ir < n; ++ir) {
                    const size_t idx = harmIndex(ir, ip);
                    const HarmGeom g = harmGeom(ir, ip);
                    HarmCons u = harmPrimToCons(harmP_[idx], g);
                    const HarmCons frp = harmHllFlux(harmP_[harmIndex(ir, ip)], harmP_[harmIndex(ir + 1, ip)], g, 0);
                    const HarmCons frm = harmHllFlux(harmP_[harmIndex(ir - 1, ip)], harmP_[harmIndex(ir, ip)], g, 0);
                    const HarmCons fpp = harmHllFlux(harmP_[harmIndex(ir, ip)], harmP_[harmIndex(ir, ip + 1)], g, 1);
                    const HarmCons fpm = harmHllFlux(harmP_[harmIndex(ir, ip - 1)], harmP_[harmIndex(ir, ip)], g, 1);
                    harmAddScaled(u, frp, -dt / std::max(dx1 * g.r, 1e-5f));
                    harmAddScaled(u, frm,  dt / std::max(dx1 * g.r, 1e-5f));
                    harmAddScaled(u, fpp, -dt / std::max(dx2 * g.r, 1e-5f));
                    harmAddScaled(u, fpm,  dt / std::max(dx2 * g.r, 1e-5f));

                    const HarmPrim& p = harmP_[idx];
                    u.sr += dt * harmMetricSourceR(p, g);
                    if (g.r < rin * 1.08f) {
                        u.d *= 0.78f;
                        u.sr *= 0.55f;
                        u.sphi *= 0.75f;
                        u.tau *= 0.78f;
                        u.br *= 0.65f;
                        u.bphi *= 0.65f;
                    }
                    HarmPrim pn = harmConsToPrim(u, g);
                    if (!std::isfinite(pn.rho) || !std::isfinite(pn.u)) {
                        pn = harmP_[idx];
                        pn.fail = 1.0f;
                    }
                    harmPNext_[idx] = pn;
                    harmUNext_[idx] = harmPrimToCons(pn, g);
                }
            }
            harmP_.swap(harmPNext_);
            harmU_.swap(harmUNext_);
            harmTime_ += dt;
        }
    }

    void uploadHarmField() {
        if (!harmStorage_ || harmP_.empty() || harmUpload_.empty()) return;
        packHarmPrimitivesForGpu();
        pipelineHarmGrmhd_->uniforms.updateStorageBuffer(harmStorage_, harmUpload_.data(), harmUpload_.size() * sizeof(float));
    }

    void packHarmPrimitivesForGpu() {
        const int n1 = harmGridSize_;
        const int n2 = harmThetaSize();
        const int n3 = harmPhiSize();
        const size_t cells = harm3dCellCount();
        if (harmUpload_.size() != harm3dCellCount() * 12) {
            harmUpload_.assign(harm3dCellCount() * 12, 0.0f);
        }

        const float rin = std::max(harmRin_, 1.05f);
        const float rout = std::max(harmRout_, rin + 4.0f);
        const bool mad = (harmInitMode_ == 1);
        const float rIn = mad ? 5.8f : 7.2f;
        const float rMax = mad ? 11.5f : 15.0f;
        const float ell = harmKeplerianEll(rMax);
        const float utIn = -harmKerrMetricUt(rIn, 0.5f * kPi, ell);
        const float polytropeK = mad ? 0.010f : 0.012f;
        const float hOverR = mad ? 0.38f : 0.32f;
        const float logRange = std::max(std::log(rout) - std::log(rin), 1e-6f);
        const float dtheta = 0.84f * kPi / static_cast<float>(std::max(n2, 1));
        const float dphi = 2.0f * kPi / static_cast<float>(std::max(n3, 1));
        std::vector<float> rhoField(cells, harmRhoFloor_);
        std::vector<float> uField(cells, harmUFloor_);
        std::vector<float> pressureField(cells, harmUFloor_ / 3.0f);
        std::vector<float> vrField(cells, 0.0f);
        std::vector<float> vthField(cells, 0.0f);
        std::vector<float> vphField(cells, 0.0f);
        std::vector<float> aPhi(cells, 0.0f);

        float rhoMax = harmRhoFloor_;
        float pressureMax = harmUFloor_ / 3.0f;
        for (int ip = 0; ip < n3; ++ip) {
            const float phi = (static_cast<float>(ip) + 0.5f) * (2.0f * kPi / static_cast<float>(n3));
            for (int it = 0; it < n2; ++it) {
                const float y = (static_cast<float>(it) + 0.5f) / static_cast<float>(n2);
                const float theta = 0.08f * kPi + y * 0.84f * kPi;
                const float z = std::cos(theta) / std::max(std::sin(theta), 0.08f);
                const float vertical = std::exp(-(z * z) / std::max(2.0f * hOverR * hOverR, 1e-5f));
                for (int ir = 0; ir < n1; ++ir) {
                    const float x = (static_cast<float>(ir) + 0.5f) / static_cast<float>(n1);
                    const float r = std::exp(std::log(rin) + x * logRange);
                    const float ut = -harmKerrMetricUt(r, theta, ell);
                    const float potential = std::log(std::max(utIn, 1e-8f)) - std::log(std::max(ut, 1e-8f));
                    const float hMinusOne = std::max(std::exp(std::clamp(potential, -20.0f, 20.0f)) - 1.0f, 0.0f);
                    const float fmEnvelope = std::pow(std::max(hMinusOne, 0.0f), 1.5f);
                    const float verticalLimiter = std::exp(-std::pow(std::abs(z) / std::max(hOverR, 1e-4f), 4.0f));
                    const float torus = fmEnvelope * verticalLimiter;
                    const float logr = std::log(std::max(r, 1.0f));
                    const float arm2 = std::sin(2.0f * phi - 3.6f * logr + 1.4f * z);
                    const float arm3 = std::sin(3.0f * phi - 5.4f * logr + 0.7f - 0.8f * z);
                    const float arm5 = std::sin(5.0f * phi + 1.7f * logr + 0.35f * static_cast<float>(it));
                    const float perturb = std::clamp(1.0f + 0.075f * arm2 + 0.045f * arm3 + 0.025f * arm5, 0.72f, 1.28f);
                    const float atmosphere = 1e-5f * std::pow(std::max(r / rin, 1.0f), -1.5f);
                    const float rhoTorus = std::pow(std::max(hMinusOne * 0.25f / (4.0f * polytropeK / 3.0f), 0.0f), 3.0f);
                    const float rho = std::max((mad ? 1.20f : 1.0f) * rhoTorus * verticalLimiter * perturb + atmosphere, harmRhoFloor_);
                    const float pressure = polytropeK * std::pow(std::max(rho - atmosphere, 0.0f), 4.0f / 3.0f) + harmUFloor_ / 3.0f;
                    const float omegaK = 1.0f / (std::pow(std::max(r, 1.0f), 1.5f) + harmSpin_);
                    const float mriSeed = std::sin(11.0f * phi + 3.0f * std::log(std::max(r, 1.0f)) + 7.0f * theta)
                        * std::sin(5.0f * phi - 2.0f * theta);
                    const float vr = -(mad ? 0.0060f : 0.0025f) * std::exp(-r / std::max(rout, 1.0f))
                        - 0.002f * torus * std::max(arm2, 0.0f)
                        + 0.0045f * torus * mriSeed;
                    const float vth = 0.006f * vertical * std::sin(theta - 0.5f * kPi) * std::sin(2.0f * phi - 2.0f * logr)
                        + 0.0035f * torus * std::cos(7.0f * phi + 4.0f * theta);
                    const float vphi = (mad ? 0.70f : 0.76f) * omegaK * (1.0f + 0.035f * arm2 + 0.010f * mriSeed);
                    const size_t idx = harm3dIndex(ir, it, ip);
                    rhoField[idx] = rho;
                    pressureField[idx] = pressure;
                    uField[idx] = std::max(pressure / (1.0f / 3.0f), harmUFloor_);
                    vrField[idx] = vr;
                    vthField[idx] = vth;
                    vphField[idx] = vphi;
                    rhoMax = std::max(rhoMax, rho);
                    pressureMax = std::max(pressureMax, pressure);
                }
            }
        }

        for (int ip = 0; ip < n3; ++ip) {
            for (int it = 0; it < n2; ++it) {
                const float y = (static_cast<float>(it) + 0.5f) / static_cast<float>(n2);
                const float theta = 0.08f * kPi + y * 0.84f * kPi;
                const float sinTh = std::max(std::sin(theta), 0.08f);
                for (int ir = 0; ir < n1; ++ir) {
                    const float x = (static_cast<float>(ir) + 0.5f) / static_cast<float>(n1);
                    const float r = std::exp(std::log(rin) + x * logRange);
                    const float phi = (static_cast<float>(ip) + 0.5f) * (2.0f * kPi / static_cast<float>(n3));
                    const size_t idx = harm3dIndex(ir, it, ip);
                    const float rhoNorm = rhoField[idx] / std::max(rhoMax, harmRhoFloor_);
                    const float cutoff = mad ? 0.025f : 0.16f;
                    const float core = std::max(rhoNorm - cutoff, 0.0f);
                    if (mad) {
                        // One coherent polarity gives large net horizon-threading flux after inflow.
                        aPhi[idx] = core * r * r * sinTh * sinTh;
                    } else {
                        // Alternating loops keep the net flux small: the SANE topology.
                        const float loopPhase = 3.0f * kPi * (r - rin) / std::max(rout - rin, 1e-4f);
                        const float wobble = 1.0f + 0.08f * std::sin(2.0f * phi + 0.7f * static_cast<float>(it));
                        aPhi[idx] = core * std::sin(loopPhase) * r * sinTh * wobble;
                    }
                }
            }
        }

        std::vector<float> brField(cells, 0.0f);
        std::vector<float> bthField(cells, 0.0f);
        float b2Max = 1e-20f;
        for (int ip = 0; ip < n3; ++ip) {
            for (int it = 0; it < n2; ++it) {
                const float y = (static_cast<float>(it) + 0.5f) / static_cast<float>(n2);
                const float theta = 0.08f * kPi + y * 0.84f * kPi;
                const float sinTh = std::max(std::sin(theta), 0.08f);
                for (int ir = 0; ir < n1; ++ir) {
                    const float x = (static_cast<float>(ir) + 0.5f) / static_cast<float>(n1);
                    const float r = std::exp(std::log(rin) + x * logRange);
                    const float dr = std::max(r * logRange / static_cast<float>(std::max(n1, 1)), 1e-4f);
                    const float apTp = aPhi[harm3dIndex(ir, std::min(it + 1, n2 - 1), ip)];
                    const float apTm = aPhi[harm3dIndex(ir, std::max(it - 1, 0), ip)];
                    const float apRp = aPhi[harm3dIndex(std::min(ir + 1, n1 - 1), it, ip)];
                    const float apRm = aPhi[harm3dIndex(std::max(ir - 1, 0), it, ip)];
                    const float dATh = (apTp - apTm) / ((it == 0 || it == n2 - 1) ? dtheta : 2.0f * dtheta);
                    const float dAR = (apRp - apRm) / ((ir == 0 || ir == n1 - 1) ? dr : 2.0f * dr);
                    const size_t idx = harm3dIndex(ir, it, ip);
                    brField[idx] = dATh / std::max(r * r * sinTh, 1e-5f);
                    bthField[idx] = -dAR / std::max(r * sinTh, 1e-5f);
                    b2Max = std::max(b2Max, brField[idx] * brField[idx] + bthField[idx] * bthField[idx]);
                }
            }
        }

        const float requested = std::clamp(harmMagneticLoop_ / 0.055f, 0.15f, 4.0f);
        const float targetBeta = (mad ? 8.0f : 85.0f) / requested;
        const float bScale = std::sqrt(std::max(2.0f * pressureMax / std::max(targetBeta * b2Max, 1e-20f), 0.0f));
        float betaSum = 0.0f;
        float betaMin = std::numeric_limits<float>::max();
        float sigmaMax = 0.0f;
        float fluxBH = 0.0f;
        float mdot = 0.0f;
        int betaCount = 0;
        const int fluxIr = std::min(2, n1 - 1);
        for (int ip = 0; ip < n3; ++ip) {
            const float phi = (static_cast<float>(ip) + 0.5f) * dphi;
            for (int it = 0; it < n2; ++it) {
                const float y = (static_cast<float>(it) + 0.5f) / static_cast<float>(n2);
                const float theta = 0.08f * kPi + y * 0.84f * kPi;
                const float sinTh = std::max(std::sin(theta), 0.08f);
                for (int ir = 0; ir < n1; ++ir) {
                    const float x = (static_cast<float>(ir) + 0.5f) / static_cast<float>(n1);
                    const float r = std::exp(std::log(rin) + x * logRange);
                    const float logr = std::log(std::max(r, 1.0f));
                    const float arm2 = std::sin(2.0f * phi - 3.6f * logr);
                    const size_t idx = harm3dIndex(ir, it, ip);
                    const float br = bScale * brField[idx];
                    const float bth = bScale * bthField[idx];
                    const float bph = bScale * (mad ? 0.42f : 0.18f) * std::sqrt(std::max(pressureField[idx], harmUFloor_)) *
                        (1.0f + 0.12f * arm2);
                    const float b2 = br * br + bth * bth + bph * bph;
                    const float beta = pressureField[idx] / std::max(0.5f * b2, 1e-12f);
                    if (rhoField[idx] > 8.0f * harmRhoFloor_) {
                        betaMin = std::min(betaMin, beta);
                        betaSum += beta;
                        ++betaCount;
                    }
                    sigmaMax = std::max(sigmaMax, b2 / std::max(rhoField[idx], harmRhoFloor_));
                    if (ir == fluxIr) {
                        const float area = r * r * sinTh * dtheta * dphi;
                        fluxBH += std::abs(br) * area;
                        mdot += std::max(-rhoField[idx] * vrField[idx], 0.0f) * area;
                    }
                    const size_t base = idx * 12;
                    harmUpload_[base + 0] = rhoField[idx];
                    harmUpload_[base + 1] = uField[idx];
                    harmUpload_[base + 2] = vrField[idx];
                    harmUpload_[base + 3] = vthField[idx];
                    harmUpload_[base + 4] = vphField[idx];
                    harmUpload_[base + 5] = br;
                    harmUpload_[base + 6] = bth;
                    harmUpload_[base + 7] = bph;
                    harmUpload_[base + 8] = 0.0f;
                    harmUpload_[base + 9] = 0.0f;
                    harmUpload_[base + 10] = 0.0f;
                    harmUpload_[base + 11] = 0.0f;
                }
            }
        }
        harmDiagBetaMin_ = (betaCount > 0) ? betaMin : 0.0f;
        harmDiagBetaMean_ = (betaCount > 0) ? betaSum / static_cast<float>(betaCount) : 0.0f;
        harmDiagSigmaMax_ = sigmaMax;
        harmDiagPhiBH_ = 0.5f * fluxBH / std::sqrt(std::max(mdot, 1e-10f));
    }

    void uploadHarmStateToGpu() {
        if (!harmGpuA_ || !harmGpuB_) return;
        if (harmP_.empty() || harmNeedsReset_) {
            resetHarmTorus();
        }
        packHarmPrimitivesForGpu();
        const size_t bytes = harmUpload_.size() * sizeof(float);
        wgfx::queue.writeBuffer(harmGpuA_->buffer, 0, harmUpload_.data(), bytes);
        wgfx::queue.writeBuffer(harmGpuB_->buffer, 0, harmUpload_.data(), bytes);
        harmGpuNeedsUpload_ = false;
    }

    uint32_t harmWorkgroups(int n) const {
        return (static_cast<uint32_t>(n) + 7u) / 8u;
    }

    void dispatchHarmCompute(wgfx::ComputePass& cp) {
        if (!harmComputeStep_ || !harmComputeCopy_) return;
        if (harmP_.empty() || harmNeedsReset_) {
            resetHarmTorus();
        }
        if (harmGpuNeedsUpload_) {
            uploadHarmStateToGpu();
        }
        if (harmPaused_) return;

        harmComputeParams_.gridN = static_cast<uint32_t>(harmGridSize_);
        harmComputeParams_.thetaN = static_cast<uint32_t>(harmThetaSize());
        harmComputeParams_.phiN = static_cast<uint32_t>(harmPhiSize());
        harmComputeParams_.substeps = static_cast<uint32_t>(std::clamp(harmSubsteps_, 1, 12));
        harmComputeParams_.dt = std::clamp(harmDt_, 0.00005f, 0.03f);
        harmComputeParams_.rin = std::max(harmRin_, 1.05f);
        harmComputeParams_.rout = std::max(harmRout_, harmComputeParams_.rin + 4.0f);
        harmComputeParams_.spin = std::clamp(harmSpin_, -0.98f, 0.98f);
        harmComputeParams_.rhoFloor = std::max(harmRhoFloor_, 1e-8f);
        harmComputeParams_.uFloor = std::max(harmUFloor_, 1e-9f);
        harmComputeParams_.magneticLoop = harmMagneticLoop_;
        harmComputeParams_.time = harmTime_;
        harmComputeParams_.problem = static_cast<uint32_t>(std::clamp(harmProblem_, 0, 3));

        wgfx::queue.writeBuffer(harmComputeParamsUni_->buffer, 0,
            &harmComputeParams_, sizeof(HarmComputeParams));
        auto pinOffset = [](wgfx::Compute* c) {
            if (!c) return;
            c->uniforms.dynamicOffsets.resize(1, 0);
            c->uniforms.dynamicOffsets[0] = 0;
            if (!c->uniforms.uniforms.empty()) {
                c->uniforms.uniforms[0]->quantity = 0;
            }
        };
        pinOffset(harmComputeStep_);
        pinOffset(harmComputeCopy_);

        const uint32_t wg1 = harmWorkgroups(harmGridSize_);
        const uint32_t wg2 = harmWorkgroups(harmThetaSize());
        const uint32_t wg3 = (static_cast<uint32_t>(harmPhiSize()) + 3u) / 4u;
        const int substeps = std::clamp(harmSubsteps_, 1, 12);
        for (int s = 0; s < substeps; ++s) {
            cp.drawXYZ(harmComputeStep_, wg1, wg2, wg3);
            cp.drawXYZ(harmComputeCopy_, wg1, wg2, wg3);
            harmTime_ += harmComputeParams_.dt;
        }
    }

    void renderHarmGrmhd(float dt) {
        if (harmP_.empty()) {
            resizeHarmBuffers();
        }
        if (harmUseGpu_) {
            if (harmNeedsReset_ || harmGpuNeedsUpload_) {
                uploadHarmStateToGpu();
            }
        } else {
            stepHarmSimulation();
            uploadHarmField();
        }
        int width = 1280;
        int height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);
        processHarmNavigation(width, height, aspect);

        gpu2dState_.orbital = glm::vec4(0.0f, 0.0f, 0.0f, static_cast<float>(harmViewMode_));
        gpu2dState_.tuning = glm::vec4(std::max(harmColorScale_, 0.001f), harmRin_, std::max(twoDZoom_, 1e-6f), harmSpin_);
        gpu2dState_.render = glm::vec4(harmTime_, aspect, harmCameraInclination_, harmCameraYaw_);
        gpu2dState_.pan = glm::vec4(twoDPan_.x, twoDPan_.y, harmRin_, 0.0f);
        gpu2dState_.tdse = glm::vec4(
            static_cast<float>(harmGridSize_),
            std::max(harmRout_, 1.0f),
            static_cast<float>(harmThetaSize()),
            static_cast<float>(harmPhiSize()));

        writeRenderUniform(pipelineHarmGrmhd_, reinterpret_cast<const float*>(&gpu2dState_));
        pipelineHarmGrmhd_->setVertexBuffer(vbo2d_.get());
        pipelineHarmGrmhd_->setIndexBuffer(ibo2d_.get());
        pipeline = pipelineHarmGrmhd_;
    }

    void render2d(float dt) {
        twoDTime_ += std::max(dt, 0.0f);

        if (twoDUseTdse_) {
            stepTdseSimulation();
            uploadTdseField();
        }

        int width = 1280;
        int height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);
        process2dNavigation(width, height, aspect, dt);

        gpu2dState_.orbital = glm::vec4(
            static_cast<float>(quantum_.n),
            static_cast<float>(quantum_.l),
            static_cast<float>(quantum_.m),
            static_cast<float>(colorMode_));
        gpu2dState_.tuning = glm::vec4(
            std::max(intensityScale_, 0.001f),
            std::max(intensityRange_, 0.001f),
            std::max(twoDZoom_, 1e-6f),
            std::max(twoDThickness_, 0.01f));
        gpu2dState_.render = glm::vec4(twoDTime_, aspect, std::max(twoDPhaseSpeed_, 0.0f), 0.0f);
        gpu2dState_.pan = glm::vec4(twoDPan_.x, twoDPan_.y, twoDSliceZ_, twoDIntegrateDepth_ ? 1.0f : 0.0f);
        gpu2dState_.tdse = glm::vec4(
            static_cast<float>(tdseGridSize_),
            std::max(tdseDomainHalfExtent_, 1.0f),
            tdseOverlayPotential_ ? 1.0f : 0.0f,
            0.0f);

        // Choose pipeline based on mode
        wgfx::Pipeline* activePipeline = twoDUseTdse_ ? pipelineTdse2d_ : pipeline2d_;
        writeRenderUniform(activePipeline, reinterpret_cast<const float*>(&gpu2dState_));
        activePipeline->setVertexBuffer(vbo2d_.get());
        activePipeline->setIndexBuffer(ibo2d_.get());
        pipeline = activePipeline;
    }

    void processHarmNavigation(int width, int height, float aspect) {
        ImGuiIO& io = ImGui::GetIO();
        const bool allowMouseCapture = !io.WantCaptureMouse;

        float wheel = Context::Instance().consumeWheelDelta();
        if (wheel != 0.0f && allowMouseCapture) {
            twoDZoom_ *= std::exp(wheel * 0.12f);
            twoDZoom_ = std::clamp(twoDZoom_, 1e-6f, 1e6f);
        }

        float mx = 0.0f;
        float my = 0.0f;
        Uint32 mask = SDL_GetMouseState(&mx, &my);
        const bool leftDown = (mask & SDL_BUTTON_LMASK) != 0;
        const bool panDown = (mask & SDL_BUTTON_MMASK) != 0 || (mask & SDL_BUTTON_RMASK) != 0;
        const bool draggingNow = allowMouseCapture && (leftDown || panDown);
        if (!draggingNow) {
            twoDDragging_ = false;
            return;
        }

        if (!twoDDragging_) {
            twoDDragging_ = true;
            twoDLastMouse_ = glm::vec2(mx, my);
            return;
        }

        const glm::vec2 curr(mx, my);
        const glm::vec2 delta = curr - twoDLastMouse_;
        twoDLastMouse_ = curr;

        const float safeWidth = static_cast<float>(std::max(1, width));
        const float safeHeight = static_cast<float>(std::max(1, height));
        if (leftDown && !panDown) {
            harmCameraYaw_ += delta.x * (2.4f / safeWidth);
            harmCameraInclination_ += delta.y * (1.8f / safeHeight);
            if (harmCameraYaw_ > kPi) harmCameraYaw_ -= 2.0f * kPi;
            if (harmCameraYaw_ < -kPi) harmCameraYaw_ += 2.0f * kPi;
            harmCameraInclination_ = std::clamp(harmCameraInclination_, 0.05f, 1.45f);
            return;
        }

        const float ndcDx = (2.0f * delta.x) / safeWidth;
        const float ndcDy = (-2.0f * delta.y) / safeHeight;
        twoDPan_.x -= ndcDx / (std::max(twoDZoom_, 0.01f) * std::max(aspect, 0.001f));
        twoDPan_.y -= ndcDy / std::max(twoDZoom_, 0.01f);
    }

    void process2dNavigation(int width, int height, float aspect, float dt) {
        ImGuiIO& io = ImGui::GetIO();
        const bool allowMouseCapture = !io.WantCaptureMouse;
        const bool allowKeyboardCapture = !io.WantCaptureKeyboard;

        if (twoDUseTdse_ && allowKeyboardCapture) {
            const Uint8* keys = SDL_GetKeyboardState(nullptr);
            glm::vec2 move(0.0f);
            if (keys[SDL_SCANCODE_W]) move.y += 1.0f;
            if (keys[SDL_SCANCODE_S]) move.y -= 1.0f;
            if (keys[SDL_SCANCODE_A]) move.x -= 1.0f;
            if (keys[SDL_SCANCODE_D]) move.x += 1.0f;
            if (glm::dot(move, move) > 0.0f) {
                const float len = std::sqrt(move.x * move.x + move.y * move.y);
                move /= len;
                tdsePacketCenter_ += move * tdsePacketMoveSpeed_ * std::max(dt, 0.0f);
                const float domain = std::max(tdseDomainHalfExtent_, 1.0f);
                tdsePacketCenter_.x = std::clamp(tdsePacketCenter_.x, -0.95f * domain, 0.95f * domain);
                tdsePacketCenter_.y = std::clamp(tdsePacketCenter_.y, -0.95f * domain, 0.95f * domain);
                tdseWaveNeedsReset_ = true;
            }
        }

        float wheel = Context::Instance().consumeWheelDelta();
        if (wheel != 0.0f && allowMouseCapture) {
            // Exponential wheel zoom avoids sign issues and feels scale-invariant.
            twoDZoom_ *= std::exp(wheel * 0.12f);
            twoDZoom_ = std::clamp(twoDZoom_, 1e-6f, 1e6f);
        }

        float mx = 0.0f;
        float my = 0.0f;
        Uint32 mask = SDL_GetMouseState(&mx, &my);
        const bool draggingNow = allowMouseCapture && ((mask & SDL_BUTTON_LMASK) != 0 || (mask & SDL_BUTTON_MMASK) != 0);
        if (!draggingNow) {
            twoDDragging_ = false;
            return;
        }

        if (!twoDDragging_) {
            twoDDragging_ = true;
            twoDLastMouse_ = glm::vec2(mx, my);
            return;
        }

        const glm::vec2 curr(mx, my);
        const glm::vec2 delta = curr - twoDLastMouse_;
        twoDLastMouse_ = curr;

        const float safeWidth = std::max(1, width);
        const float safeHeight = std::max(1, height);
        const float ndcDx = (2.0f * delta.x) / static_cast<float>(safeWidth);
        const float ndcDy = (-2.0f * delta.y) / static_cast<float>(safeHeight);
        twoDPan_.x -= ndcDx / (std::max(twoDZoom_, 0.01f) * std::max(aspect, 0.001f));
        twoDPan_.y -= ndcDy / std::max(twoDZoom_, 0.01f);
    }

    static int orbitalLetterToL(char c) {
        static const std::string letters = "spdfghiklmnoqrtuvwxyz";
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        const size_t pos = letters.find(c);
        if (pos == std::string::npos) {
            return -1;
        }
        return static_cast<int>(pos);
    }

    static const char* elementNameFromZ(int z) {
        static constexpr std::array<const char*, 31> kNames = {
            "Unknown",
            "Hydrogen", "Helium", "Lithium", "Beryllium", "Boron",
            "Carbon", "Nitrogen", "Oxygen", "Fluorine", "Neon",
            "Sodium", "Magnesium", "Aluminum", "Silicon", "Phosphorus",
            "Sulfur", "Chlorine", "Argon", "Potassium", "Calcium",
            "Scandium", "Titanium", "Vanadium", "Chromium", "Manganese",
            "Iron", "Cobalt", "Nickel", "Copper", "Zinc"
        };
        if (z >= 1 && z < static_cast<int>(kNames.size())) {
            return kNames[static_cast<size_t>(z)];
        }
        return "Custom atom";
    }

    bool parseElectronConfiguration(const std::string& config, std::vector<SpinOrbitalState>& outStates, std::string& err) const {
        outStates.clear();
        err.clear();

        std::stringstream ss(config);
        std::string token;
        while (ss >> token) {
            size_t i = 0;
            while (i < token.size() && std::isdigit(static_cast<unsigned char>(token[i])) != 0) {
                ++i;
            }
            if (i == 0 || i >= token.size()) {
                err = "Invalid token: '" + token + "'";
                return false;
            }

            int n = std::stoi(token.substr(0, i));
            int l = orbitalLetterToL(token[i]);
            if (l < 0) {
                err = "Unknown orbital letter in token: '" + token + "'";
                return false;
            }
            ++i;

            if (n < 1 || n > 30) {
                err = "n must be in [1, 30] in token: '" + token + "'";
                return false;
            }
            if (l >= n) {
                err = "Need l < n in token: '" + token + "'";
                return false;
            }
            if (l > 13) {
                err = "l > 13 is not supported in this renderer yet";
                return false;
            }

            if (i < token.size() && token[i] == '^') {
                ++i;
            }

            int occupancy = 1;
            if (i < token.size()) {
                const std::string occText = token.substr(i);
                if (occText.empty() ||
                    std::all_of(occText.begin(), occText.end(), [](char ch) {
                        return std::isdigit(static_cast<unsigned char>(ch)) != 0;
                    }) == false) {
                    err = "Invalid occupancy in token: '" + token + "'";
                    return false;
                }
                occupancy = std::stoi(occText);
            }

            const int capacity = 2 * (2 * l + 1);
            if (occupancy < 1 || occupancy > capacity) {
                err = "Occupancy exceeds shell capacity in token: '" + token + "'";
                return false;
            }

            int assigned = 0;
            for (int m = -l; m <= l && assigned < occupancy; ++m) {
                outStates.push_back({ n, l, m, +1 });
                ++assigned;
            }
            for (int m = -l; m <= l && assigned < occupancy; ++m) {
                outStates.push_back({ n, l, m, -1 });
                ++assigned;
            }
        }

        if (outStates.empty()) {
            err = "Configuration is empty";
            return false;
        }
        return true;
    }

    void fillSingleOrbitalAttribData() {
        for (int i = 0; i < kMaxParticles; ++i) {
            const size_t base = static_cast<size_t>(i) * 3;
            orbitalAttribData_[base + 0] = static_cast<float>(quantum_.n);
            orbitalAttribData_[base + 1] = static_cast<float>(quantum_.l);
            orbitalAttribData_[base + 2] = static_cast<float>(quantum_.m);
        }
    }

    void rebuildOrbitalAttributeBuffer() {
        if (!vbo_) {
            return;
        }

        if (!useConfiguration_ || electronOrbitals_.empty()) {
            fillSingleOrbitalAttribData();
            vbo_->write(orbitalAttribData_);
            return;
        }

        const int sampleCount = std::clamp(quantum_.sampleCount, 1000, kMaxParticles);
        std::vector<int> map(static_cast<size_t>(sampleCount));
        for (int i = 0; i < sampleCount; ++i) {
            map[static_cast<size_t>(i)] = i % static_cast<int>(electronOrbitals_.size());
        }

        std::mt19937 rng(static_cast<uint32_t>(sampleSeed_ * 997.0f) ^ 0x9e3779b9u);
        std::shuffle(map.begin(), map.end(), rng);

        for (int i = 0; i < kMaxParticles; ++i) {
            int orbitalIndex = 0;
            if (i < sampleCount) {
                orbitalIndex = map[static_cast<size_t>(i)];
            }
            const SpinOrbitalState& st = electronOrbitals_[static_cast<size_t>(orbitalIndex)];
            const size_t base = static_cast<size_t>(i) * 3;
            orbitalAttribData_[base + 0] = static_cast<float>(st.n);
            orbitalAttribData_[base + 1] = static_cast<float>(st.l);
            orbitalAttribData_[base + 2] = static_cast<float>(st.m);
        }

        vbo_->write(orbitalAttribData_);
    }

    void rebuildVmcPointBuffer() {
        if (!vbo_) {
            return;
        }

        const std::vector<glm::vec3>& cloud = vmc_.pointCloud();
        const std::vector<glm::vec3>& source = cloud.empty() ? vmc_.positions() : cloud;
        vmcDrawCount_ = std::min<int>(static_cast<int>(source.size()), kMaxParticles);

        if (vmcDrawCount_ <= 0) {
            vmcUploadData_.assign({ 0.0f, 0.0f, 0.0f });
            vmcDrawCount_ = 1;
            vbo_->write(vmcUploadData_);
            return;
        }

        vmcUploadData_.resize(static_cast<size_t>(vmcDrawCount_) * 3);
        for (int i = 0; i < vmcDrawCount_; ++i) {
            const glm::vec3& p = source[static_cast<size_t>(i)];
            const size_t base = static_cast<size_t>(i) * 3;
            vmcUploadData_[base + 0] = p.x;
            vmcUploadData_[base + 1] = p.y;
            vmcUploadData_[base + 2] = p.z;
        }

        vbo_->write(vmcUploadData_);
    }

    void centerCameraOnVmcCloud(bool immediate) {
        const std::vector<glm::vec3>& cloud = vmc_.pointCloud();
        const std::vector<glm::vec3>& source = cloud.empty() ? vmc_.positions() : cloud;
        if (source.empty()) {
            return;
        }

        const size_t maxSamples = std::min<size_t>(source.size(), 4000);
        const size_t stride = std::max<size_t>(1, source.size() / maxSamples);

        glm::vec3 centroid(0.0f);
        size_t count = 0;
        for (size_t i = 0; i < source.size(); i += stride) {
            centroid += source[i];
            ++count;
            if (count >= maxSamples) {
                break;
            }
        }
        if (count == 0) {
            return;
        }
        centroid /= static_cast<float>(count);

        float maxDist = 0.0f;
        for (size_t i = 0; i < source.size(); i += stride) {
            maxDist = std::max(maxDist, glm::length(source[i] - centroid));
        }

        const float desiredRadius = std::clamp(maxDist * 4.0f + 8.0f, 8.0f, 320.0f);
        const float blend = immediate ? 1.0f : 0.08f;
        camera_.target = glm::mix(camera_.target, centroid, blend);
        camera_.radius = glm::mix(camera_.radius, desiredRadius, blend);
    }

    void updateCameraForVmc() {
        if (!vmcAutoCenterCamera_) {
            return;
        }
        centerCameraOnVmcCloud(false);
    }

    void applyElectronConfiguration(const std::string& text) {
        std::vector<SpinOrbitalState> parsed;
        std::string parseError;
        if (!parseElectronConfiguration(text, parsed, parseError)) {
            configError_ = parseError;
            configMessage_.clear();
            return;
        }

        electronOrbitals_ = std::move(parsed);
        useConfiguration_ = true;
        configError_.clear();

        const int z = static_cast<int>(electronOrbitals_.size());
        configMessage_ = "Loaded " + std::to_string(z) + " electron(s): " + elementNameFromZ(z);

        vmc_.setNucleusCharge(z);
        vmc_.setWalkerCount(vmcWalkerCount_);
        vmc_.setParameters(vmcStepSize_, vmcZetaScale_, vmcJastrowBeta_);
        std::string vmcErr;
        if (!vmc_.configure(electronOrbitals_, vmcErr)) {
            configError_ = "VMC setup failed: " + vmcErr;
            vmcConfigured_ = false;
            return;
        }
        vmcConfigured_ = true;
        vmcMode_ = true;
        vmc_.setMaxCloudPoints(static_cast<size_t>(quantum_.sampleCount));
        vmc_.resetWalkers(static_cast<uint32_t>(sampleSeed_ * 7919.0f) ^ 0x9e3779b9u);
        centerCameraOnVmcCloud(true);

        quantum_.n = electronOrbitals_.front().n;
        quantum_.l = electronOrbitals_.front().l;
        quantum_.m = electronOrbitals_.front().m;
        quantum_.clamp();

        sampleSeed_ += 1.0f;
        orbitalBufferDirty_ = true;
    }

    void processShortcuts() {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureKeyboard) {
            return;
        }

        if (vmcMode_) {
            return;
        }

        const Uint8* keys = SDL_GetKeyboardState(nullptr);
        bool w = keys[SDL_SCANCODE_W];
        bool s = keys[SDL_SCANCODE_S];
        bool e = keys[SDL_SCANCODE_E];
        bool d = keys[SDL_SCANCODE_D];
        bool r = keys[SDL_SCANCODE_R];
        bool f = keys[SDL_SCANCODE_F];
        bool t = keys[SDL_SCANCODE_T];
        bool g = keys[SDL_SCANCODE_G];

        int oldN = quantum_.n;
        int oldL = quantum_.l;
        int oldM = quantum_.m;

        if (w && !prevW_) quantum_.n += 1;
        if (s && !prevS_) quantum_.n -= 1;
        if (e && !prevE_) quantum_.l += 1;
        if (d && !prevD_) quantum_.l -= 1;
        if (r && !prevR_) quantum_.m += 1;
        if (f && !prevF_) quantum_.m -= 1;
        if (t && !prevT_) quantum_.sampleCount += 10000;
        if (g && !prevG_) quantum_.sampleCount -= 10000;

        quantum_.sampleCount = std::clamp(quantum_.sampleCount, 1000, kMaxParticles);
        quantum_.clamp();

        if (oldN != quantum_.n || oldL != quantum_.l || oldM != quantum_.m) {
            sampleSeed_ += 1.0f;
        }

        prevW_ = w;
        prevS_ = s;
        prevE_ = e;
        prevD_ = d;
        prevR_ = r;
        prevF_ = f;
        prevT_ = t;
        prevG_ = g;
    }

    void advanceSimulation(float dt) {
        simulationTime_ += std::max(dt, 0.0f) * flowSpeed_;
    }
};
