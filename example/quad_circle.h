#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <SDL3/SDL.h>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include "context.h"
#include "imgui.h"
#include "wgfx.h"

class Quad {
public:
    static Quad& Instance() {
        static Quad instance;
        return instance;
    }

    bool isCityWalkMode() const {
        return false;
    }

    bool shouldDrawMainSceneGeometry() const {
        return true;
    }

    wgfx::Pipeline* pipeline = nullptr;

    void dispatchCompute3d() {}

    void drawImGuiPanel() {
        ImGui::Begin("Spectral Path Tracer");
        ImGui::Text("Decanter-style glass with wavelength-based refraction");
        ImGui::Text("Move: WASD, Look: Num4/6/8/2");
        ImGui::SliderInt("max bounces", &maxBounces_, 1, 12);
        ImGui::SliderInt("samples / pixel", &spp_, 1, 32);
        ImGui::SliderFloat("dispersion", &dispersionStrength_, 0.0f, 0.12f, "%.4f");
        ImGui::SliderFloat("glass roughness", &surfaceRoughness_, 0.0f, 0.08f, "%.4f");
        const char* envItems = "Studio HDR\0Physical Sky\0Sunset Gradient\0";
        ImGui::Combo("environment", &envMode_, envItems);
        ImGui::SliderFloat("env rotation", &envRotation_, -3.14159f, 3.14159f, "%.3f");
        ImGui::SliderFloat("exposure", &exposure_, 0.2f, 2.0f, "%.3f");
        ImGui::SliderFloat("env brightness", &envBrightness_, 0.0f, 5.0f, "%.3f");
        if (ImGui::Button("Reset Camera")) {
            cameraPos_ = glm::vec3(0.0f, 1.1f, 3.2f);
            cameraYaw_ = 3.14159f;
            cameraPitch_ = -0.12f;
        }
        ImGui::TextWrapped("Decanter GLB: %s", decanterGlbPresent_ ? "found in res/ (next step: triangle intersection)" : "missing");
        ImGui::End();
    }

    void render(float dt) {
        time_ += std::max(dt, 0.0f);
        updateCamera(dt);

        int width = 1280;
        int height = 720;
        SDL_GetWindowSize(Context::Instance().window, &width, &height);
        const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : (16.0f / 9.0f);

        gpu2dState_.orbital = glm::vec4(
            std::clamp(static_cast<float>(maxBounces_), 1.0f, 32.0f),
            std::clamp(static_cast<float>(spp_), 1.0f, 256.0f),
            std::max(dispersionStrength_, 0.0f),
            std::max(surfaceRoughness_, 0.0f));
        gpu2dState_.tuning = glm::vec4(
            cameraPos_.x,
            cameraPos_.y,
            cameraPos_.z,
            std::max(exposure_, 0.01f));
        gpu2dState_.render = glm::vec4(time_, aspect, cameraYaw_, cameraPitch_);
        gpu2dState_.pan = glm::vec4(
            std::max(envBrightness_, 0.0f),
            static_cast<float>(envMode_),
            envRotation_,
            0.0f);
        gpu2dState_.tdse = glm::vec4(0.0f);

        writeRenderUniform(pipeline2d_, reinterpret_cast<const float*>(&gpu2dState_));
        pipeline2d_->setVertexBuffer(vbo2d_.get());
        pipeline2d_->setIndexBuffer(ibo2d_.get());
        pipeline = pipeline2d_;
    }

private:
    struct alignas(16) Gpu2dState {
        glm::vec4 orbital;
        glm::vec4 tuning;
        glm::vec4 render;
        glm::vec4 pan;
        glm::vec4 tdse;
    };

    std::unique_ptr<wgfx::VertexBuffer> vbo2d_;
    std::unique_ptr<wgfx::IndexBuffer> ibo2d_;
    wgfx::Pipeline* pipeline2d_ = nullptr;
    wgfx::Uniform* stateUniform2d_ = nullptr;
    wgfx::Texture envTexture_{};
    wgfx::Uniform* bvhNodesStorage_ = nullptr;
    wgfx::Uniform* bvhTrisStorage_ = nullptr;

    Gpu2dState gpu2dState_{};
    float time_ = 0.0f;

    int maxBounces_ = 8;
    int spp_ = 10;
    float dispersionStrength_ = 0.03f;
    float surfaceRoughness_ = 0.004f;
    glm::vec3 cameraPos_ = glm::vec3(0.0f, 1.1f, 3.2f);
    float cameraYaw_ = 3.14159f;
    float cameraPitch_ = -0.12f;
    float exposure_ = 1.0f;
    float envBrightness_ = 1.5f;
    int envMode_ = 0;
    float envRotation_ = 0.0f;
    float moveSpeed_ = 2.6f;
    float lookSpeed_ = 2.8f;
    bool decanterGlbPresent_ = false;
    int triangleCount_ = 0;
    int bvhNodeCount_ = 0;

    struct CpuTri {
        glm::vec3 v0;
        glm::vec3 v1;
        glm::vec3 v2;
    };

    struct BvhNodeCpu {
        glm::vec3 bmin = glm::vec3(0.0f);
        glm::vec3 bmax = glm::vec3(0.0f);
        int left = -1;
        int right = -1;
        int first = 0;
        int count = 0;
    };

    struct GpuTri {
        glm::vec4 v0;
        glm::vec4 v1;
        glm::vec4 v2;
    };

    struct GpuNode {
        glm::vec4 bminLeft;
        glm::vec4 bmaxCount;
    };

    static void writeRenderUniform(wgfx::Pipeline* activePipeline, const float* data) {
        if (!activePipeline || activePipeline->uniforms.uniforms.empty()) return;
        wgfx::Uniform* uniform = activePipeline->uniforms.uniforms.at(0);
        wgfx::queue.writeBuffer(uniform->buffer, 0, data, uniform->minBindingSize);
        if (activePipeline->uniforms.dynamicOffsets.empty()) {
            activePipeline->uniforms.dynamicOffsets.resize(1, 0);
        }
        activePipeline->uniforms.dynamicOffsets[0] = 0;
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

    void updateCamera(float dt) {
        const float frameDt = std::clamp(dt, 0.0f, 0.06f);
        const Uint8* ks = SDL_GetKeyboardState(nullptr);
        if (!ks) return;

        const float yawLeft = (ks[SDL_SCANCODE_KP_4] || ks[SDL_SCANCODE_LEFT] || ks[SDL_SCANCODE_J]) ? 1.0f : 0.0f;
        const float yawRight = (ks[SDL_SCANCODE_KP_6] || ks[SDL_SCANCODE_RIGHT] || ks[SDL_SCANCODE_L]) ? 1.0f : 0.0f;
        const float pitchUp = (ks[SDL_SCANCODE_KP_8] || ks[SDL_SCANCODE_UP] || ks[SDL_SCANCODE_I]) ? 1.0f : 0.0f;
        const float pitchDown = (ks[SDL_SCANCODE_KP_2] || ks[SDL_SCANCODE_DOWN] || ks[SDL_SCANCODE_K]) ? 1.0f : 0.0f;
        cameraYaw_ += (yawLeft - yawRight) * lookSpeed_ * frameDt;
        cameraPitch_ += (pitchUp - pitchDown) * lookSpeed_ * frameDt;
        cameraPitch_ = std::clamp(cameraPitch_, -1.45f, 1.45f);

        const glm::vec3 forward = glm::normalize(glm::vec3(
            std::cos(cameraPitch_) * std::sin(cameraYaw_),
            std::sin(cameraPitch_),
            std::cos(cameraPitch_) * std::cos(cameraYaw_)));
        const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));

        glm::vec3 move(0.0f);
        if (ks[SDL_SCANCODE_W]) move += forward;
        if (ks[SDL_SCANCODE_S]) move -= forward;
        if (ks[SDL_SCANCODE_D]) move += right;
        if (ks[SDL_SCANCODE_A]) move -= right;
        if (glm::length(move) > 0.0f) {
            cameraPos_ += glm::normalize(move) * moveSpeed_ * frameDt;
        }
    }

    static bool readAccessorPositions(
        const fastgltf::Asset& asset,
        const fastgltf::Accessor& accessor,
        std::vector<glm::vec3>& outPositions) {
        outPositions.resize(accessor.count);
        bool ok = true;
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
            asset, accessor, [&](fastgltf::math::fvec3 p, std::size_t idx) {
                if (idx >= outPositions.size()) {
                    ok = false;
                    return;
                }
                outPositions[idx] = glm::vec3(p.x(), p.y(), p.z());
            });
        return ok;
    }

    static bool readAccessorIndices(
        const fastgltf::Asset& asset,
        const fastgltf::Accessor& accessor,
        std::vector<uint32_t>& outIndices) {
        outIndices.resize(accessor.count);
        bool ok = true;
        fastgltf::iterateAccessorWithIndex<std::uint32_t>(
            asset, accessor, [&](std::uint32_t i, std::size_t idx) {
                if (idx >= outIndices.size()) {
                    ok = false;
                    return;
                }
                outIndices[idx] = static_cast<uint32_t>(i);
            });
        return ok;
    }

    static void computeBoundsForRange(
        const std::vector<CpuTri>& tris,
        const std::vector<int>& triIndices,
        int first,
        int count,
        glm::vec3& outMin,
        glm::vec3& outMax,
        glm::vec3& outCentroidMin,
        glm::vec3& outCentroidMax) {
        outMin = glm::vec3(std::numeric_limits<float>::max());
        outMax = glm::vec3(-std::numeric_limits<float>::max());
        outCentroidMin = glm::vec3(std::numeric_limits<float>::max());
        outCentroidMax = glm::vec3(-std::numeric_limits<float>::max());
        for (int i = 0; i < count; ++i) {
            const CpuTri& t = tris[triIndices[first + i]];
            const glm::vec3 triMin = glm::min(t.v0, glm::min(t.v1, t.v2));
            const glm::vec3 triMax = glm::max(t.v0, glm::max(t.v1, t.v2));
            outMin = glm::min(outMin, triMin);
            outMax = glm::max(outMax, triMax);
            const glm::vec3 c = (t.v0 + t.v1 + t.v2) * (1.0f / 3.0f);
            outCentroidMin = glm::min(outCentroidMin, c);
            outCentroidMax = glm::max(outCentroidMax, c);
        }
    }

    static int buildBvhRecursive(
        const std::vector<CpuTri>& tris,
        std::vector<int>& triIndices,
        std::vector<BvhNodeCpu>& nodes,
        int first,
        int count) {
        const int nodeIndex = static_cast<int>(nodes.size());
        nodes.push_back(BvhNodeCpu{});
        BvhNodeCpu& node = nodes.back();

        glm::vec3 bmin, bmax, cmin, cmax;
        computeBoundsForRange(tris, triIndices, first, count, bmin, bmax, cmin, cmax);
        node.bmin = bmin;
        node.bmax = bmax;

        if (count <= 6) {
        node.first = first;
        node.count = count;
        return nodeIndex;
        }

        const glm::vec3 cext = cmax - cmin;
        int axis = 0;
        if (cext.y > cext.x && cext.y >= cext.z) axis = 1;
        else if (cext.z > cext.x && cext.z >= cext.y) axis = 2;

        const float split = 0.5f * (cmin[axis] + cmax[axis]);
        int mid = first;
        for (int i = first; i < first + count; ++i) {
            const CpuTri& t = tris[triIndices[i]];
            const glm::vec3 c = (t.v0 + t.v1 + t.v2) * (1.0f / 3.0f);
            if (c[axis] < split) {
                std::swap(triIndices[i], triIndices[mid]);
                mid++;
            }
        }

        if (mid == first || mid == first + count) {
            mid = first + count / 2;
        }

        const int leftCount = mid - first;
        const int rightCount = count - leftCount;
        const int leftChild = buildBvhRecursive(tris, triIndices, nodes, first, leftCount);
        const int rightChild = buildBvhRecursive(tris, triIndices, nodes, mid, rightCount);
        (void)rightChild;

        node.left = leftChild;
        node.right = rightChild;
        node.count = 0;
        return nodeIndex;
    }

    bool loadDecanterTriangles(std::vector<CpuTri>& outTris) {
        const std::filesystem::path glbPath = std::filesystem::path(RESOURCE_DIR) / "wine_decanter_and_glass.glb";
        if (!std::filesystem::exists(glbPath)) {
            return false;
        }

        auto data = fastgltf::GltfDataBuffer::FromPath(glbPath);
        if (data.error() != fastgltf::Error::None) {
            return false;
        }

        fastgltf::Parser parser(
            fastgltf::Extensions::KHR_mesh_quantization |
            fastgltf::Extensions::KHR_texture_transform |
            fastgltf::Extensions::KHR_materials_ior);
        constexpr auto options =
            fastgltf::Options::DontRequireValidAssetMember |
            fastgltf::Options::AllowDouble |
            fastgltf::Options::LoadExternalBuffers |
            fastgltf::Options::LoadExternalImages |
            fastgltf::Options::GenerateMeshIndices;

        auto type = fastgltf::determineGltfFileType(data.get());
        auto assetResult = (type == fastgltf::GltfType::GLB)
            ? parser.loadGltfBinary(data.get(), glbPath.parent_path(), options)
            : parser.loadGltf(data.get(), glbPath.parent_path(), options);
        if (assetResult.error() != fastgltf::Error::None) {
            return false;
        }
        const fastgltf::Asset& asset = assetResult.get();

        for (const fastgltf::Mesh& mesh : asset.meshes) {
            for (const fastgltf::Primitive& prim : mesh.primitives) {
                if (prim.type != fastgltf::PrimitiveType::Triangles) continue;
                auto posIt = prim.findAttribute("POSITION");
                if (posIt == prim.attributes.end()) continue;
                const fastgltf::Accessor& posAccessor = asset.accessors[posIt->accessorIndex];

                std::vector<glm::vec3> positions;
                if (!readAccessorPositions(asset, posAccessor, positions)) continue;

                std::vector<uint32_t> indices;
                if (prim.indicesAccessor.has_value()) {
                    if (!readAccessorIndices(asset, asset.accessors[*prim.indicesAccessor], indices)) continue;
                } else {
                    indices.resize(positions.size());
                    for (size_t i = 0; i < positions.size(); ++i) indices[i] = static_cast<uint32_t>(i);
                }
                if (indices.size() < 3) continue;

                for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                    const uint32_t i0 = indices[i + 0];
                    const uint32_t i1 = indices[i + 1];
                    const uint32_t i2 = indices[i + 2];
                    if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) continue;
                    CpuTri tri;
                    tri.v0 = positions[i0];
                    tri.v1 = positions[i1];
                    tri.v2 = positions[i2];
                    outTris.push_back(tri);
                }
            }
        }
        return !outTris.empty();
    }

    bool buildAndUploadDecanterBvh() {
        std::vector<CpuTri> tris;
        if (!loadDecanterTriangles(tris)) {
            return false;
        }

        glm::vec3 modelMin(std::numeric_limits<float>::max());
        glm::vec3 modelMax(-std::numeric_limits<float>::max());
        for (const CpuTri& t : tris) {
            modelMin = glm::min(modelMin, glm::min(t.v0, glm::min(t.v1, t.v2)));
            modelMax = glm::max(modelMax, glm::max(t.v0, glm::max(t.v1, t.v2)));
        }
        const glm::vec3 center = 0.5f * (modelMin + modelMax);
        const glm::vec3 extent = modelMax - modelMin;
        const float longest = std::max(extent.x, std::max(extent.y, extent.z));
        const float invScale = (longest > 0.0f) ? (1.6f / longest) : 1.0f;
        for (CpuTri& t : tris) {
            t.v0 = (t.v0 - center) * invScale + glm::vec3(0.0f, 0.2f, 0.0f);
            t.v1 = (t.v1 - center) * invScale + glm::vec3(0.0f, 0.2f, 0.0f);
            t.v2 = (t.v2 - center) * invScale + glm::vec3(0.0f, 0.2f, 0.0f);
        }

        std::vector<int> triIndices(tris.size());
        for (size_t i = 0; i < triIndices.size(); ++i) triIndices[i] = static_cast<int>(i);

        std::vector<BvhNodeCpu> nodes;
        nodes.reserve(tris.size() * 2);
        buildBvhRecursive(tris, triIndices, nodes, 0, static_cast<int>(tris.size()));

        std::vector<CpuTri> sortedTris(tris.size());
        for (size_t i = 0; i < triIndices.size(); ++i) {
            sortedTris[i] = tris[triIndices[i]];
        }

        std::vector<GpuTri> gpuTris(sortedTris.size());
        for (size_t i = 0; i < sortedTris.size(); ++i) {
            gpuTris[i].v0 = glm::vec4(sortedTris[i].v0, 0.0f);
            gpuTris[i].v1 = glm::vec4(sortedTris[i].v1, 0.0f);
            gpuTris[i].v2 = glm::vec4(sortedTris[i].v2, 0.0f);
        }

        std::vector<GpuNode> gpuNodes(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            const float w0 = (nodes[i].count > 0) ? static_cast<float>(nodes[i].first) : static_cast<float>(nodes[i].left);
            const float w1 = (nodes[i].count > 0) ? static_cast<float>(nodes[i].count) : -static_cast<float>(nodes[i].right + 1);
            gpuNodes[i].bminLeft = glm::vec4(nodes[i].bmin, w0);
            gpuNodes[i].bmaxCount = glm::vec4(nodes[i].bmax, w1);
        }

        bvhNodesStorage_ = wgfx::createStorage(3, gpuNodes.size() * sizeof(GpuNode), gpuNodes.data(), true);
        bvhTrisStorage_ = wgfx::createStorage(4, gpuTris.size() * sizeof(GpuTri), gpuTris.data(), true);
        pipeline2d_->uniforms.setStorage(bvhNodesStorage_);
        pipeline2d_->uniforms.setStorage(bvhTrisStorage_);
        triangleCount_ = static_cast<int>(gpuTris.size());
        bvhNodeCount_ = static_cast<int>(gpuNodes.size());
        return true;
    }

    Quad() {
        pipeline2d_ = wgfx::loadPipeline(
            wgfx::loadFromFile((std::string(RESOURCE_DIR) + "/" + "circle_2d.wgsl").c_str()));
        stateUniform2d_ = wgfx::createUniform(0, sizeof(Gpu2dState), reinterpret_cast<const float*>(&gpu2dState_));
        pipeline2d_->uniforms.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        pipeline2d_->uniforms.setUniform(stateUniform2d_);
        envTexture_ = wgfx::loadTexture(std::string(RESOURCE_DIR) + "/" + "ferndale_studio_04_1k.hdr");
        pipeline2d_->addTexture(1, envTexture_);
        pipeline2d_->addSampler(2, envTexture_);
        decanterGlbPresent_ = std::filesystem::exists(std::string(RESOURCE_DIR) + "/" + "wine_decanter_and_glass.glb");
        if (!buildAndUploadDecanterBvh()) {
            static const GpuNode fallbackNode[1] = {
                { glm::vec4(-0.1f, -0.1f, -0.1f, 0.0f), glm::vec4(0.1f, 0.1f, 0.1f, 0.0f) }
            };
            static const GpuTri fallbackTri[1] = {
                { glm::vec4(0.0f, 0.0f, 0.0f, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 0.0f) }
            };
            bvhNodesStorage_ = wgfx::createStorage(3, sizeof(fallbackNode), fallbackNode, true);
            bvhTrisStorage_ = wgfx::createStorage(4, sizeof(fallbackTri), fallbackTri, true);
            pipeline2d_->uniforms.setStorage(bvhNodesStorage_);
            pipeline2d_->uniforms.setStorage(bvhTrisStorage_);
        }
        pipeline2d_->targets = 1;
        pipeline2d_->useDepth = false;

        init2dBuffers();
        pipeline2d_->init(vbo2d_.get());

        pipeline = pipeline2d_;
        pipeline->setVertexBuffer(vbo2d_.get());
        pipeline->setIndexBuffer(ibo2d_.get());
    }

    Quad(const Quad&) = delete;
    void operator=(const Quad&) = delete;
};
