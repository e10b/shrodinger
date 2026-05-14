#include "city_mesh.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/util.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <stb_image.h>

namespace {

glm::mat4 nodeLocalToMat4(const fastgltf::Node& node) {
    if (const auto* trs = std::get_if<fastgltf::TRS>(&node.transform)) {
        glm::vec3 t(trs->translation.x(), trs->translation.y(), trs->translation.z());
        glm::quat q(trs->rotation.w(), trs->rotation.x(), trs->rotation.y(), trs->rotation.z());
        glm::vec3 s(trs->scale.x(), trs->scale.y(), trs->scale.z());
        const glm::mat4 translate = glm::translate(glm::mat4(1.0f), t);
        const glm::mat4 rotate = glm::mat4_cast(q);
        const glm::mat4 scaleM = glm::scale(glm::mat4(1.0f), s);
        return translate * rotate * scaleM;
    }
    const auto& m = std::get<fastgltf::math::fmat4x4>(node.transform);
    glm::mat4 g(1.0f);
    for (int c = 0; c < 4; ++c) {
        const auto col = m.col(static_cast<std::size_t>(c));
        g[c] = glm::vec4(col.x(), col.y(), col.z(), col.w());
    }
    return g;
}

void mergeBounds(glm::vec3 p, glm::vec3& bmin, glm::vec3& bmax) {
    bmin = glm::min(bmin, p);
    bmax = glm::max(bmax, p);
}

bool readTriangleIndices(const fastgltf::Asset& asset, std::size_t accessorIndex, std::vector<uint32_t>& out,
    std::string& err) {
    const auto& acc = asset.accessors[accessorIndex];
    if (acc.type != fastgltf::AccessorType::Scalar) {
        err = "Index accessor must be scalar";
        return false;
    }
    out.resize(acc.count);
    switch (acc.componentType) {
    case fastgltf::ComponentType::UnsignedInt:
        fastgltf::copyFromAccessor<std::uint32_t>(asset, acc, out.data());
        return true;
    case fastgltf::ComponentType::UnsignedShort: {
        std::vector<std::uint16_t> tmp(acc.count);
        fastgltf::copyFromAccessor<std::uint16_t>(asset, acc, tmp.data());
        for (std::size_t i = 0; i < tmp.size(); ++i) {
            out[i] = tmp[i];
        }
        return true;
    }
    case fastgltf::ComponentType::UnsignedByte: {
        std::vector<std::uint8_t> tmp(acc.count);
        fastgltf::copyFromAccessor<std::uint8_t>(asset, acc, tmp.data());
        for (std::size_t i = 0; i < tmp.size(); ++i) {
            out[i] = tmp[i];
        }
        return true;
    }
    default:
        err = "Unsupported index component type";
        return false;
    }
}

std::vector<uint8_t> copyGltfImageBytes(const fastgltf::Asset& asset, const fastgltf::Image& img) {
    return std::visit(
        fastgltf::visitor{
            [](const std::monostate&) -> std::vector<uint8_t> { return {}; },
            [](const fastgltf::sources::Fallback&) -> std::vector<uint8_t> { return {}; },
            [](const fastgltf::sources::URI&) -> std::vector<uint8_t> { return {}; },
            [](const fastgltf::sources::CustomBuffer&) -> std::vector<uint8_t> { return {}; },
            [&](const fastgltf::sources::BufferView& bv) -> std::vector<uint8_t> {
                fastgltf::DefaultBufferDataAdapter adapter;
                const auto span = adapter(asset, bv.bufferViewIndex);
                std::vector<uint8_t> out(span.size());
                if (!out.empty()) {
                    std::memcpy(out.data(), span.data(), span.size());
                }
                return out;
            },
            [&](const fastgltf::sources::Array& arr) -> std::vector<uint8_t> {
                const std::size_t n = arr.bytes.size_bytes();
                std::vector<uint8_t> out(n);
                if (n > 0) {
                    std::memcpy(out.data(), arr.bytes.data(), n);
                }
                return out;
            },
            [&](const fastgltf::sources::Vector& vec) -> std::vector<uint8_t> {
                std::vector<uint8_t> out(vec.bytes.size());
                if (!out.empty()) {
                    std::memcpy(out.data(), vec.bytes.data(), vec.bytes.size());
                }
                return out;
            },
            [&](const fastgltf::sources::ByteView& bv) -> std::vector<uint8_t> {
                std::vector<uint8_t> out(bv.bytes.size());
                if (!out.empty()) {
                    std::memcpy(out.data(), bv.bytes.data(), bv.bytes.size());
                }
                return out;
            },
        },
        img.data);
}

std::vector<uint8_t> decodeAndResizeLayer(const std::vector<uint8_t>& fileBytes, int tw, int th) {
    if (fileBytes.empty() || tw < 1 || th < 1) {
        return {};
    }
    int iw = 0;
    int ih = 0;
    int nc = 0;
    unsigned char* pix = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc*>(fileBytes.data()), static_cast<int>(fileBytes.size()), &iw, &ih, &nc, 4);
    if (!pix || iw < 1 || ih < 1) {
        return {};
    }
    const size_t outPixels = static_cast<size_t>(tw) * static_cast<size_t>(th) * 4u;
    std::vector<uint8_t> out(outPixels);
    for (int ty = 0; ty < th; ++ty) {
        for (int tx = 0; tx < tw; ++tx) {
            const int sx = std::min(iw - 1, tx * iw / std::max(tw, 1));
            const int sy = std::min(ih - 1, ty * ih / std::max(th, 1));
            std::memcpy(&out[static_cast<size_t>((ty * tw + tx) * 4)], &pix[static_cast<size_t>((sy * iw + sx) * 4)], 4);
        }
    }
    stbi_image_free(pix);
    return out;
}

float acquireTextureLayer(const fastgltf::Asset& asset, const fastgltf::Primitive& prim, CityMeshData& acc) {
    if (!prim.materialIndex.has_value()) {
        return 0.0f;
    }
    const auto& mat = asset.materials[prim.materialIndex.value()];
    if (!mat.pbrData.baseColorTexture.has_value()) {
        return 0.0f;
    }
    const std::size_t texIdx = mat.pbrData.baseColorTexture->textureIndex;
    if (texIdx >= asset.textures.size()) {
        return 0.0f;
    }
    const auto& tex = asset.textures[texIdx];
    std::optional<std::size_t> imageIdx = tex.imageIndex;
    if (!imageIdx.has_value() && tex.webpImageIndex.has_value()) {
        imageIdx = tex.webpImageIndex;
    }
    if (!imageIdx.has_value()) {
        return 0.0f;
    }
    if (imageIdx.value() >= asset.images.size()) {
        return 0.0f;
    }

    const auto it = acc.gltfImageToLayer_.find(imageIdx.value());
    if (it != acc.gltfImageToLayer_.end()) {
        return static_cast<float>(it->second);
    }
    // city.glb ships ~100 PNG atlases; keep below typical WebGPU maxTextureArrayLayers (often 256+).
    if (acc.texLayers_.size() >= 256) {
        return 0.0f;
    }

    const std::vector<uint8_t> raw = copyGltfImageBytes(asset, asset.images[imageIdx.value()]);
    std::vector<uint8_t> rgba = decodeAndResizeLayer(raw, acc.texArrayW, acc.texArrayH);
    if (rgba.empty()) {
        return 0.0f;
    }
    const uint32_t layer = static_cast<uint32_t>(acc.texLayers_.size());
    acc.texLayers_.push_back(std::move(rgba));
    acc.gltfImageToLayer_[imageIdx.value()] = layer;
    return static_cast<float>(layer);
}

bool appendPrimitive(const fastgltf::Asset& asset, const fastgltf::Primitive& prim, const glm::mat4& world,
    CityMeshData& acc, std::string& err) {
    if (prim.dracoCompression) {
        err = "Draco-compressed meshes are not supported";
        return false;
    }
    if (prim.type != fastgltf::PrimitiveType::Triangles) {
        return true;
    }

    const auto* posAttr = prim.findAttribute("POSITION");
    if (posAttr == prim.attributes.cend()) {
        return true;
    }

    const auto& posAcc = asset.accessors[posAttr->accessorIndex];
    if (posAcc.type != fastgltf::AccessorType::Vec3 || posAcc.componentType != fastgltf::ComponentType::Float) {
        err = "POSITION must be vec3 float";
        return false;
    }

    const std::size_t vcount = posAcc.count;
    if (vcount == 0) {
        return true;
    }

    std::vector<glm::vec3> localPos(vcount);
    fastgltf::copyFromAccessor<glm::vec3>(asset, posAcc, localPos.data());

    const glm::mat3 rotScale = glm::mat3(world);
    glm::mat3 normalMat = glm::transpose(glm::inverse(rotScale));
    if (!std::isfinite(normalMat[0][0])) {
        normalMat = glm::mat3(1.0f);
    }

    std::vector<glm::vec3> worldPos(vcount);
    for (std::size_t i = 0; i < vcount; ++i) {
        const glm::vec4 wp = world * glm::vec4(localPos[i], 1.0f);
        worldPos[i] = glm::vec3(wp);
        mergeBounds(worldPos[i], acc.boundsMin, acc.boundsMax);
    }

    std::vector<uint32_t> idx;
    if (!prim.indicesAccessor.has_value()) {
        if (vcount % 3 != 0) {
            err = "Non-indexed primitive without multiple-of-3 vertices";
            return false;
        }
        idx.resize(vcount);
        for (std::size_t i = 0; i < vcount; ++i) {
            idx[static_cast<std::size_t>(i)] = static_cast<uint32_t>(i);
        }
    } else {
        if (!readTriangleIndices(asset, prim.indicesAccessor.value(), idx, err)) {
            return false;
        }
    }

    std::vector<glm::vec3> norms(vcount, glm::vec3(0.0f));
    bool usedFileNormals = false;
    const auto* nrmAttr = prim.findAttribute("NORMAL");
    if (nrmAttr != prim.attributes.cend()) {
        const auto& nAcc = asset.accessors[nrmAttr->accessorIndex];
        if (nAcc.type == fastgltf::AccessorType::Vec3 && nAcc.componentType == fastgltf::ComponentType::Float
            && nAcc.count == vcount) {
            std::vector<glm::vec3> ln(vcount);
            fastgltf::copyFromAccessor<glm::vec3>(asset, nAcc, ln.data());
            for (std::size_t i = 0; i < vcount; ++i) {
                glm::vec3 n = glm::normalize(normalMat * ln[i]);
                if (!std::isfinite(n.x) || glm::dot(n, n) < 1e-12f) {
                    n = glm::vec3(0.0f, 1.0f, 0.0f);
                }
                norms[i] = n;
            }
            usedFileNormals = true;
        }
    }

    if (!usedFileNormals) {
        std::fill(norms.begin(), norms.end(), glm::vec3(0.0f));
        for (std::size_t t = 0; t + 2 < idx.size(); t += 3) {
            const uint32_t i0 = idx[t];
            const uint32_t i1 = idx[t + 1];
            const uint32_t i2 = idx[t + 2];
            if (i0 >= vcount || i1 >= vcount || i2 >= vcount) {
                continue;
            }
            const glm::vec3 e1 = worldPos[i1] - worldPos[i0];
            const glm::vec3 e2 = worldPos[i2] - worldPos[i0];
            glm::vec3 fn = glm::cross(e1, e2);
            const float len = glm::length(fn);
            if (len > 1e-12f) {
                fn /= len;
            } else {
                fn = glm::vec3(0.0f, 1.0f, 0.0f);
            }
            norms[i0] += fn;
            norms[i1] += fn;
            norms[i2] += fn;
        }
        for (std::size_t i = 0; i < vcount; ++i) {
            const float len = glm::length(norms[i]);
            if (len > 1e-12f) {
                norms[i] /= len;
            } else {
                norms[i] = glm::vec3(0.0f, 1.0f, 0.0f);
            }
        }
    }

    std::vector<glm::vec2> uvs(vcount, glm::vec2(0.0f));
    std::string uvAttrName = "TEXCOORD_0";
    if (prim.materialIndex.has_value()) {
        const auto& mat = asset.materials[prim.materialIndex.value()];
        if (mat.pbrData.baseColorTexture.has_value()) {
            const std::size_t tc = mat.pbrData.baseColorTexture->texCoordIndex;
            uvAttrName = (tc == 1) ? "TEXCOORD_1" : "TEXCOORD_0";
        }
    }
    const auto* uvAttr = prim.findAttribute(uvAttrName);
    if (uvAttr != prim.attributes.cend()) {
        const auto& uAcc = asset.accessors[uvAttr->accessorIndex];
        if (uAcc.type == fastgltf::AccessorType::Vec2 && uAcc.componentType == fastgltf::ComponentType::Float
            && uAcc.count == vcount) {
            fastgltf::copyFromAccessor<glm::vec2>(asset, uAcc, uvs.data());
        }
    }

    const float texLayer = acquireTextureLayer(asset, prim, acc);

    const uint32_t base = static_cast<uint32_t>(acc.interleaved.size() / static_cast<size_t>(CityMeshData::kFloatsPerVertex));
    for (std::size_t i = 0; i < vcount; ++i) {
        acc.interleaved.push_back(worldPos[i].x);
        acc.interleaved.push_back(worldPos[i].y);
        acc.interleaved.push_back(worldPos[i].z);
        acc.interleaved.push_back(norms[i].x);
        acc.interleaved.push_back(norms[i].y);
        acc.interleaved.push_back(norms[i].z);
        acc.interleaved.push_back(uvs[i].x);
        acc.interleaved.push_back(uvs[i].y);
        acc.interleaved.push_back(texLayer);
    }
    for (uint32_t id : idx) {
        acc.indices.push_back(base + id);
    }
    return true;
}

void appendMesh(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh, const glm::mat4& world, CityMeshData& acc,
    std::string& err) {
    for (const auto& prim : mesh.primitives) {
        if (!appendPrimitive(asset, prim, world, acc, err)) {
            return;
        }
    }
}

void visitNode(const fastgltf::Asset& asset, std::size_t nodeIndex, const glm::mat4& parent, CityMeshData& acc,
    std::string& err) {
    const fastgltf::Node& node = asset.nodes[nodeIndex];
    const glm::mat4 world = parent * nodeLocalToMat4(node);
    if (node.meshIndex.has_value()) {
        appendMesh(asset, asset.meshes[node.meshIndex.value()], world, acc, err);
        if (!err.empty()) {
            return;
        }
    }
    for (std::size_t child : node.children) {
        visitNode(asset, child, world, acc, err);
        if (!err.empty()) {
            return;
        }
    }
}

void loadSceneGraph(const fastgltf::Asset& asset, CityMeshData& acc, std::string& err) {
    if (asset.scenes.empty()) {
        err = "glTF has no scenes";
        return;
    }
    std::size_t sceneIdx = 0;
    if (asset.defaultScene.has_value()) {
        sceneIdx = asset.defaultScene.value();
    }
    if (sceneIdx >= asset.scenes.size()) {
        sceneIdx = 0;
    }
    const fastgltf::Scene& scene = asset.scenes[sceneIdx];
    for (std::size_t root : scene.nodeIndices) {
        visitNode(asset, root, glm::mat4(1.0f), acc, err);
        if (!err.empty()) {
            return;
        }
    }
}

void loadMeshesFlat(const fastgltf::Asset& asset, CityMeshData& acc, std::string& err) {
    for (const auto& mesh : asset.meshes) {
        appendMesh(asset, mesh, glm::mat4(1.0f), acc, err);
        if (!err.empty()) {
            return;
        }
    }
}

} // namespace

void FlyCameraState::resetToBounds(const glm::vec3 bmin, const glm::vec3 bmax) {
    lookSamplePrev = false;
    const glm::vec3 c = 0.5f * (bmin + bmax);
    const glm::vec3 ext = bmax - bmin;
    const float pad = std::max(glm::length(ext) * 0.08f, 1.5f);
    const float rxz = std::max(ext.x, ext.z) * 0.55f + pad;
    position = c + glm::vec3(0.0f, ext.y * 0.12f + 2.0f, rxz);
    const glm::vec3 to = c - position;
    const float len = glm::length(to);
    if (len > 1e-4f) {
        const glm::vec3 dir = to / len;
        yaw = std::atan2(dir.x, dir.z);
        pitch = std::asin(std::clamp(dir.y, -1.0f, 1.0f));
    } else {
        yaw = 0.0f;
        pitch = 0.0f;
    }
}

glm::mat4 flyCameraView(const FlyCameraState& cam) {
    const glm::vec3 forward(std::sin(cam.yaw) * std::cos(cam.pitch), std::sin(cam.pitch),
        std::cos(cam.yaw) * std::cos(cam.pitch));
    const glm::vec3 target = cam.position + forward;
    return glm::lookAt(cam.position, target, glm::vec3(0.0f, 1.0f, 0.0f));
}

void flyCameraUpdate(FlyCameraState& cam, const float dt, SDL_Window* window, const bool imguiWantsKeyboard,
    const bool imguiWantsMouse) {
    (void)window;
    const Uint8* keys = SDL_GetKeyboardState(nullptr);
    const bool vHeld = keys[SDL_SCANCODE_V] != 0;
    const bool applyMouseLook = vHeld && !imguiWantsKeyboard && !imguiWantsMouse;
    if (!applyMouseLook) {
        cam.lookSamplePrev = false;
    } else {
        float mx = 0.0f;
        float my = 0.0f;
        (void)SDL_GetMouseState(&mx, &my);
        if (!cam.lookSamplePrev) {
            cam.lastLookMx = mx;
            cam.lastLookMy = my;
            cam.lookSamplePrev = true;
        } else {
            const float dmx = mx - cam.lastLookMx;
            const float dmy = my - cam.lastLookMy;
            cam.lastLookMx = mx;
            cam.lastLookMy = my;
            // Match common FPS feel: mouse right → look right.
            cam.yaw -= dmx * cam.mouseSensitivity;
            cam.pitch -= dmy * cam.mouseSensitivity;
            constexpr float lim = 1.553343f; // ~89 deg
            cam.pitch = std::clamp(cam.pitch, -lim, lim);
        }
    }

    const glm::vec3 forward(std::sin(cam.yaw) * std::cos(cam.pitch), std::sin(cam.pitch),
        std::cos(cam.yaw) * std::cos(cam.pitch));
    glm::vec3 flatF(forward.x, 0.0f, forward.z);
    if (glm::dot(flatF, flatF) < 1e-8f) {
        flatF = glm::vec3(std::sin(cam.yaw), 0.0f, std::cos(cam.yaw));
    } else {
        flatF = glm::normalize(flatF);
    }
    const glm::vec3 flatR = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), flatF));

    glm::vec3 move(0.0f);
    if (keys[SDL_SCANCODE_W]) {
        move += flatF;
    }
    if (keys[SDL_SCANCODE_S]) {
        move -= flatF;
    }
    if (keys[SDL_SCANCODE_A]) {
        move += flatR;
    }
    if (keys[SDL_SCANCODE_D]) {
        move -= flatR;
    }
    if (keys[SDL_SCANCODE_SPACE]) {
        move.y += 1.0f;
    }
    if (keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT]) {
        move.y -= 1.0f;
    }

    if (glm::dot(move, move) > 1e-8f) {
        move = glm::normalize(move);
        const bool sprint = keys[SDL_SCANCODE_LCTRL] != 0 || keys[SDL_SCANCODE_RCTRL] != 0;
        const float speed = sprint ? cam.moveSpeed : cam.moveSpeed * 0.5f;
        move *= speed * dt;
        cam.position += move;
    }
}

bool loadCityGlbInto(const std::string& path, CityMeshData& out, std::string& errMsg) {
    out.interleaved.clear();
    out.indices.clear();
    out.boundsMin = glm::vec3(std::numeric_limits<float>::max());
    out.boundsMax = glm::vec3(std::numeric_limits<float>::lowest());
    out.texArrayW = 256;
    out.texArrayH = 256;
    out.texLayers_.clear();
    out.gltfImageToLayer_.clear();
    out.texLayers_.push_back(std::vector<uint8_t>(
        static_cast<size_t>(out.texArrayW) * static_cast<size_t>(out.texArrayH) * 4u, 255u));

    auto dataExpected = fastgltf::GltfDataBuffer::FromPath(path);
    if (dataExpected.error() != fastgltf::Error::None) {
        errMsg = std::string("Failed to read file: ") + std::string(fastgltf::getErrorMessage(dataExpected.error()));
        return false;
    }
    fastgltf::GltfDataBuffer data = std::move(dataExpected.get());

    constexpr auto extensions = fastgltf::Extensions::KHR_mesh_quantization | fastgltf::Extensions::KHR_texture_basisu;
    fastgltf::Parser parser(extensions);
    auto assetExpected = parser.loadGltf(
        data, std::filesystem::path(path).parent_path(), fastgltf::Options::None, fastgltf::Category::All);
    if (assetExpected.error() != fastgltf::Error::None) {
        errMsg = std::string("glTF load error: ") + std::string(fastgltf::getErrorMessage(assetExpected.error()));
        return false;
    }
    fastgltf::Asset asset = std::move(assetExpected.get());

    std::string err;
    loadSceneGraph(asset, out, err);
    if (!err.empty()) {
        errMsg = err;
        return false;
    }
    if (out.indices.empty()) {
        out.interleaved.clear();
        out.boundsMin = glm::vec3(std::numeric_limits<float>::max());
        out.boundsMax = glm::vec3(std::numeric_limits<float>::lowest());
        out.texLayers_.clear();
        out.gltfImageToLayer_.clear();
        out.texLayers_.push_back(std::vector<uint8_t>(
            static_cast<size_t>(out.texArrayW) * static_cast<size_t>(out.texArrayH) * 4u, 255u));
        loadMeshesFlat(asset, out, err);
        if (!err.empty()) {
            errMsg = err;
            return false;
        }
    }
    if (out.indices.empty()) {
        errMsg = "No triangle geometry found in city.glb";
        return false;
    }
    if (!std::isfinite(out.boundsMin.x) || out.boundsMax.x < out.boundsMin.x) {
        errMsg = "Invalid bounds";
        return false;
    }
    return true;
}
