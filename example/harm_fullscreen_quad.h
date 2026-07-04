#pragma once

#include <memory>
#include <vector>

#include "wgfx.h"

namespace harm {

class FullscreenQuad {
public:
    void init() {
        const std::vector<float> vertices = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f
        };
        const std::vector<uint16_t> indices = { 0, 1, 2, 0, 2, 3 };

        vbo_.reset(wgfx::createVertexBuffer(vertices));
        vbo_->setTopology(PrimitiveTopology::TriangleList);
        vbo_->setAttribute(0, wgfx::vec3f, 0);
        ibo_.reset(wgfx::createIndexBuffer(indices));
    }

    void bind(wgfx::Pipeline* pipeline) {
        pipeline->setVertexBuffer(vbo_.get());
        pipeline->setIndexBuffer(ibo_.get());
    }

    wgfx::VertexBuffer* vertexBuffer() const { return vbo_.get(); }
    wgfx::IndexBuffer* indexBuffer() const { return ibo_.get(); }

private:
    std::unique_ptr<wgfx::VertexBuffer> vbo_;
    std::unique_ptr<wgfx::IndexBuffer> ibo_;
};

} // namespace harm
