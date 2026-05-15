#define WGPU_IMPLEMENTATION
#include <wgfx.h>
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_wgpu.h"
#include "context.h"
#include "clock.h"
#include "quad_circle.h"

#include <SDL3/SDL.h>
#include <memory>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

int main()
{
	Context& context = Context::Instance();

	Quad& quad = Quad::Instance();

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplSDL3_InitForOther(context.window);
	ImGui_ImplWGPU_InitInfo init_info = {};
	init_info.Device = (WGPUDevice)wgfx::device;
	init_info.NumFramesInFlight = 2;
	init_info.RenderTargetFormat = (WGPUTextureFormat)wgfx::surfaceFormat;
	init_info.DepthStencilFormat = WGPUTextureFormat_Undefined;
	ImGui_ImplWGPU_Init(&init_info);

	wgfx::ColorTexture* color = new wgfx::ColorTexture();
	wgfx::RenderPass* pass = new wgfx::RenderPass();
	wgfx::RenderPass* uiPass = new wgfx::RenderPass();
	pass->addTarget(color);
	uiPass->addTarget(color);
	pass->setClear({ 0.0f, 0.0f, 0.0f, 1.0f });
	uiPass->shouldClear = false;

	struct AppState {
		Context* context;
		Quad* quad;
		wgfx::ColorTexture* color;
		wgfx::RenderPass* pass;
		wgfx::RenderPass* uiPass;
		std::unique_ptr<wgfx::DepthTexture> depthTex;
		int depthW = -1;
		int depthH = -1;
		float fpsTimer = 0.0f;
		float frameTimeAccumulator = 0.0f;
		int frameCount = 0;
	};

	AppState* appState = new AppState{ &context, &quad, color, pass, uiPass };

	auto loop = [](void* arg) {
		AppState* state = static_cast<AppState*>(arg);
		static Clock clock;
		float dt = clock.restart();
		
		// Accumulate frame times for averaging
		state->frameTimeAccumulator += dt;
		state->frameCount++;
		state->fpsTimer += dt;
		
		// Update FPS display every 0.5 seconds using average frame time
		if (state->fpsTimer > 0.5f)
		{
			float averageFrameTime = state->frameTimeAccumulator / state->frameCount;
			state->context->fps(averageFrameTime);
			state->fpsTimer = 0;
			state->frameTimeAccumulator = 0;
			state->frameCount = 0;
		}

		state->context->update();
		ImGui_ImplWGPU_NewFrame();
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();
		state->quad->drawImGuiPanel();
		ImGui::Render();

		int winW = 0;
		int winH = 0;
		SDL_GetWindowSize(state->context->window, &winW, &winH);
		if (winW < 1) {
			winW = 1;
		}
		if (winH < 1) {
			winH = 1;
		}
		// Depth attachment must only be active when the bound pipeline has matching
		// depth-stencil state (city mesh). Orbital / 2D / 3D TDSE pipelines use useDepth=false.
		if (state->quad->isCityWalkMode()) {
			if (!state->depthTex || winW != state->depthW || winH != state->depthH) {
				state->depthW = winW;
				state->depthH = winH;
				state->depthTex = std::make_unique<wgfx::DepthTexture>();
			}
			state->pass->depth = state->depthTex.get();
			state->pass->depth->useDepth = true;
		} else {
			state->pass->depth = nullptr;
		}

		wgfx::touch(state->color);
		// One encoder per frame: compute + scene + ImGui passes encode here; frame() finishes it.
		wgfx::start();

		// Run GPU compute (FDTD step) — must happen BEFORE the render pass
		// so the updated waveB storage buffer is ready for the fragment shader.
		state->quad->dispatchCompute3d();

		if (state->quad->isCityWalkMode()) {
			state->pass->setClear({ 0.45f, 0.62f, 0.88f, 1.0f });
		} else {
			state->pass->setClear({ 0.0f, 0.0f, 0.0f, 1.0f });
		}

		// Render the fullscreen quad with analytic sphere ray tracing in fragment WGSL.
		state->pass->prepare();
			state->quad->render(dt);
		if (state->quad->shouldDrawMainSceneGeometry()) {
			state->pass->draw(state->quad->pipeline);
		}
		state->pass->end();

		state->uiPass->prepare();
		ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), (WGPURenderPassEncoder)state->uiPass->renderPass);
		state->uiPass->end();

		state->context->draw();

#ifdef __EMSCRIPTEN__
		if (state->context->close) {
			emscripten_cancel_main_loop();
		}
#endif
	};

#ifdef __EMSCRIPTEN__
	emscripten_set_main_loop_arg(loop, appState, 0, true);
#else
	while (!context.close)
	{
		loop(appState);
	}
	ImGui_ImplWGPU_Shutdown();
	ImGui_ImplSDL3_Shutdown();
	ImGui::DestroyContext();
#endif
}
