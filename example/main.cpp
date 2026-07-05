#define WGPU_IMPLEMENTATION
#include <wgfx.h>
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_wgpu.h"
#include "context.h"
#include "clock.h"
#include "harm_config.h"
#include "quad.h"

#include <iostream>
#include <string>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

int main(int argc, char** argv)
{
	bool headless = false;
	std::string videoOut = "output.mp4";
	int renderWidth = 1920;
	int renderHeight = 1080;
	int renderFrames = 600;
	bool playAnim = false;
	int customGridSize = 0;

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--headless") {
			headless = true;
		} else if (arg == "--play") {
			playAnim = true;
		} else if (arg == "--video" && i + 1 < argc) {
			videoOut = argv[++i];
		} else if (arg == "--frames" && i + 1 < argc) {
			renderFrames = std::stoi(argv[++i]);
		} else if (arg == "--resolution" && i + 1 < argc) {
			std::string res = argv[++i];
			auto xpos = res.find('x');
			if (xpos != std::string::npos) {
				renderWidth = std::stoi(res.substr(0, xpos));
				renderHeight = std::stoi(res.substr(xpos + 1));
			}
		} else if (arg == "--grid" && i + 1 < argc) {
			customGridSize = std::stoi(argv[++i]);
			if (customGridSize > harm::Config::kTiledMaxGrid) {
				std::cout << "Requested --grid " << customGridSize
					<< " exceeds the current two-slab HARM storage path; clamping to "
					<< harm::Config::kTiledMaxGrid << ".\n";
			} else if (customGridSize > harm::Config::kSingleBufferMaxGrid) {
				std::cout << "Requested --grid " << customGridSize
					<< " will use tiled two-slab HARM storage.\n";
			}
		}
	}
	Context& context = Context::Instance(headless);

	Quad& quad = Quad::Instance();
	if (customGridSize > 0) {
		quad.setMaxGridSize(customGridSize);
	}
	quad.setHarmMode(playAnim);

	if (!headless) {
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
	}

	wgfx::width = renderWidth;
	wgfx::height = renderHeight;
	wgfx::ColorTexture* color = new wgfx::ColorTexture(!headless);
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
		float fpsTimer = 0.0f;
		float frameTimeAccumulator = 0.0f;
		int fpsFrameCount = 0;
		int renderFrameCount = 0;
		bool headless;
		int renderWidth;
		int renderHeight;
		int renderFrames;
		wgpu::Buffer readbackBuffer;
		FILE* ffmpegPipe;
	};

	wgpu::Buffer readbackBuffer = nullptr;
	FILE* ffmpegPipe = nullptr;
	if (headless) {
		wgpu::BufferDescriptor bufDesc = {};
		bufDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
		bufDesc.size = renderWidth * renderHeight * 4;
		bufDesc.mappedAtCreation = false;
		readbackBuffer = wgpu::Device(wgfx::device).createBuffer(bufDesc);

		std::string cmd = "ffmpeg -y -f rawvideo -pix_fmt bgra -s " + std::to_string(renderWidth) + "x" + std::to_string(renderHeight) + " -r 60 -i - -c:v libx264 -pix_fmt yuv420p " + videoOut;
		ffmpegPipe = popen(cmd.c_str(), "w");
		std::cout << "Starting headless render to " << videoOut << " (" << renderWidth << "x" << renderHeight << ")\n";
	}

	AppState* appState = new AppState{ &context, &quad, color, pass, uiPass, 0.0f, 0.0f, 0, 0, headless, renderWidth, renderHeight, renderFrames, readbackBuffer, ffmpegPipe };

	auto loop = [](void* arg) {
		AppState* state = static_cast<AppState*>(arg);
		static Clock clock;
		float dt = clock.restart();
		
		// Accumulate frame times for averaging
		state->frameTimeAccumulator += dt;
		state->fpsFrameCount++;
		state->renderFrameCount++;
		state->fpsTimer += dt;
		
		// Update FPS display every 0.5 seconds using average frame time
		if (state->fpsTimer > 0.5f)
		{
			float averageFrameTime = state->frameTimeAccumulator / state->fpsFrameCount;
			state->context->fps(averageFrameTime);
			state->fpsTimer = 0;
			state->frameTimeAccumulator = 0;
			state->fpsFrameCount = 0;
		}

		state->context->update();
		if (!state->headless) {
			ImGui_ImplWGPU_NewFrame();
			ImGui_ImplSDL3_NewFrame();
			ImGui::NewFrame();
			state->quad->drawImGuiPanel();
			ImGui::Render();
		}

		if (!state->headless) {
			wgfx::touch(state->color);
		}
		// One encoder per frame: compute + scene + ImGui passes encode here; frame() finishes it.
		wgfx::start();

		// Run HARM GPU compute before the render pass so the evolved storage
		// buffer is ready for the fragment shader.
		state->quad->dispatchCompute3d();
		state->quad->dispatchComputeHarm();

		// Render the HARM fullscreen view.
		state->pass->prepare();
			state->quad->render(dt);
		state->pass->draw(state->quad->pipeline);
		state->pass->end();

		if (!state->headless) {
			state->uiPass->prepare();
			ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), (WGPURenderPassEncoder)state->uiPass->renderPass);
			state->uiPass->end();
			state->context->draw();
			state->quad->afterFrameSubmit();
		} else {
			// Submit the render passes that were encoded into wgfx::encoder
			if (wgfx::encoder) {
				wgpu::CommandBuffer cmd = wgfx::encoder.finish();
				wgfx::queue.submit(1, &cmd);
				wgfx::encoder.release();
				wgfx::encoder = nullptr;
				state->quad->afterFrameSubmit();
			}

			// Encode texture to buffer copy
			wgpu::CommandEncoder encoder = wgpu::Device(wgfx::device).createCommandEncoder();
			wgpu::ImageCopyTexture src = {};
			src.texture = state->color->colorTexture;
			wgpu::ImageCopyBuffer dst = {};
			dst.buffer = state->readbackBuffer;
			dst.layout.bytesPerRow = state->renderWidth * 4;
			dst.layout.rowsPerImage = state->renderHeight;
			wgpu::Extent3D ext = {(uint32_t)state->renderWidth, (uint32_t)state->renderHeight, 1};
			encoder.copyTextureToBuffer(src, dst, ext);
			wgpu::CommandBuffer cmd = encoder.finish();
			wgpu::Queue(wgfx::queue).submit(1, &cmd);

			// Map buffer and wait
			bool done = false;
			wgpuBufferMapAsync((WGPUBuffer)state->readbackBuffer, WGPUMapMode_Read, 0, state->renderWidth * state->renderHeight * 4, [](WGPUBufferMapAsyncStatus status, void* userdata) {
				*static_cast<bool*>(userdata) = true;
			}, &done);
			
			while (!done) {
				wgpuDevicePoll((WGPUDevice)wgfx::device, true, nullptr);
			}
			
			const void* data = state->readbackBuffer.getConstMappedRange(0, state->renderWidth * state->renderHeight * 4);
			fwrite(data, 1, state->renderWidth * state->renderHeight * 4, state->ffmpegPipe);
			state->readbackBuffer.unmap();
			
			std::cout << "Rendered frame " << state->renderFrameCount << " / " << state->renderFrames << "\r" << std::flush;
			if (state->renderFrameCount >= state->renderFrames) {
				std::cout << "\nDone rendering video.\n";
				pclose(state->ffmpegPipe);
				state->context->close = true;
			}
		}

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
	if (!headless) {
		ImGui_ImplWGPU_Shutdown();
		ImGui_ImplSDL3_Shutdown();
		ImGui::DestroyContext();
	}
#endif
}
