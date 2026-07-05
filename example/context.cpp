#include "context.h"
#include "imgui_impl_sdl3.h"

#include <algorithm>
#include <iostream>

namespace {

bool initHeadlessWebGpu()
{
	wgfx::instance = wgpuCreateInstance(nullptr);
	std::cout << "Requesting headless adapter...\n";
	wgpu::RequestAdapterOptions adapterOpts = {};
	wgfx::adapter = wgfx::instance.requestAdapter(adapterOpts);
	std::cout << "Got adapter: " << wgfx::adapter << std::endl;
	if (!wgfx::adapter) {
		return false;
	}

	wgpu::SupportedLimits adapterLimits;
	wgfx::adapter.getLimits(&adapterLimits);
	std::cout << "Adapter supported maxStorageBufferBindingSize: "
		<< adapterLimits.limits.maxStorageBufferBindingSize << " bytes\n";

	wgpu::RequiredLimits requiredLimits = {};
	requiredLimits.limits = adapterLimits.limits;
	requiredLimits.limits.maxStorageBufferBindingSize = std::min<uint64_t>(
		adapterLimits.limits.maxStorageBufferBindingSize, 1073741824ull);
	requiredLimits.limits.maxBufferSize = std::min<uint64_t>(
		adapterLimits.limits.maxBufferSize, 1073741824ull);

	wgpu::DeviceDescriptor deviceDesc = {};
	deviceDesc.label = "Headless HARM Device";
	deviceDesc.requiredLimits = &requiredLimits;
	deviceDesc.defaultQueue.nextInChain = nullptr;
	deviceDesc.defaultQueue.label = "Headless queue";
	wgfx::device = wgfx::adapter.requestDevice(deviceDesc);
	std::cout << "Got device: " << wgfx::device << std::endl;
	if (!wgfx::device) {
		return false;
	}

	wgpu::SupportedLimits deviceLimits;
	wgfx::device.getLimits(&deviceLimits);
	wgfx::deviceLimits = deviceLimits.limits;
	std::cout << "Device maxStorageBufferBindingSize: "
		<< wgfx::deviceLimits.maxStorageBufferBindingSize << " bytes\n";
	wgfx::queue = wgfx::device.getQueue();
	wgfx::surfaceFormat = wgpu::TextureFormat::BGRA8UnormSrgb;
	return static_cast<bool>(wgfx::queue);
}

} // namespace

Context::Context(bool headless)
{
	this->headless = headless;
	if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO)) { std::cout << "Couldn't init SDL!\n"; }
	
	Uint32 windowFlags = SDL_WINDOW_RESIZABLE;
	if (headless) {
		windowFlags |= SDL_WINDOW_HIDDEN;
	}
	
	window = SDL_CreateWindow("Learn WebGPU", 1280, 720, windowFlags);
	if (window) {
		wgfx::init(wgfx::getSurface(window));
	} else if (headless && initHeadlessWebGpu()) {
		std::cout << "Using headless WebGPU without an SDL window.\n";
	} else {
		std::cerr << "Failed to create SDL window or headless WebGPU device.\n";
		close = true;
	}
}

void Context::update()
{
	if (headless || !window) return;
	SDL_Event event;
	while (SDL_PollEvent(&event))
	{
		ImGui_ImplSDL3_ProcessEvent(&event);
		switch (event.type)
		{
		case SDL_EVENT_WINDOW_RESIZED:
		{
			wgfx::initSurface();
			//wgfx::initDepth();
			//gonna need a different way to do this


			int width, height;
			SDL_GetWindowSize(window, &width, &height);

			//proj = glm::perspective(glm::radians(50.0f), float(width) / float(height), 0.1f, 100.0f);
			//pipeline.updateUniform(projUniform, glm::value_ptr(proj));
		}
		break;

		case SDL_EVENT_QUIT:
			close = true;
			break;

		case SDL_EVENT_KEY_DOWN:
			if (event.key.key == SDLK_ESCAPE) {
				close = true; // Close the application if Escape is pressed
			}
			break;

		case SDL_EVENT_WINDOW_EXPOSED:
			wgfx::initSurface();
			break;

		case SDL_EVENT_MOUSE_WHEEL:
			wheelDeltaY += event.wheel.y;
			break;
		}
	}
}

void Context::draw()
{
	wgfx::frame();
}
