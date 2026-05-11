cmake_minimum_required(VERSION 3.5...3.25)
project(shrodinger VERSION 0.1.0 LANGUAGES CXX C)

# Graphics stack: Vyscosity/wgfx (WebGPU + SDL3 + helpers). Init submodules with:
#   git submodule update --init --recursive

option(DEV_MODE "Load resources from the source tree (edit shaders without copying)" ON)
option(ATOMS_ENABLE_ONNX "Enable ONNX Runtime integration for TDSE inference" OFF)
option(ATOMS_ENABLE_TORCHSCRIPT "Enable TorchScript/libtorch integration for TDSE inference" OFF)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation root")
set(TORCH_ROOT "" CACHE PATH "Path to libtorch installation root")

add_subdirectory(deps/wgfx)

set(APP_SRC
	example/context.cpp
	example/clock.cpp
)
file(GLOB_RECURSE WGFX_LIB_SRC CONFIGURE_DEPENDS "deps/wgfx/src/*.cpp")
set(IMGUI_SRC
	deps/imgui/imgui.cpp
	deps/imgui/imgui_draw.cpp
	deps/imgui/imgui_tables.cpp
	deps/imgui/imgui_widgets.cpp
	deps/imgui/imgui_demo.cpp
	deps/imgui/backends/imgui_impl_sdl3.cpp
	third_party/imgui_impl_wgpu.cpp
)

add_executable(App ${APP_SRC} ${WGFX_LIB_SRC} ${IMGUI_SRC} third_party/stb_image_impl.cpp example/main.cpp)
set_source_files_properties(third_party/imgui_impl_wgpu.cpp PROPERTIES COMPILE_FLAGS "-x objective-c++")

if(DEV_MODE)
	target_compile_definitions(App PRIVATE
		RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res"
	)
else()
	target_compile_definitions(App PRIVATE
		RESOURCE_DIR="./res"
	)
endif()

target_link_libraries(App PRIVATE webgpu sdl2webgpu SDL3-static glm::glm)
target_include_directories(App PRIVATE
	"deps/wgfx/src"
	"deps/wgfx/deps"
	"deps/wgfx/deps/sdl3/include"
	"deps/imgui"
	"deps/imgui/backends"
	"third_party"
	"example"
)
target_compile_definitions(App PRIVATE IMGUI_IMPL_WEBGPU_BACKEND_WGPU)

if(ATOMS_ENABLE_ONNX)
	find_path(ONNXRUNTIME_INCLUDE_DIR
		NAMES onnxruntime_cxx_api.h
		HINTS ${ONNXRUNTIME_ROOT}
		PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session onnxruntime/include
	)
	find_library(ONNXRUNTIME_LIBRARY
		NAMES onnxruntime
		HINTS ${ONNXRUNTIME_ROOT}
		PATH_SUFFIXES lib lib64
	)
	if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
		message(FATAL_ERROR "ATOMS_ENABLE_ONNX=ON but ONNX Runtime not found. Set ONNXRUNTIME_ROOT.")
	endif()
	target_include_directories(App PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
	target_link_libraries(App PRIVATE ${ONNXRUNTIME_LIBRARY})
	target_compile_definitions(App PRIVATE ATOMS_ENABLE_ONNX=1)
endif()

if(ATOMS_ENABLE_TORCHSCRIPT)
	if(TORCH_ROOT)
		list(PREPEND CMAKE_PREFIX_PATH "${TORCH_ROOT}")
	endif()
	find_package(Torch REQUIRED)
	target_link_libraries(App PRIVATE ${TORCH_LIBRARIES})
	target_compile_definitions(App PRIVATE ATOMS_ENABLE_TORCHSCRIPT=1)
endif()

target_copy_webgpu_binaries(App)

set_target_properties(App PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
)
