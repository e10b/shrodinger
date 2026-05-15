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

set(IMGUI_WGPU_SRC third_party/imgui_impl_wgpu.cpp)
set(IMGUI_WGPU_INCLUDE_DIR deps/imgui/backends)
if(EMSCRIPTEN)
	set(IMGUI_WGPU_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/imgui_webgpu_compat")
	set(IMGUI_WGPU_SRC "${IMGUI_WGPU_INCLUDE_DIR}/imgui_impl_wgpu.cpp")
	file(MAKE_DIRECTORY "${IMGUI_WGPU_INCLUDE_DIR}")

	file(READ "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui_impl_wgpu.cpp" IMGUI_WGPU_CPP)
	string(REPLACE
		"#if defined(__EMSCRIPTEN__) && defined(IMGUI_IMPL_WEBGPU_BACKEND_WGPU)"
		"#if defined(__EMSCRIPTEN__) && defined(IMGUI_IMPL_WEBGPU_BACKEND_WGPU) && defined(WGPU_STRLEN)"
		IMGUI_WGPU_CPP "${IMGUI_WGPU_CPP}"
	)
	string(REPLACE
		"#if defined(IMGUI_IMPL_WEBGPU_BACKEND_WGPU) && !defined(__EMSCRIPTEN__) && !defined(WGPU_STRLEN)"
		"#if defined(IMGUI_IMPL_WEBGPU_BACKEND_WGPU) && !defined(WGPU_STRLEN)"
		IMGUI_WGPU_CPP "${IMGUI_WGPU_CPP}"
	)
	file(WRITE "${IMGUI_WGPU_SRC}" "${IMGUI_WGPU_CPP}")

	file(READ "${CMAKE_CURRENT_SOURCE_DIR}/deps/imgui/backends/imgui_impl_wgpu.h" IMGUI_WGPU_H)
	string(REPLACE
		"#elif defined(IMGUI_IMPL_WEBGPU_BACKEND_WGPU)"
		"#elif defined(IMGUI_IMPL_WEBGPU_BACKEND_WGPU) && !defined(__EMSCRIPTEN__)"
		IMGUI_WGPU_H "${IMGUI_WGPU_H}"
	)
	file(WRITE "${IMGUI_WGPU_INCLUDE_DIR}/imgui_impl_wgpu.h" "${IMGUI_WGPU_H}")
endif()

set(IMGUI_SRC
	deps/imgui/imgui.cpp
	deps/imgui/imgui_draw.cpp
	deps/imgui/imgui_tables.cpp
	deps/imgui/imgui_widgets.cpp
	deps/imgui/imgui_demo.cpp
	deps/imgui/backends/imgui_impl_sdl3.cpp
	${IMGUI_WGPU_SRC}
)

add_executable(App ${APP_SRC} ${WGFX_LIB_SRC} ${IMGUI_SRC} third_party/stb_image_impl.cpp example/main.cpp)
set_source_files_properties(${IMGUI_WGPU_SRC} PROPERTIES COMPILE_FLAGS "-x objective-c++")

if(EMSCRIPTEN)
	set_target_properties(App PROPERTIES
		SUFFIX ".html"
		LINK_FLAGS "-s ASYNCIFY -s ALLOW_MEMORY_GROWTH=1 --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/res@res"
	)
endif()

if(DEV_MODE AND NOT EMSCRIPTEN)
	target_compile_definitions(App PRIVATE
		RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res"
	)
else()
	target_compile_definitions(App PRIVATE
		RESOURCE_DIR="./res"
	)
endif()

target_link_libraries(App PRIVATE webgpu sdl2webgpu SDL3-static glm::glm fastgltf::fastgltf)
target_include_directories(App PRIVATE
	"deps/wgfx/src"
	"deps/wgfx/deps"
	"deps/wgfx/deps/fastgltf/include"
	"deps/wgfx/deps/sdl3/include"
	"deps/imgui"
	"${IMGUI_WGPU_INCLUDE_DIR}"
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
