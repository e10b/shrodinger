# shrodinger

Standalone TDSE / “atom” playground built on **[wgfx](https://github.com/Vyscosity/wgfx)** (WebGPU + SDL3) and **Dear ImGui**.

This repository is included as a **submodule** from the parent [atoms](https://github.com/Vyscosity/atoms) monorepo under `shrodinger/`.

## Clone

```bash
git clone --recurse-submodules git@github.com:vyscosity/shrodinger.git
cd shrodinger
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Build

```bash
cmake -S . -B out -DCMAKE_BUILD_TYPE=Release
cmake --build out -j
./out/App
```

On macOS, `imgui_impl_wgpu.cpp` is compiled as Objective-C++ (same as atoms).

## Layout

| Path | Role |
|------|------|
| `deps/wgfx` | Graphics library submodule |
| `deps/imgui` | Dear ImGui submodule |
| `third_party/imgui_impl_wgpu.cpp` | WebGPU backend shim compatible with Elie Michel / wgpu-native 0.19 headers |
| `example/` | App entry (`main.cpp`), `Quad` TDSE UI, SDL context |
| `res/` | WGSL shaders and assets |

Optional CMake flags match atoms: `ATOMS_ENABLE_ONNX`, `ATOMS_ENABLE_TORCHSCRIPT`, plus `ONNXRUNTIME_ROOT` / `TORCH_ROOT`.
