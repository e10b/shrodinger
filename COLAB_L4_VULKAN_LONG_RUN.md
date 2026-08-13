# Colab L4 + Vulkan GRMHD Long-Run Procedure

This is the exact procedure verified on 2026-08-12 for running `HarmGpuBench` on a real NVIDIA GPU through WebGPU/wgpu in a Google Colab CLI session. It also records the failed configuration so a successful-looking `llvmpipe` run is not mistaken for GPU execution.

## Result summary

The default Colab GPU image had an NVIDIA CUDA driver and device but lacked the NVIDIA graphics/Vulkan userspace packages. Without them, wgpu silently selected Mesa's CPU renderer:

```text
WebGPU adapter: llvmpipe (LLVM 15.0.7, 256 bits) | vendor= | backend=7 | type=2
```

Here, backend `7` is OpenGL and adapter type `2` is CPU. `nvidia-smi` showed zero GPU memory and no process. A completed render or benchmark in this state is not evidence of NVIDIA GPU execution.

After installing the graphics/Vulkan packages matching Colab's loaded `580.82.07` driver, `vulkaninfo` and wgpu reported:

```text
deviceName = NVIDIA L4
driverInfo = 580.82.07
WebGPU adapter: NVIDIA L4 | vendor=NVIDIA | backend=6 | type=0
```

Backend `6` is Vulkan and type `0` is a discrete GPU. `nvidia-smi` then showed `HarmGpuBench` as a `C+G` process with GPU memory allocated.

## 1. Build the resumable runner locally

The current `HarmGpuBench` supports simulated-time termination, progress diagnostics, and binary restart checkpoints:

```sh
cmake --build build --target HarmGpuBench -j8
```

Important arguments:

| Argument | Meaning |
|---|---|
| `--target-time 10000` | Stop when device-measured simulated time reaches `10,000M` |
| `--substeps 200` | Execute 200 GRMHD steps per host submission batch |
| `--dt 0.03` | Time-step ceiling; the state-dependent CFL reduction chooses the accepted step |
| `--checkpoint-every 25` | Read diagnostics and checkpoint every 25 host batches, or 5,000 steps |
| `--checkpoint FILE` | Atomically replace a binary state checkpoint after every diagnostic batch |
| `--resume FILE` | Restore packed primitives, simulated time, accepted time step, and completed batch count |
| `--high-order` | Enable the current MC-limited reconstruction path |

## 2. Package the source and dependencies

From the repository root:

```sh
COPYFILE_DISABLE=1 tar \
  --exclude='.git' \
  --exclude='build' \
  --exclude='out' \
  --exclude='artifacts' \
  --exclude='*.png' \
  --exclude='*.mp4' \
  --exclude='Porth_2019_ApJS_243_26.pdf' \
  --exclude='deps/wgfx/res' \
  --exclude='deps/wgfx/deps/sdl3/test' \
  --exclude='deps/wgfx/deps/sdl3/examples' \
  --exclude='deps/wgfx/deps/sdl3/docs' \
  --exclude='deps/wgfx/deps/sdl3/Xcode' \
  --exclude='deps/wgfx/deps/sdl3/VisualC' \
  --exclude='deps/wgfx/deps/glm/test' \
  --exclude='deps/wgfx/deps/glm/doc' \
  --exclude='deps/imgui/docs' \
  --exclude='._*' \
  -czf /tmp/shrodinger_harm32_colab.tgz .
```

## 3. Create and populate the L4 session

```sh
/Users/ethan/.local/bin/colab new -s harm-32-t10000-l4 --gpu L4

/Users/ethan/.local/bin/colab upload -s harm-32-t10000-l4 \
  /tmp/shrodinger_harm32_colab.tgz \
  /content/shrodinger_harm32_colab.tgz
```

Use `colab exec -f SCRIPT.py` for the following remote operations. The Python snippets are shown inline for reproducibility.

## 4. Install the matching NVIDIA Vulkan userspace stack

First confirm the loaded driver version:

```python
import subprocess
print(subprocess.check_output(['nvidia-smi'], text=True))
```

The verified VM reported driver `580.82.07`. Install the exact same version of the NVIDIA libraries; do not allow apt to mix the newest `580.x` dependencies with the loaded driver:

```python
import subprocess

packages = [
    'libvulkan1',
    'vulkan-tools',
    'libnvidia-gl-580=580.82.07-0ubuntu1',
    'libnvidia-gpucomp-580=580.82.07-0ubuntu1',
    'libnvidia-compute-580=580.82.07-0ubuntu1',
    'libnvidia-cfg1-580=580.82.07-0ubuntu1',
    'libnvidia-decode-580=580.82.07-0ubuntu1',
]

subprocess.run(['apt-get', 'update'], check=True)
subprocess.run(
    ['apt-get', 'install', '-y', '--no-install-recommends', *packages],
    check=True,
)
subprocess.run(['vulkaninfo', '--summary'], check=True)
```

Why all packages are pinned: installing only `libnvidia-gl-580=580.82.07-0ubuntu1` initially failed because apt selected `libnvidia-gpucomp-580` version `580.178.04`. Merely unpacking `.deb` files into a private directory also failed because it did not reproduce the package's alternatives/symlink configuration; the Vulkan loader could find the ICD but returned `ERROR_INCOMPATIBLE_DRIVER`.

The installation does not replace the running kernel driver. It supplies the missing matching userspace graphics and Vulkan libraries inside the disposable VM.

## 5. Build remotely

```python
import pathlib
import subprocess

root = pathlib.Path('/content/shrodinger_harm32')
subprocess.run(['rm', '-rf', str(root)], check=True)
root.mkdir(parents=True)
subprocess.run(
    ['tar', '-xzf', '/content/shrodinger_harm32_colab.tgz', '-C', str(root)],
    check=True,
)
subprocess.run(
    ['cmake', '-S', str(root), '-B', str(root / 'build'), '-DCMAKE_BUILD_TYPE=Release'],
    check=True,
)
subprocess.run(
    ['cmake', '--build', str(root / 'build'), '--target', 'HarmGpuBench', '-j2'],
    check=True,
)
```

## 6. Require a real-GPU smoke test

Run one small case and inspect the new adapter line before starting a paid long run:

```sh
cd /content/shrodinger_harm32
./build/HarmGpuBench \
  --grid 32 \
  --frames 1 \
  --substeps 1 \
  --dt 0.03 \
  --high-order \
  --checkpoint-every 1 \
  --out /content/harm32_gpu_smoke.md
```

Required evidence:

```text
WebGPU adapter: NVIDIA L4 | vendor=NVIDIA | backend=6 | type=0
```

Also run `nvidia-smi` while the process is alive. It must list `HarmGpuBench` and nonzero GPU memory. Abort if the adapter is `llvmpipe`, backend is `7`, type is `2`, or `nvidia-smi` has no process.

## 7. Launch the long run detached

Use a detached subprocess because a Colab CLI websocket disconnect must not kill the simulation:

```python
import pathlib
import subprocess

root = pathlib.Path('/content/shrodinger_harm32')
checkpoint = root / 'harm_32_t10000.chk'
report = root / 'harm_32_t10000.md'
log = pathlib.Path('/content/harm_32_t10000.log')

command = [
    'stdbuf', '-oL', '-eL', str(root / 'build/HarmGpuBench'),
    '--grid', '32',
    '--target-time', '10000',
    '--substeps', '200',
    '--dt', '0.03',
    '--high-order',
    '--checkpoint-every', '25',
    '--checkpoint', str(checkpoint),
    '--out', str(report),
]

with log.open('wb') as output:
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

pathlib.Path('/content/harm_32_t10000.pid').write_text(str(process.pid))
print(process.pid)
```

To resume after an interrupted VM process while `/content` still exists, append:

```text
--resume /content/shrodinger_harm32/harm_32_t10000.chk
```

Colab `/content` is ephemeral. For a genuinely interruption-resistant production campaign, periodically download the checkpoint or copy it to mounted Drive/object storage. A local binary checkpoint is not sufficient if Colab reclaims the entire VM.

## 8. Poll and validate

Read lines beginning with `PROGRESS` from `/content/harm_32_t10000.log`. Each contains:

```text
PROGRESS t=... dt=... frames=... wall_s=... fail=... divB_L1=...
```

Do not judge success only by process survival. Require bounded recovery failures, magnetic divergence, floors, and global conserved diagnostics.

The verified L4 attempt produced:

| Simulated time | Accepted dt | `divB L1` | Recovery failure fraction |
|---:|---:|---:|---:|
| `4.38584M` | `7.00768e-4` | `2.72305e-5` | `0` |
| `24.90784M` | `3.90696e-4` | `2.80581e-5` | `0` |
| `40.73392M` | `2.43560e-4` | `4.08489e-4` | `0` |
| `48.10112M` | `2.48700e-4` | `5.01672e-3` | `0` |

The run reached `48.1M` in about 69 seconds, but `divB L1` grew approximately 184 times above its initial value and was accelerating. It was stopped because the magnetic evolution was numerically invalid even though primitive recovery had not failed.

The preserved local evidence is under `artifacts/colab_32_t10000_attempt/` when available. That directory is ignored by Git because it contains generated checkpoints and logs.

## 9. Shut down the paid session

Always release the VM:

```sh
/Users/ethan/.local/bin/colab stop -s harm-32-t10000-l4
/Users/ethan/.local/bin/colab sessions
```

The final command should report no active sessions.

## A100 qualification

An A100 session was tested before the complete matching-package installation was discovered. Its default adapter was also `llvmpipe`. An isolated unpack of the NVIDIA Vulkan libraries failed, but that is not equivalent to the successful full apt installation used on L4. Therefore:

- real NVIDIA/Vulkan execution is proven on the L4;
- the old A100-render claim was not instrumented and must not be treated as proof of A100 GPU execution;
- A100 Vulkan compatibility after the complete package procedure remains unverified.
