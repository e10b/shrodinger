# Colab A100 Render Runbook

This records the Colab CLI workflow used to render the 96^3 HARM GRMHD clip on an A100.

The successful artifact was:

```sh
artifacts/harm_grmhd_grid96_640x360_2s_substeps240_a100.mp4
```

## Prerequisites

Install and authenticate the Colab CLI first. In this repo, the binary was available at:

```sh
/Users/ethan/.local/bin/colab
```

Check sessions:

```sh
/Users/ethan/.local/bin/colab status
/Users/ethan/.local/bin/colab sessions
```

## Create An A100 Session

```sh
/Users/ethan/.local/bin/colab new -s harm-a100 --gpu A100
```

## Package Local Changes And Dependencies

The public repo can be cloned on Colab, but the render used local uncommitted changes plus local dependency bundles. This also avoids Colab/GitHub submodule timeouts.

From the repo root:

```sh
git diff --binary > /tmp/shrodinger_local_changes.patch
git ls-files --others --exclude-standard -z | \
  xargs -0 -I{} sh -c 'git diff --no-index --binary /dev/null "$1" || true' sh {} \
  >> /tmp/shrodinger_local_changes.patch
```

Package `wgfx` without heavy sample assets:

```sh
COPYFILE_DISABLE=1 tar \
  --exclude='.git' \
  --exclude='res' \
  --exclude='deps/sdl3/test' \
  --exclude='deps/sdl3/examples' \
  --exclude='deps/sdl3/docs' \
  --exclude='deps/sdl3/Xcode' \
  --exclude='deps/sdl3/VisualC' \
  --exclude='deps/glm/test' \
  --exclude='deps/glm/doc' \
  --exclude='._*' \
  -czf /tmp/wgfx_trimmed_for_colab.tgz \
  -C deps wgfx
```

Package `imgui`:

```sh
COPYFILE_DISABLE=1 tar \
  --exclude='.git' \
  --exclude='._*' \
  -czf /tmp/imgui_trimmed_for_colab.tgz \
  -C deps imgui
```

## Upload Payloads

```sh
/Users/ethan/.local/bin/colab upload -s harm-a100 \
  /tmp/shrodinger_local_changes.patch \
  /content/shrodinger_local_changes.patch

/Users/ethan/.local/bin/colab upload -s harm-a100 \
  /tmp/wgfx_trimmed_for_colab.tgz \
  /content/wgfx_trimmed_for_colab.tgz

/Users/ethan/.local/bin/colab upload -s harm-a100 \
  /tmp/imgui_trimmed_for_colab.tgz \
  /content/imgui_trimmed_for_colab.tgz
```

## Create The Remote Runner

Create `/tmp/run_a100_render_96_no_submodules.sh` locally:

```sh
cat > /tmp/run_a100_render_96_no_submodules.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /content
rm -rf /content/shrodinger
git clone --branch blackhole --depth 1 https://github.com/e10b/shrodinger /content/shrodinger

cd /content/shrodinger
rm -rf deps/imgui deps/wgfx
tar -xzf /content/imgui_trimmed_for_colab.tgz -C deps
tar -xzf /content/wgfx_trimmed_for_colab.tgz -C deps
find deps/imgui deps/wgfx -name '._*' -delete

if [ -s /content/shrodinger_local_changes.patch ]; then
  git apply --whitespace=nowarn /content/shrodinger_local_changes.patch
fi

git status --short
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2

mkdir -p artifacts
./build/App \
  --headless \
  --grid 96 \
  --frames 120 \
  --resolution 640x360 \
  --substeps 240 \
  --dt 0.0002 \
  --view 5 \
  --video artifacts/harm_grmhd_grid96_640x360_2s_substeps240_a100.mp4

ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 \
  artifacts/harm_grmhd_grid96_640x360_2s_substeps240_a100.mp4
SH

chmod +x /tmp/run_a100_render_96_no_submodules.sh
```

Upload it:

```sh
/Users/ethan/.local/bin/colab upload -s harm-a100 \
  /tmp/run_a100_render_96_no_submodules.sh \
  /content/run_a100_render_96_no_submodules.sh
```

## Launch Detached

Use a detached job because the Colab CLI websocket can disconnect while the render continues.

```sh
cat > /tmp/colab_relaunch_no_submodules.py <<'PY'
import os, subprocess

os.makedirs('/content', exist_ok=True)
os.chdir('/content')
subprocess.run('chmod +x /content/run_a100_render_96_no_submodules.sh', shell=True, check=True)
subprocess.run(
    'nohup /content/run_a100_render_96_no_submodules.sh '
    '> /content/a100_render_96.log 2>&1 '
    '& echo $! > /content/a100_render_96.pid',
    shell=True,
    check=True,
)
print(open('/content/a100_render_96.pid').read().strip())
PY

/Users/ethan/.local/bin/colab exec -s harm-a100 \
  -f /tmp/colab_relaunch_no_submodules.py \
  --timeout 30
```

## Poll Progress

Create a small polling script:

```sh
cat > /tmp/colab_tail_a100.py <<'PY'
import pathlib

pid = pathlib.Path('/content/a100_render_96.pid')
log = pathlib.Path('/content/a100_render_96.log')
video = pathlib.Path('/content/shrodinger/artifacts/harm_grmhd_grid96_640x360_2s_substeps240_a100.mp4')

print('pid', pid.read_text().strip() if pid.exists() else 'none')
if log.exists():
    data = log.read_text(errors='replace')
    frames = [line for line in data.splitlines() if line.startswith('Rendered frame')]
    print(frames[-1] if frames else 'no rendered-frame line yet')
    print('done', 'Done rendering video.' in data)
    print(data[-5000:])
else:
    print('no log yet')

print('video_exists', video.exists(), 'size', video.stat().st_size if video.exists() else 0)
PY
```

Poll every few minutes:

```sh
/Users/ethan/.local/bin/colab exec -s harm-a100 \
  -f /tmp/colab_tail_a100.py \
  --timeout 30
```

Completion is indicated by:

```text
Rendered frame 120 / 120
done True
Done rendering video.
width=640
height=360
r_frame_rate=60/1
duration=2.000000
nb_frames=120
```

## Download The Finished MP4

```sh
mkdir -p artifacts

/Users/ethan/.local/bin/colab download -s harm-a100 \
  /content/shrodinger/artifacts/harm_grmhd_grid96_640x360_2s_substeps240_a100.mp4 \
  artifacts/harm_grmhd_grid96_640x360_2s_substeps240_a100.mp4
```

## Verify Locally

```sh
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 \
  artifacts/harm_grmhd_grid96_640x360_2s_substeps240_a100.mp4
```

Expected:

```text
width=640
height=360
r_frame_rate=60/1
duration=2.000000
nb_frames=120
```

## Make A Preview Sheet

```sh
rm -rf artifacts/grid96_640x360_2s_substeps240_a100_frames
mkdir -p artifacts/grid96_640x360_2s_substeps240_a100_frames

ffmpeg -y \
  -i artifacts/harm_grmhd_grid96_640x360_2s_substeps240_a100.mp4 \
  -vf "select='eq(n,0)+eq(n,30)+eq(n,60)+eq(n,90)+eq(n,119)'" \
  -vsync 0 \
  artifacts/grid96_640x360_2s_substeps240_a100_frames/frame_%02d.png

ffmpeg -y \
  -pattern_type glob \
  -i 'artifacts/grid96_640x360_2s_substeps240_a100_frames/frame_*.png' \
  -filter_complex "tile=5x1:padding=8:margin=8:color=black" \
  -frames:v 1 \
  -update 1 \
  artifacts/grid96_640x360_2s_substeps240_a100_contact.png
```

## Notes

- This render used the real Colab A100 runtime and the repo's headless WebGPU path.
- It did not use AMR block refinement in the interactive GPU hot path.
- At `96^3`, the log reported one phi slab: `phi slabs 1, slab phi 96`.
- Colab's WebGPU path reported a `128 MiB` storage-buffer binding cap.
- The render was slow despite A100 hardware because this path is dominated by the current WebGPU/headless submit/readback behavior.
- If the session disappears, `/content` artifacts disappear too. Download immediately after `Done rendering video.`
