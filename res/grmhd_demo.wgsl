struct VertexInput {
    @location(0) position: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

struct GrmhdUniform {
    orbital: vec4f, // w:view mode
    tuning: vec4f,  // x:color scale, y:reserved, z:zoom, w:spin proxy
    render: vec4f,  // x:time, y:aspect, z:reserved, w:mode marker
    pan: vec4f,     // xy:pan, z:horizon radius
    tdse: vec4f,    // x:grid, y:domain half, z:mag loop, w:diffusion
};

struct GrmhdCell {
    value: vec4f, // x:density, y:sigma, z:beta, w:reserved
    flow: vec4f,  // x:radial speed, y:divB, z:speed, w:reserved
};

@group(0) @binding(0) var<uniform> u: GrmhdUniform;
@group(0) @binding(1) var<storage, read> field: array<GrmhdCell>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(input.position, 1.0);
    out.uv = input.position.xy;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let gridSize = max(2, i32(round(u.tdse.x)));
    let domainHalf = max(u.tdse.y, 0.001);
    let zoom = max(u.tuning.z, 0.0001);
    let aspect = max(u.render.y, 0.0001);
    let viewMode = i32(round(u.orbital.w));

    var p = input.uv;
    p.x *= aspect;
    p /= zoom;
    p -= u.pan.xy;

    let r = length(p);
    let nx = 0.5 * (p.x / domainHalf + 1.0);
    let ny = 0.5 * (p.y / domainHalf + 1.0);
    if (nx < 0.0 || nx > 1.0 || ny < 0.0 || ny > 1.0) {
        return vec4f(0.006, 0.007, 0.010, 1.0);
    }

    let ix = clamp(i32(floor(nx * f32(gridSize - 1))), 0, gridSize - 1);
    let iy = clamp(i32(floor(ny * f32(gridSize - 1))), 0, gridSize - 1);
    let sample = field[iy * gridSize + ix];
    let cell = sample.value;

    var scalar = clamp(cell.x, 0.0, 1.0);
    var color = densityPalette(scalar);
    if (viewMode == 1) {
        scalar = clamp(cell.y, 0.0, 1.0);
        color = magneticPalette(scalar);
    } else if (viewMode == 2) {
        scalar = clamp(cell.z, 0.0, 1.0);
        color = betaPalette(scalar);
    } else if (viewMode == 3) {
        scalar = clamp(sample.flow.x, 0.0, 1.0);
        color = mix(vec3f(0.06, 0.40, 0.84), vec3f(1.00, 0.42, 0.12), scalar);
    } else if (viewMode == 4) {
        scalar = clamp(sample.flow.y, 0.0, 1.0);
        color = mix(vec3f(0.02, 0.03, 0.04), vec3f(1.0, 0.16, 0.38), scalar);
    }

    let horizon = max(u.pan.z, 0.01);
    if (r < horizon) {
        let edge = smoothstep(horizon * 0.76, horizon, r);
        color = mix(vec3f(0.0), vec3f(0.018, 0.012, 0.008), edge);
        scalar = 1.0;
    } else {
        let diskLine = exp(-abs(p.y) * 0.18) * smoothstep(horizon * 1.2, horizon * 2.6, r);
        let ring = smoothstep(0.012, 0.0, abs(r - horizon * 1.45));
        color += vec3f(0.85, 0.55, 0.25) * (0.13 * diskLine + 0.32 * ring);
    }

    let bg = vec3f(0.006, 0.007, 0.010);
    let vignette = smoothstep(domainHalf * 1.05, domainHalf * 0.15, r);
    return vec4f(mix(bg, color, clamp(0.18 + scalar * 0.95, 0.0, 1.0)) * (0.42 + 0.58 * vignette), 1.0);
}

fn densityPalette(x: f32) -> vec3f {
    let a = vec3f(0.020, 0.024, 0.045);
    let b = vec3f(0.070, 0.220, 0.360);
    let c = vec3f(0.980, 0.390, 0.110);
    let d = vec3f(1.000, 0.930, 0.550);
    if (x < 0.42) {
        return mix(a, b, smoothstep(0.0, 0.42, x));
    }
    if (x < 0.78) {
        return mix(b, c, smoothstep(0.42, 0.78, x));
    }
    return mix(c, d, smoothstep(0.78, 1.0, x));
}

fn magneticPalette(x: f32) -> vec3f {
    return mix(vec3f(0.025, 0.060, 0.090), vec3f(0.180, 0.940, 0.820), pow(clamp(x, 0.0, 1.0), 0.65));
}

fn betaPalette(x: f32) -> vec3f {
    let cold = vec3f(0.080, 0.120, 0.680);
    let mid = vec3f(0.150, 0.800, 0.520);
    let hot = vec3f(1.000, 0.760, 0.180);
    if (x < 0.5) {
        return mix(cold, mid, smoothstep(0.0, 0.5, x));
    }
    return mix(mid, hot, smoothstep(0.5, 1.0, x));
}
