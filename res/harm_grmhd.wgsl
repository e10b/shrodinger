struct VertexInput {
    @location(0) position: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

struct HarmUniform {
    orbital: vec4f, // w:view mode
    tuning: vec4f,  // x:color scale, y:r_in, z:zoom, w:spin
    render: vec4f,  // x:time, y:aspect
    pan: vec4f,     // xy:pan, z:r_in
    tdse: vec4f,    // x:grid, y:r_out, z:mag loop, w:dt
};

struct HarmPrim {
    state0: vec4f, // x:rho, y:u, z:U1, w:U3
    state1: vec4f, // x:B1, y:B3, z:fail, w:reserved
};

@group(0) @binding(0) var<uniform> u: HarmUniform;
@group(0) @binding(1) var<storage, read> field: array<HarmPrim>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(input.position, 1.0);
    out.uv = input.position.xy;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let n = max(2, i32(round(u.tdse.x)));
    let rin = max(u.tuning.y, 0.001);
    let rout = max(u.tdse.y, rin + 0.1);
    let zoom = max(u.tuning.z, 0.0001);
    let aspect = max(u.render.y, 0.0001);
    let viewMode = i32(round(u.orbital.w));

    if (viewMode == 5) {
        return renderShadowImage(input.uv, n, rin, rout, zoom, aspect);
    }

    var p = input.uv;
    p.x *= aspect;
    p /= zoom;
    p -= u.pan.xy;

    let r = length(p);
    if (r < rin || r > rout) {
        let hole = smoothstep(rin * 0.72, rin, r);
        let bg = mix(vec3f(0.0), vec3f(0.006, 0.008, 0.012), hole);
        return vec4f(bg, 1.0);
    }

    var phi = atan2(p.y, p.x);
    if (phi < 0.0) {
        phi += 6.283185307179586;
    }

    let xr = clamp((log(r) - log(rin)) / max(log(rout) - log(rin), 0.001), 0.0, 0.9999);
    let xp = clamp(phi / 6.283185307179586, 0.0, 0.9999);
    let ir = clamp(i32(floor(xr * f32(n))), 0, n - 1);
    let ip = clamp(i32(floor(xp * f32(n))), 0, n - 1);
    let sample = field[ip * n + ir];
    let rho = max(sample.state0.x, 1e-8);
    let uu = max(sample.state0.y, 1e-9);
    let ur = sample.state0.z;
    let uphi = sample.state0.w;
    let b1 = sample.state1.x;
    let b3 = sample.state1.y;
    let fail = sample.state1.z;
    let b2 = b1 * b1 + b3 * b3;
    let sigma = clamp(b2 / rho * 8.0, 0.0, 1.0);
    let beta = clamp(log(1.0 + ((1.0 / 3.0) * uu) / max(0.5 * b2, 1e-8)) / 6.0, 0.0, 1.0);
    let speed = clamp(sqrt(ur * ur + uphi * uphi), 0.0, 1.0);

    var scalar = clamp(log(1.0 + rho * u.tuning.x * 40.0) / log(1.0 + u.tuning.x * 40.0), 0.0, 1.0);
    var color = densityPalette(scalar);
    if (viewMode == 1) {
        scalar = sigma;
        color = magneticPalette(scalar);
    } else if (viewMode == 2) {
        scalar = beta;
        color = betaPalette(scalar);
    } else if (viewMode == 3) {
        scalar = clamp(0.5 + 0.5 * ur / 0.4, 0.0, 1.0);
        color = mix(vec3f(0.05, 0.32, 0.86), vec3f(1.0, 0.44, 0.11), scalar);
    } else if (viewMode == 4) {
        scalar = clamp(fail + 0.25 * speed, 0.0, 1.0);
        color = mix(vec3f(0.02, 0.03, 0.04), vec3f(1.0, 0.08, 0.30), scalar);
    }

    let orbitLines = 0.08 * smoothstep(0.97, 1.0, sin(phi * 18.0 + log(r) * 12.0 - u.render.x * 1.7) * 0.5 + 0.5);
    let horizon = smoothstep(rin * 1.3, rin, r);
    color += orbitLines * vec3f(0.25, 0.42, 0.70) * (1.0 - horizon);
    color = mix(vec3f(0.0), color, 1.0 - horizon);

    let bg = vec3f(0.006, 0.008, 0.012);
    let fade = smoothstep(rout, rout * 0.82, r);
    return vec4f(mix(bg, color, clamp(0.16 + scalar, 0.0, 1.0)) * fade, 1.0);
}

fn densityPalette(x: f32) -> vec3f {
    if (x < 0.35) {
        return mix(vec3f(0.012, 0.018, 0.038), vec3f(0.060, 0.210, 0.340), smoothstep(0.0, 0.35, x));
    }
    if (x < 0.78) {
        return mix(vec3f(0.060, 0.210, 0.340), vec3f(0.980, 0.360, 0.090), smoothstep(0.35, 0.78, x));
    }
    return mix(vec3f(0.980, 0.360, 0.090), vec3f(1.0, 0.94, 0.54), smoothstep(0.78, 1.0, x));
}

fn magneticPalette(x: f32) -> vec3f {
    return mix(vec3f(0.025, 0.045, 0.070), vec3f(0.120, 0.980, 0.840), pow(clamp(x, 0.0, 1.0), 0.7));
}

fn betaPalette(x: f32) -> vec3f {
    if (x < 0.5) {
        return mix(vec3f(0.090, 0.130, 0.700), vec3f(0.140, 0.780, 0.520), smoothstep(0.0, 0.5, x));
    }
    return mix(vec3f(0.140, 0.780, 0.520), vec3f(1.0, 0.720, 0.160), smoothstep(0.5, 1.0, x));
}

fn sampleHarmDisk(pos: vec2f, n: i32, rin: f32, rout: f32) -> HarmPrim {
    let r = clamp(length(pos), rin * 1.001, rout * 0.999);
    var phi = atan2(pos.y, pos.x);
    if (phi < 0.0) {
        phi += 6.283185307179586;
    }

    let xr = clamp((log(r) - log(rin)) / max(log(rout) - log(rin), 0.001), 0.0, 0.9999);
    let xp = clamp(phi / 6.283185307179586, 0.0, 0.9999);
    let ir = clamp(i32(floor(xr * f32(n))), 0, n - 1);
    let ip = clamp(i32(floor(xp * f32(n))), 0, n - 1);
    return field[ip * n + ir];
}

fn firePalette(x: f32) -> vec3f {
    let v = clamp(x, 0.0, 1.0);
    if (v < 0.32) {
        return mix(vec3f(0.030, 0.000, 0.000), vec3f(0.560, 0.030, 0.000), smoothstep(0.0, 0.32, v));
    }
    if (v < 0.70) {
        return mix(vec3f(0.560, 0.030, 0.000), vec3f(1.000, 0.390, 0.000), smoothstep(0.32, 0.70, v));
    }
    return mix(vec3f(1.000, 0.390, 0.000), vec3f(1.000, 0.970, 0.300), smoothstep(0.70, 1.0, v));
}

fn hash21(p: vec2f) -> f32 {
    let q = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
    return fract(sin(q.x + q.y) * 43758.5453);
}

fn valueNoise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let a = hash21(i);
    let b = hash21(i + vec2f(1.0, 0.0));
    let c = hash21(i + vec2f(0.0, 1.0));
    let d = hash21(i + vec2f(1.0, 1.0));
    let u2 = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u2.x), mix(c, d, u2.x), u2.y);
}

fn fbm(p: vec2f) -> f32 {
    var q = p;
    var amp = 0.5;
    var sum = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        sum += amp * valueNoise(q);
        q = q * 2.07 + vec2f(13.1, 7.7);
        amp *= 0.52;
    }
    return sum;
}

fn renderShadowImage(uv: vec2f, n: i32, rin: f32, rout: f32, zoom: f32, aspect: f32) -> vec4f {
    var p = uv;
    p.x *= aspect;
    p /= max(zoom, 0.0001);
    p -= u.pan.xy;

    let b = length(p);
    let spin = clamp(u.tuning.w, -0.98, 0.98);
    let shadowRadius = rin * (1.92 - 0.10 * abs(spin));
    let criticalRadius = rin * (2.78 - 0.16 * abs(spin));
    let photonWidth = rin * 0.030;
    let outerImage = min(rout * 0.40, criticalRadius * 5.5);

    let lensP = p;
    let imageB = length(lensP);
    let dir = select(vec2f(1.0, 0.0), lensP / max(imageB, 1e-4), imageB > 1e-4);
    let basePhi = atan2(dir.y, dir.x);
    let frameDrag = spin * (1.10 / max(imageB / rin, 0.55));
    let observer = normalize(vec2f(-0.72, 0.44));

    var scalarSum = 0.0;
    var colorSum = vec3f(0.0);
    var weightSum = 0.0;
    for (var k = 0; k < 9; k = k + 1) {
        let lane = f32(k) - 4.0;
        let laneWeight = exp(-0.5 * lane * lane / 4.2);
        let vertical = lane * rin * 0.18;
        let impact = length(lensP + dir * vertical * 0.08);
        let lensedRadius = clamp(
            impact + 0.54 * criticalRadius * criticalRadius / max(impact + shadowRadius * 0.55 + abs(vertical) * 0.20, 0.06),
            rin * 1.04,
            rout * 0.98);
        let diskPhi = basePhi + frameDrag + 0.72 * log(max(lensedRadius / rin, 1.0)) + lane * 0.035;
        let diskPos = vec2f(cos(diskPhi), sin(diskPhi)) * lensedRadius;

        let sample = sampleHarmDisk(diskPos, n, rin, rout);
        let rho = max(sample.state0.x, 1e-8);
        let heat = max(sample.state0.y, 1e-8);
        let ur = sample.state0.z;
        let uphi = sample.state0.w;
        let b1 = sample.state1.x;
        let b3 = sample.state1.y;
        let mag = sqrt(max(b1 * b1 + b3 * b3, 0.0));
        let tangent = normalize(vec2f(-sin(diskPhi), cos(diskPhi)));
        let beta = clamp(abs(uphi) * 0.70 + abs(ur) * 0.08 + 0.12, 0.0, 0.78);
        let beam = clamp(dot(tangent, observer), -1.0, 1.0);
        let doppler = pow(clamp(1.0 / max(1.0 - beta * beam, 0.25), 0.24, 3.6), 1.80);
        let redshift = sqrt(clamp(1.0 - rin / max(lensedRadius, rin * 1.10), 0.10, 1.0));
        let logr = log(max(lensedRadius / rin, 1.001));
        let local = fbm(vec2f(logr * 4.4 + 0.14 * lane - u.render.x * 0.05, diskPhi * 3.4 + u.render.x * 0.08));
        let fine = fbm(vec2f(logr * 12.0 - u.render.x * 0.14, diskPhi * 9.0 + lane * 0.9));
        let shearPhase = diskPhi - 2.65 * logr + 0.15 * u.render.x;
        let arm2 = pow(smoothstep(0.25, 0.98, sin(2.0 * shearPhase + 1.0 * local) * 0.5 + 0.5), 2.6);
        let arm3 = pow(smoothstep(0.38, 0.99, sin(3.0 * shearPhase - 0.55 + 1.4 * fine) * 0.5 + 0.5), 2.2);
        let streaks = pow(smoothstep(0.36, 0.98, sin(7.0 * diskPhi - 8.0 * logr + 1.2 * local) * 0.5 + 0.5), 2.6);
        let turbulence = 0.22 + 0.58 * local + 0.26 * fine + 1.20 * arm2 + 0.82 * arm3 + 0.46 * streaks;
        let plume = 0.70 + 0.85 * smoothstep(-0.20, 0.85, dot(tangent, observer));
        let annulus = exp(-pow((imageB - criticalRadius * 1.02) / (criticalRadius * 0.34), 2.0));
        let innerFilaments = exp(-pow((imageB - criticalRadius * 0.74) / (criticalRadius * 0.20), 2.0));
        let outerHaze = exp(-pow((imageB - criticalRadius * 1.70) / (criticalRadius * 0.95), 2.0));
        let ringWindow =
            smoothstep(shadowRadius * 0.98, criticalRadius * 0.95, imageB) *
            smoothstep(outerImage, criticalRadius * 1.0, imageB) *
            (0.05 + 1.05 * annulus + 0.55 * innerFilaments + 0.42 * outerHaze);
        let plasma = log(1.0 + u.tuning.x * (64.0 * rho + 36.0 * heat + 16.0 * mag));
        let opticalDepth = clamp((rho * 26.0 + heat * 10.0) * laneWeight, 0.0, 1.0);
        let emission = plasma * doppler * redshift * ringWindow * turbulence * plume * laneWeight * (0.14 + opticalDepth);
        let sampleScalar = clamp(emission / 10.5, 0.0, 1.0);
        scalarSum += sampleScalar * laneWeight;
        colorSum += firePalette(pow(sampleScalar, 0.74)) * sampleScalar * laneWeight;
        weightSum += laneWeight;
    }

    let scalar = clamp(scalarSum / max(weightSum, 1e-4), 0.0, 1.0);
    var color = colorSum / max(weightSum, 1e-4);
    let photon = exp(-pow((b - criticalRadius) / photonWidth, 2.0));
    let ringTexture = 0.55 + 0.45 * fbm(vec2f(atan2(p.y, p.x) * 8.0 + u.render.x * 0.05, b * 2.2));
    color += photon * ringTexture * vec3f(1.0, 0.48, 0.04) * 0.78;

    let shadowShape = length(p);
    let shadow = smoothstep(shadowRadius * 0.86, shadowRadius * 1.02, shadowShape);
    let centralGlow = vec3f(0.002, 0.0, 0.0);
    color = mix(centralGlow, color, shadow);

    let bg = vec3f(0.001, 0.0, 0.0);
    let vignette = smoothstep(outerImage * 1.15, criticalRadius * 0.75, b);
    let bloom = firePalette(scalar) * scalar * 0.18;
    let redFloor = vec3f(0.055, 0.002, 0.0) * smoothstep(outerImage * 0.95, criticalRadius * 1.2, b) * (0.35 + 0.45 * fbm(p * 0.45));
    return vec4f(mix(bg, redFloor + color + bloom, clamp(vignette + photon * 0.18, 0.0, 1.0)), 1.0);
}
