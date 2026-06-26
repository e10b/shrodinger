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
    render: vec4f,  // x:time, y:aspect, z:inclination, w:yaw
    pan: vec4f,     // xy:pan, z:r_in
    tdse: vec4f,    // x:n1, y:r_out, z:n2, w:n3
};

struct HarmPrim {
    state0: vec4f, // x:rho, y:u, z:U1, w:U2
    state1: vec4f, // x:U3, y:B1, z:B2, w:B3
    state2: vec4f, // x:fail, yzw:reserved
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
    let n2 = max(2, i32(round(u.tdse.z)));
    let n3 = max(2, i32(round(u.tdse.w)));
    let rin = max(u.tuning.y, 0.001);
    let rout = max(u.tdse.y, rin + 0.1);
    let zoom = max(u.tuning.z, 0.0001);
    let aspect = max(u.render.y, 0.0001);
    let viewMode = i32(round(u.orbital.w));

    if (viewMode == 5) {
        return renderShadowImage(input.uv, n, n2, n3, rin, rout, zoom, aspect);
    }
    if (viewMode == 6) {
        return renderVerticalSlice(input.uv, n, n2, n3, rin, rout, zoom, aspect);
    }
    if (viewMode == 7) {
        return renderThetaPhiSlice(input.uv, n, n2, n3, rin, rout);
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
    let ip = clamp(i32(floor(xp * f32(n3))), 0, n3 - 1);
    let it = n2 / 2;
    let sample = field[(ip * n2 + it) * n + ir];
    let rho = max(sample.state0.x, 1e-8);
    let uu = max(sample.state0.y, 1e-9);
    let ur = sample.state0.z;
    let uphi = sample.state1.x;
    let b1 = sample.state1.y;
    let b2p = sample.state1.z;
    let b3 = sample.state1.w;
    let fail = sample.state2.x;
    let b2 = b1 * b1 + b2p * b2p + b3 * b3;
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

fn sampleHarmCell(irIn: i32, itIn: i32, ipIn: i32, n1: i32, n2: i32, n3: i32) -> HarmPrim {
    let ir = clamp(irIn, 0, n1 - 1);
    let it = clamp(itIn, 0, n2 - 1);
    var ip = ipIn % n3;
    if (ip < 0) {
        ip += n3;
    }
    return field[(ip * n2 + it) * n1 + ir];
}

fn mixHarmPrim(a: HarmPrim, b: HarmPrim, t: f32) -> HarmPrim {
    var out: HarmPrim;
    out.state0 = mix(a.state0, b.state0, t);
    out.state1 = mix(a.state1, b.state1, t);
    out.state2 = mix(a.state2, b.state2, t);
    return out;
}

fn sampleHarmGrid(fr: f32, ft: f32, fp: f32, n1: i32, n2: i32, n3: i32) -> HarmPrim {
    let gr = clamp(fr * f32(n1) - 0.5, 0.0, f32(max(n1 - 2, 0)));
    let gt = clamp(ft * f32(n2) - 0.5, 0.0, f32(max(n2 - 2, 0)));
    let gp = fp * f32(n3) - 0.5;
    let ir0 = i32(floor(gr));
    let it0 = i32(floor(gt));
    let ip0 = i32(floor(gp));
    let wr = fract(gr);
    let wt = fract(gt);
    let wp = fract(gp);

    let c000 = sampleHarmCell(ir0,     it0,     ip0,     n1, n2, n3);
    let c100 = sampleHarmCell(ir0 + 1, it0,     ip0,     n1, n2, n3);
    let c010 = sampleHarmCell(ir0,     it0 + 1, ip0,     n1, n2, n3);
    let c110 = sampleHarmCell(ir0 + 1, it0 + 1, ip0,     n1, n2, n3);
    let c001 = sampleHarmCell(ir0,     it0,     ip0 + 1, n1, n2, n3);
    let c101 = sampleHarmCell(ir0 + 1, it0,     ip0 + 1, n1, n2, n3);
    let c011 = sampleHarmCell(ir0,     it0 + 1, ip0 + 1, n1, n2, n3);
    let c111 = sampleHarmCell(ir0 + 1, it0 + 1, ip0 + 1, n1, n2, n3);

    let c00 = mixHarmPrim(c000, c100, wr);
    let c10 = mixHarmPrim(c010, c110, wr);
    let c01 = mixHarmPrim(c001, c101, wr);
    let c11 = mixHarmPrim(c011, c111, wr);
    let c0 = mixHarmPrim(c00, c10, wt);
    let c1 = mixHarmPrim(c01, c11, wt);
    return mixHarmPrim(c0, c1, wp);
}

fn harmBrightness(sample: HarmPrim) -> f32 {
    let rho = max(sample.state0.x, 1e-8);
    let heat = max(sample.state0.y, 1e-9);
    let b1 = sample.state1.y;
    let b2 = sample.state1.z;
    let b3 = sample.state1.w;
    let mag = sqrt(max(b1 * b1 + b2 * b2 + b3 * b3, 0.0));
    return clamp(log(1.0 + u.tuning.x * (44.0 * rho + 25.0 * heat + 9.0 * mag)) / 4.8, 0.0, 1.0);
}

fn renderVerticalSlice(uv: vec2f, n1: i32, n2: i32, n3: i32, rin: f32, rout: f32, zoom: f32, aspect: f32) -> vec4f {
    var p = uv;
    p.x *= aspect;
    p /= max(zoom, 0.0001);
    p -= u.pan.xy;

    let cyl = abs(p.x);
    let z = p.y;
    let r = sqrt(cyl * cyl + z * z);
    if (r < rin || r > rout) {
        return vec4f(vec3f(0.002, 0.001, 0.001), 1.0);
    }

    let theta = clamp(atan2(cyl, z), 0.08 * 3.141592653589793, 0.92 * 3.141592653589793);
    let phi = select(0.0, 3.141592653589793, p.x < 0.0) + 0.18 * u.render.x;
    let xr = clamp((log(r) - log(rin)) / max(log(rout) - log(rin), 0.001), 0.0, 0.9999);
    let xt = clamp((theta / 3.141592653589793 - 0.08) / 0.84, 0.0, 0.9999);
    let xp = fract(phi / 6.283185307179586);
    let ir = clamp(i32(floor(xr * f32(n1))), 0, n1 - 1);
    let it = clamp(i32(floor(xt * f32(n2))), 0, n2 - 1);
    let ip = clamp(i32(floor(xp * f32(n3))), 0, n3 - 1);
    let sample = sampleHarmCell(ir, it, ip, n1, n2, n3);

    let scalar = harmBrightness(sample);
    let midplane = exp(-pow((theta - 0.5 * 3.141592653589793) / 0.22, 2.0));
    let funnel = smoothstep(0.18, 0.02, abs(theta - 0.5 * 3.141592653589793));
    var color = firePalette(pow(scalar, 0.72));
    color += vec3f(0.08, 0.20, 0.26) * funnel;
    color *= 0.20 + 0.95 * midplane + 0.45 * scalar;

    let horizon = smoothstep(rin * 1.45, rin * 1.02, r);
    color = mix(color, vec3f(0.0), horizon);
    return vec4f(color, 1.0);
}

fn renderThetaPhiSlice(uv: vec2f, n1: i32, n2: i32, n3: i32, rin: f32, rout: f32) -> vec4f {
    let sx = clamp(uv.x * 0.5 + 0.5, 0.0, 0.9999);
    let sy = clamp(uv.y * 0.5 + 0.5, 0.0, 0.9999);
    let r = clamp(rout * 0.34, rin * 1.2, rout * 0.86);
    let theta = mix(0.08 * 3.141592653589793, 0.92 * 3.141592653589793, sy);
    let phi = 6.283185307179586 * sx + 0.10 * u.render.x;

    let xr = clamp((log(r) - log(rin)) / max(log(rout) - log(rin), 0.001), 0.0, 0.9999);
    let ir = clamp(i32(floor(xr * f32(n1))), 0, n1 - 1);
    let it = clamp(i32(floor(sy * f32(n2))), 0, n2 - 1);
    let ip = clamp(i32(floor(fract(phi / 6.283185307179586) * f32(n3))), 0, n3 - 1);
    let sample = sampleHarmCell(ir, it, ip, n1, n2, n3);

    let scalar = harmBrightness(sample);
    let equator = exp(-pow((theta - 0.5 * 3.141592653589793) / 0.26, 2.0));
    var color = firePalette(pow(scalar, 0.70)) * (0.18 + 1.05 * equator);
    let gridLine = max(
        smoothstep(0.985, 1.0, sin(sx * 6.283185307179586 * 12.0) * 0.5 + 0.5),
        smoothstep(0.990, 1.0, sin(sy * 3.141592653589793 * 8.0) * 0.5 + 0.5));
    color += 0.025 * gridLine * vec3f(0.35, 0.20, 0.08);
    return vec4f(color, 1.0);
}

fn sampleHarmDisk(pos: vec2f, n1: i32, n2: i32, n3: i32, rin: f32, rout: f32, thetaOffset: f32) -> HarmPrim {
    let r = clamp(length(pos), rin * 1.001, rout * 0.999);
    var phi = atan2(pos.y, pos.x);
    if (phi < 0.0) {
        phi += 6.283185307179586;
    }

    let xr = clamp((log(r) - log(rin)) / max(log(rout) - log(rin), 0.001), 0.0, 0.9999);
    let xp = clamp(phi / 6.283185307179586, 0.0, 0.9999);
    let xt = clamp(0.5 + thetaOffset, 0.0, 0.9999);
    let ir = clamp(i32(floor(xr * f32(n1))), 0, n1 - 1);
    let it = clamp(i32(floor(xt * f32(n2))), 0, n2 - 1);
    let ip = clamp(i32(floor(xp * f32(n3))), 0, n3 - 1);
    return field[(ip * n2 + it) * n1 + ir];
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

fn sampleHarmVolume(pos: vec3f, n1: i32, n2: i32, n3: i32, rin: f32, rout: f32) -> HarmPrim {
    let r = clamp(length(pos), rin * 1.001, rout * 0.999);
    let theta = acos(clamp(pos.z / max(r, 1e-5), -1.0, 1.0));
    var phi = atan2(pos.y, pos.x);
    if (phi < 0.0) {
        phi += 6.283185307179586;
    }

    let xr = clamp((log(r) - log(rin)) / max(log(rout) - log(rin), 0.001), 0.0, 0.9999);
    let xt = clamp((theta / 3.141592653589793 - 0.08) / 0.84, 0.0, 0.9999);
    let xp = clamp(phi / 6.283185307179586, 0.0, 0.9999);
    return sampleHarmGrid(xr, xt, xp, n1, n2, n3);
}

fn renderShadowImage(uv: vec2f, n: i32, n2: i32, n3: i32, rin: f32, rout: f32, zoom: f32, aspect: f32) -> vec4f {
    var p = uv;
    p.x *= aspect;
    p /= max(zoom, 0.0001);
    p -= u.pan.xy;

    let b = length(p);
    let spin = clamp(u.tuning.w, -0.98, 0.98);
    let shadowRadius = rin * (1.92 - 0.10 * abs(spin));
    let criticalRadius = rin * (2.78 - 0.16 * abs(spin));
    let photonWidth = rin * 0.030;
    let outerImage = min(rout * 0.48, criticalRadius * 6.6);
    let inc = clamp(u.render.z, 0.05, 1.45);
    let yaw = u.render.w;
    let ci = cos(inc);
    let si = sin(inc);
    let cy = cos(yaw);
    let sy = sin(yaw);
    let baseObserver = normalize(vec3f(0.0, -si, ci));
    let observerDisk = normalize(vec3f(
        baseObserver.x * cy - baseObserver.y * sy,
        baseObserver.x * sy + baseObserver.y * cy,
        baseObserver.z));

    var color = vec3f(0.0);
    var alpha = 0.0;
    var scalarMax = 0.0;
    let zMax = outerImage * 1.15;
    let raySteps = 104.0;
    let rayJitter = hash21(uv * vec2f(733.3, 421.7) + vec2f(u.render.x * 0.013, -u.render.x * 0.019));
    let ds = (2.0 * zMax) / raySteps;
    var escapedShadow = 1.0;

    for (var k = 0; k < 104; k = k + 1) {
        let t = -zMax + (f32(k) + rayJitter) * ds;
        let screen = vec3f(p.x, p.y, t);
        let screenB = length(screen.xy);
        let bend = 0.86 * criticalRadius * criticalRadius / max(screenB * screenB + t * t * 0.32 + shadowRadius * shadowRadius, 0.1);
        let toward = select(vec2f(0.0), -screen.xy / max(screenB, 1e-4), screenB > 1e-4);
        let dragged = screen.xy + toward * bend + vec2f(-screen.y, screen.x) * (spin * 0.048 * bend / max(screenB, 0.8));
        let tiltedPos = vec3f(
            dragged.x,
            dragged.y * ci - screen.z * si,
            dragged.y * si + screen.z * ci);
        let diskPos = vec3f(
            tiltedPos.x * cy - tiltedPos.y * sy,
            tiltedPos.x * sy + tiltedPos.y * cy,
            tiltedPos.z);
        let r = length(diskPos);
        if (r < shadowRadius * 0.98 && t > -zMax * 0.75) {
            escapedShadow = 0.0;
        }
        if (r > rin * 1.04 && r < rout * 0.995 && alpha < 0.985) {
            let theta = acos(clamp(diskPos.z / max(r, 1e-5), -1.0, 1.0));
            let midplane = exp(-pow((theta - 0.5 * 3.141592653589793) / 0.25, 2.0));
            if (midplane > 0.004) {
                let sample = sampleHarmVolume(diskPos, n, n2, n3, rin, rout);
                let rho = max(sample.state0.x, 0.0);
                let heat = max(sample.state0.y, 0.0);
                let ur = sample.state0.z;
                let utheta = sample.state0.w;
                let uphi = sample.state1.x;
                let b1 = sample.state1.y;
                let b2 = sample.state1.z;
                let b3 = sample.state1.w;
                let mag = sqrt(max(b1 * b1 + b2 * b2 + b3 * b3, 0.0));
                var phi = atan2(diskPos.y, diskPos.x);
                if (phi < 0.0) {
                    phi += 6.283185307179586;
                }
                let er = normalize(diskPos);
                let et = normalize(vec3f(cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)));
                let ep = vec3f(-sin(phi), cos(phi), 0.0);
                let v = er * ur + et * (r * utheta) + ep * (r * max(sin(theta), 0.08) * uphi);
                let beta = clamp(length(v), 0.0, 0.86);
                let beam = clamp(dot(normalize(select(ep, v, beta > 0.02)), observerDisk), -1.0, 1.0);
                let doppler = pow(clamp(1.0 / max(1.0 - beta * beam, 0.22), 0.25, 4.2), 2.25);
                let redshift = sqrt(clamp(1.0 - rin / max(r, rin * 1.08), 0.08, 1.0));
                let inner = exp(-pow((r - criticalRadius * 1.00) / (criticalRadius * 0.86), 2.0));
                let plunge = smoothstep(rin * 1.08, criticalRadius * 0.85, r);
                let outerFade = smoothstep(rout * 0.48, criticalRadius * 1.18, r);
                let radialWindow = clamp((0.10 + 1.65 * inner) * plunge * outerFade, 0.0, 1.75);
                let fieldTexture = clamp(0.75 + 1.35 * abs(dot(normalize(vec3f(b1, b2, b3) + vec3f(1e-4)), ep)), 0.35, 2.0);
                let azTexture = 0.70 + 0.30 * fbm(vec2f(phi * 2.8 + log(max(r / rin, 1.0)) * 6.3, theta * 7.0 + u.render.x * 0.025));
                let synch = log(1.0 + u.tuning.x * (62.0 * rho + 70.0 * heat + 32.0 * mag));
                let emiss = synch * pow(midplane, 0.76) * doppler * redshift * radialWindow * fieldTexture * azTexture;
                let opacity = clamp((rho * 9.0 + heat * 2.2 + mag * 1.0) * pow(midplane, 1.35) * radialWindow * ds * 0.012, 0.0, 0.17);
                let scalar = clamp(emiss / 5.4, 0.0, 1.0);
                let localColor = firePalette(pow(scalar, 0.68)) * scalar;
                color += (1.0 - alpha) * localColor * opacity * 3.6;
                alpha += (1.0 - alpha) * opacity;
                scalarMax = max(scalarMax, scalar);
            }
        }
    }

    let photon = exp(-pow((b - criticalRadius) / photonWidth, 2.0));
    let ringPhi = atan2(p.y, p.x);
    let ringTexture = 0.42 + 0.58 * fbm(vec2f(ringPhi * 11.0 + u.render.x * 0.05, b * 2.9));
    let ringDoppler = 0.30 + 0.95 * smoothstep(-0.15, 0.85, dot(normalize(vec2f(cos(ringPhi), sin(ringPhi))), normalize(vec2f(-0.70, 0.32))));
    color += photon * ringTexture * ringDoppler * vec3f(1.0, 0.44, 0.035) * 0.55;

    let shadow = smoothstep(shadowRadius * 0.86, shadowRadius * 1.04, b) * escapedShadow;
    let centralGlow = vec3f(0.002, 0.0, 0.0);
    color = mix(centralGlow, color, shadow);

    let bg = vec3f(0.001, 0.0, 0.0);
    let vignette = smoothstep(outerImage * 1.15, criticalRadius * 0.75, b);
    let bloom = firePalette(scalarMax) * scalarMax * 0.12;
    let redFloor = vec3f(0.010, 0.0004, 0.0) * smoothstep(outerImage * 0.82, criticalRadius * 1.15, b);
    return vec4f(mix(bg, redFloor + color + bloom, clamp(vignette + photon * 0.18, 0.0, 1.0)), 1.0);
}
