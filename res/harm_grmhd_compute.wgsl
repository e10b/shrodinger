struct HarmParams {
    n1: u32,
    n2: u32,
    n3: u32,
    substeps: u32,
    dt: f32,
    rin: f32,
    rout: f32,
    spin: f32,
    rhoFloor: f32,
    uFloor: f32,
    magneticLoop: f32,
    time: f32,
    problem: u32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
};

struct HarmPrim {
    state0: vec4f, // rho, u, U1, U2
    state1: vec4f, // U3, B1, B2, B3
    state2: vec4f, // fail, reserved
};

struct HarmCons {
    d: f32,
    s1: f32,
    s2: f32,
    s3: f32,
    tau: f32,
    b1: f32,
    b2: f32,
    b3: f32,
};

@group(0) @binding(0) var<uniform> params: HarmParams;
@group(0) @binding(1) var<storage, read_write> primA: array<HarmPrim>;
@group(0) @binding(2) var<storage, read_write> primB: array<HarmPrim>;

fn wrap(i: i32, n: i32) -> i32 {
    var q = i % n;
    if (q < 0) {
        q += n;
    }
    return q;
}

fn idx(ir: i32, it: i32, ip: i32) -> u32 {
    let n1 = i32(params.n1);
    let n2 = i32(params.n2);
    let n3 = i32(params.n3);
    let r = clamp(ir, 0, n1 - 1);
    let t = clamp(it, 0, n2 - 1);
    let p = wrap(ip, n3);
    return u32((p * n2 + t) * n1 + r);
}

fn radius(ir: i32) -> f32 {
    let n1 = max(i32(params.n1), 1);
    let x = (f32(clamp(ir, 0, n1 - 1)) + 0.5) / f32(n1);
    return exp(log(params.rin) + x * (log(params.rout) - log(params.rin)));
}

fn theta(it: i32) -> f32 {
    let n2 = max(i32(params.n2), 1);
    let x = (f32(clamp(it, 0, n2 - 1)) + 0.5) / f32(n2);
    return 0.08 * 3.141592653589793 + x * 0.84 * 3.141592653589793;
}

fn pressure(p: HarmPrim) -> f32 {
    return 0.33333334 * max(p.state0.y, params.uFloor);
}

fn sanitize(p: HarmPrim, r: f32, th: f32) -> HarmPrim {
    var q = p;
    q.state0.x = max(q.state0.x, params.rhoFloor);
    q.state0.y = max(q.state0.y, params.uFloor);
    let sth = max(sin(th), 0.08);
    var v = vec3f(q.state0.z, r * q.state0.w, r * sth * q.state1.x);
    let v2 = dot(v, v);
    if (v2 > 0.72) {
        let s = sqrt(0.72 / v2);
        q.state0.z *= s;
        q.state0.w *= s;
        q.state1.x *= s;
    }
    q.state1.y = clamp(q.state1.y, -0.60, 0.60);
    q.state1.z = clamp(q.state1.z, -0.60, 0.60);
    q.state1.w = clamp(q.state1.w, -0.60, 0.60);
    return q;
}

fn prim_to_cons(p0: HarmPrim, r: f32, th: f32) -> HarmCons {
    let p = sanitize(p0, r, th);
    let rho = max(p.state0.x, params.rhoFloor);
    let uu = max(p.state0.y, params.uFloor);
    let sth = max(sin(th), 0.08);
    let v1 = p.state0.z;
    let v2 = r * p.state0.w;
    let v3 = r * sth * p.state1.x;
    let b1 = p.state1.y;
    let b2 = p.state1.z;
    let b3 = p.state1.w;
    let bsq = b1 * b1 + b2 * b2 + b3 * b3;
    let inertia = max(rho + 1.3333334 * uu + bsq, rho * 1.02);
    var c: HarmCons;
    c.d = rho;
    c.s1 = inertia * v1;
    c.s2 = inertia * v2;
    c.s3 = inertia * v3;
    c.tau = uu + 0.5 * rho * dot(vec3f(v1, v2, v3), vec3f(v1, v2, v3)) + 0.5 * bsq;
    c.b1 = b1;
    c.b2 = b2;
    c.b3 = b3;
    return c;
}

fn cons_add(a: HarmCons, b: HarmCons, s: f32) -> HarmCons {
    var c: HarmCons;
    c.d = a.d + s * b.d;
    c.s1 = a.s1 + s * b.s1;
    c.s2 = a.s2 + s * b.s2;
    c.s3 = a.s3 + s * b.s3;
    c.tau = a.tau + s * b.tau;
    c.b1 = a.b1 + s * b.b1;
    c.b2 = a.b2 + s * b.b2;
    c.b3 = a.b3 + s * b.b3;
    return c;
}

fn cons_sub(a: HarmCons, b: HarmCons) -> HarmCons {
    return cons_add(a, b, -1.0);
}

fn flux(p0: HarmPrim, r: f32, th: f32, dir: u32) -> HarmCons {
    let p = sanitize(p0, r, th);
    let rho = max(p.state0.x, params.rhoFloor);
    let uu = max(p.state0.y, params.uFloor);
    let sth = max(sin(th), 0.08);
    let v = vec3f(p.state0.z, r * p.state0.w, r * sth * p.state1.x);
    let b = vec3f(p.state1.y, p.state1.z, p.state1.w);
    let bsq = dot(b, b);
    let ptot = pressure(p) + 0.5 * bsq;
    let inertia = max(rho + 1.3333334 * uu + bsq, rho * 1.02);
    let vdb = dot(v, b);
    let vd = select(select(v.z, v.y, dir == 1u), v.x, dir == 0u);
    let bd = select(select(b.z, b.y, dir == 1u), b.x, dir == 0u);
    var f: HarmCons;
    f.d = rho * vd;
    f.s1 = inertia * v.x * vd - b.x * bd;
    f.s2 = inertia * v.y * vd - b.y * bd;
    f.s3 = inertia * v.z * vd - b.z * bd;
    if (dir == 0u) { f.s1 += ptot; }
    if (dir == 1u) { f.s2 += ptot; }
    if (dir == 2u) { f.s3 += ptot; }
    f.tau = (uu + 0.5 * rho * dot(v, v) + pressure(p) + bsq) * vd - bd * vdb;
    f.b1 = b.x * vd - bd * v.x;
    f.b2 = b.y * vd - bd * v.y;
    f.b3 = b.z * vd - bd * v.z;
    if (dir == 0u) { f.b1 = 0.0; }
    if (dir == 1u) { f.b2 = 0.0; }
    if (dir == 2u) { f.b3 = 0.0; }
    return f;
}

fn wavespeed(p0: HarmPrim, r: f32, th: f32, dir: u32) -> f32 {
    let p = sanitize(p0, r, th);
    let rho = max(p.state0.x, params.rhoFloor);
    let uu = max(p.state0.y, params.uFloor);
    let b = vec3f(p.state1.y, p.state1.z, p.state1.w);
    let bsq = dot(b, b);
    let cs2 = clamp(1.3333334 * pressure(p) / max(rho + 1.3333334 * uu + bsq, 1e-8), 0.0, 0.55);
    let va2 = clamp(bsq / max(rho + 1.3333334 * uu + bsq, 1e-8), 0.0, 0.85);
    let cf = sqrt(clamp(cs2 + va2 - cs2 * va2, 0.0, 0.92));
    let sth = max(sin(th), 0.08);
    let v = vec3f(p.state0.z, r * p.state0.w, r * sth * p.state1.x);
    let vd = select(select(v.z, v.y, dir == 1u), v.x, dir == 0u);
    return clamp(abs(vd) + cf, 0.02, 0.98);
}

fn hll_flux(left: HarmPrim, right: HarmPrim, r: f32, th: f32, dir: u32) -> HarmCons {
    let ul = prim_to_cons(left, r, th);
    let ur = prim_to_cons(right, r, th);
    let fl = flux(left, r, th, dir);
    let fr = flux(right, r, th, dir);
    let ap = max(wavespeed(left, r, th, dir), wavespeed(right, r, th, dir));
    var out: HarmCons;
    out.d = 0.5 * (fl.d + fr.d) - 0.5 * ap * (ur.d - ul.d);
    out.s1 = 0.5 * (fl.s1 + fr.s1) - 0.5 * ap * (ur.s1 - ul.s1);
    out.s2 = 0.5 * (fl.s2 + fr.s2) - 0.5 * ap * (ur.s2 - ul.s2);
    out.s3 = 0.5 * (fl.s3 + fr.s3) - 0.5 * ap * (ur.s3 - ul.s3);
    out.tau = 0.5 * (fl.tau + fr.tau) - 0.5 * ap * (ur.tau - ul.tau);
    out.b1 = 0.5 * (fl.b1 + fr.b1) - 0.5 * ap * (ur.b1 - ul.b1);
    out.b2 = 0.5 * (fl.b2 + fr.b2) - 0.5 * ap * (ur.b2 - ul.b2);
    out.b3 = 0.5 * (fl.b3 + fr.b3) - 0.5 * ap * (ur.b3 - ul.b3);
    return out;
}

fn sane4(v: vec4f) -> bool {
    return all(v == v) && all(abs(v) < vec4f(1.0e12));
}

fn cons_to_prim(c0: HarmCons, old: HarmPrim, r: f32, th: f32) -> HarmPrim {
    var c = c0;
    c.d = max(c.d, params.rhoFloor);
    c.tau = max(c.tau, params.uFloor);
    let b = clamp(vec3f(c.b1, c.b2, c.b3), vec3f(-0.60), vec3f(0.60));
    let bsq = dot(b, b);
    let rho = max(c.d, params.rhoFloor);
    var uu = max(c.tau - 0.5 * bsq, params.uFloor);
    let inertia = max(rho + 1.3333334 * uu + bsq, rho * 1.02);
    var v = vec3f(c.s1, c.s2, c.s3) / inertia;
    let v2 = dot(v, v);
    if (v2 > 0.72) {
        v *= sqrt(0.72 / v2);
    }
    uu = max(c.tau - 0.5 * rho * dot(v, v) - 0.5 * bsq, params.uFloor);
    let sth = max(sin(th), 0.08);
    var p: HarmPrim;
    p.state0 = vec4f(rho, uu, v.x, v.y / max(r, 1.0));
    p.state1 = vec4f(v.z / max(r * sth, 1.0), b.x, b.y, b.z);
    p.state2 = vec4f(0.0);
    if (!sane4(p.state0) || !sane4(p.state1)) {
        p = old;
        p.state2.x = 1.0;
    }
    return sanitize(p, r, th);
}

@compute @workgroup_size(8, 8, 4)
fn harm_step(@builtin(global_invocation_id) gid: vec3u) {
    let ir = i32(gid.x);
    let it = i32(gid.y);
    let ip = i32(gid.z);
    let n1 = i32(params.n1);
    let n2 = i32(params.n2);
    let n3 = i32(params.n3);
    if (ir >= n1 || it >= n2 || ip >= n3) {
        return;
    }

    let r = radius(ir);
    let th = theta(it);
    let sth = max(sin(th), 0.08);
    let rin = max(params.rin, 1.01);
    let rout = max(params.rout, rin + 1.0);
    let dr = max(r * (log(rout) - log(rin)) / f32(n1), 1e-4);
    let dth = 0.84 * 3.141592653589793 / f32(n2);
    let dph = 6.283185307179586 / f32(n3);

    let c = sanitize(primA[idx(ir, it, ip)], r, th);
    let rm = sanitize(primA[idx(ir - 1, it, ip)], radius(ir - 1), th);
    let rp = sanitize(primA[idx(ir + 1, it, ip)], radius(ir + 1), th);
    let tm = sanitize(primA[idx(ir, it - 1, ip)], r, theta(it - 1));
    let tp = sanitize(primA[idx(ir, it + 1, ip)], r, theta(it + 1));
    let pm = sanitize(primA[idx(ir, it, ip - 1)], r, th);
    let pp = sanitize(primA[idx(ir, it, ip + 1)], r, th);

    var u = prim_to_cons(c, r, th);
    u = cons_add(u, cons_sub(hll_flux(rm, c, r, th, 0u), hll_flux(c, rp, r, th, 0u)), params.dt / dr);
    u = cons_add(u, cons_sub(hll_flux(tm, c, r, th, 1u), hll_flux(c, tp, r, th, 1u)), params.dt / max(r * dth, 1e-4));
    u = cons_add(u, cons_sub(hll_flux(pm, c, r, th, 2u), hll_flux(c, pp, r, th, 2u)), params.dt / max(r * sth * dph, 1e-4));

    let rho = max(c.state0.x, params.rhoFloor);
    let uu = max(c.state0.y, params.uFloor);
    let vth = r * c.state0.w;
    let vph = r * sth * c.state1.x;
    let b = vec3f(c.state1.y, c.state1.z, c.state1.w);
    let bsq = dot(b, b);
    let inertia = max(rho + 1.3333334 * uu + bsq, rho * 1.02);
    let grav = -rho / max(r * r, 1e-4);
    let centrifugal = inertia * (vth * vth + vph * vph) / max(r, 1e-4);
    let magneticStress = b.x * b.z + 0.35 * b.y * b.z;
    u.s1 += params.dt * (grav + centrifugal - pressure(c) / max(r, 1e-4) - 0.22 * bsq / max(r, 1.0));
    u.s2 += params.dt * (-0.35 * rho * cos(th) / max(r * sth, 0.1));
    u.s3 += params.dt * (-c.state0.z * u.s3 / max(r, 1.0) + 0.22 * magneticStress);
    u.tau += params.dt * (0.026 * abs(magneticStress) * max(abs(vph), 0.2));

    var out = cons_to_prim(u, c, r, th);
    let edgeOuter = smoothstep(0.0, 6.0, f32(n1 - 1 - ir));
    let edgeInner = smoothstep(0.0, 4.0, f32(ir));
    out.state0.x = mix(params.rhoFloor, out.state0.x, edgeOuter);
    out.state0.y = mix(params.uFloor, out.state0.y, edgeOuter);
    if (ir < 3) {
        out.state0.z = min(out.state0.z, -0.10);
        out.state0.x *= 0.74;
        out.state0.y *= 0.74;
        out.state1.y *= 0.68;
        out.state1.z *= 0.68;
        out.state1.w *= 0.68;
    }
    let polarDamp = smoothstep(0.0, 4.0, f32(min(it, n2 - 1 - it)));
    out.state0.x = mix(params.rhoFloor, out.state0.x, polarDamp);
    out.state0.y = mix(params.uFloor, out.state0.y, polarDamp);
    let omega = out.state1.x;
    let shearWind = -1.5 * omega * out.state1.y;
    out.state1.w += params.dt * 0.70 * shearWind;
    out.state0.z += params.dt * clamp(0.22 * magneticStress / max(out.state0.x, params.rhoFloor), -0.08, 0.05);
    out = sanitize(out, r, th);
    out.state0.x = max(out.state0.x, params.rhoFloor * edgeInner);
    out.state0.y = max(out.state0.y, params.uFloor * edgeInner);
    primB[idx(ir, it, ip)] = out;
}

@compute @workgroup_size(8, 8, 4)
fn copy_b_to_a(@builtin(global_invocation_id) gid: vec3u) {
    let ir = i32(gid.x);
    let it = i32(gid.y);
    let ip = i32(gid.z);
    if (ir >= i32(params.n1) || it >= i32(params.n2) || ip >= i32(params.n3)) {
        return;
    }
    let i = idx(ir, it, ip);
    primA[i] = primB[i];
}
