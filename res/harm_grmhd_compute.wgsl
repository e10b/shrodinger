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
    highOrder: f32,
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

fn minmod(a: f32, b: f32) -> f32 {
    if (a * b > 0.0) {
        if (a > 0.0) {
            return min(a, b);
        } else {
            return max(a, b);
        }
    }
    return 0.0;
}

fn minmod4(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(
        minmod(a.x, b.x),
        minmod(a.y, b.y),
        minmod(a.z, b.z),
        minmod(a.w, b.w)
    );
}

fn minmodPrim(a: HarmPrim, b: HarmPrim) -> HarmPrim {
    var out: HarmPrim;
    out.state0 = minmod4(a.state0, b.state0);
    out.state1 = minmod4(a.state1, b.state1);
    out.state2 = minmod4(a.state2, b.state2);
    return out;
}

fn addPrim(a: HarmPrim, b: HarmPrim, scale: f32) -> HarmPrim {
    var out: HarmPrim;
    out.state0 = a.state0 + scale * b.state0;
    out.state1 = a.state1 + scale * b.state1;
    out.state2 = a.state2 + scale * b.state2;
    return out;
}

fn subPrim(a: HarmPrim, b: HarmPrim) -> HarmPrim {
    var out: HarmPrim;
    out.state0 = a.state0 - b.state0;
    out.state1 = a.state1 - b.state1;
    out.state2 = a.state2 - b.state2;
    return out;
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

fn sqrt_minus_g(r: f32, th: f32) -> f32 {
    let a = clamp(params.spin, -0.98, 0.98);
    let cth = cos(th);
    return max((r * r + a * a * cth * cth) * max(sin(th), 0.08), 1e-6);
}

fn sanitize(p: HarmPrim, r: f32, th: f32) -> HarmPrim {
    var q = p;
    q.state0.x = max(q.state0.x, params.rhoFloor);
    q.state0.y = max(q.state0.y, params.uFloor);
    let sth = max(sin(th), 0.08);
    var v = vec3f(q.state0.z, r * q.state0.w, r * sth * q.state1.x);
    let v2 = dot(v, v);
    if (v2 > 0.92) {
        let s = sqrt(0.92 / v2);
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
    let v = vec3f(v1, v2, v3);
    let v2mag = clamp(dot(v, v), 0.0, 0.98);
    let gamma = inverseSqrt(max(1.0 - v2mag, 1e-6));
    let sqrtg = sqrt_minus_g(r, th);
    let inertia = max(rho + 1.3333334 * uu + bsq, rho * 1.02);
    let vdb = v1 * b1 + v2 * b2 + v3 * b3;
    var c: HarmCons;
    c.d = sqrtg * rho * gamma;
    c.s1 = sqrtg * (inertia * gamma * gamma * v1 - vdb * b1);
    c.s2 = sqrtg * (inertia * gamma * gamma * v2 - vdb * b2);
    c.s3 = sqrtg * (inertia * gamma * gamma * v3 - vdb * b3);
    c.tau = sqrtg * (inertia * gamma * gamma - pressure(p) - 0.5 * bsq - rho * gamma);
    c.b1 = sqrtg * b1;
    c.b2 = sqrtg * b2;
    c.b3 = sqrtg * b3;
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
    let gamma = inverseSqrt(max(1.0 - clamp(dot(v, v), 0.0, 0.98), 1e-6));
    let sqrtg = sqrt_minus_g(r, th);
    let vdb = dot(v, b);
    let vd = select(select(v.z, v.y, dir == 1u), v.x, dir == 0u);
    let bd = select(select(b.z, b.y, dir == 1u), b.x, dir == 0u);
    var f: HarmCons;
    f.d = sqrtg * rho * gamma * vd;
    f.s1 = sqrtg * (inertia * gamma * gamma * v.x * vd - b.x * bd);
    f.s2 = sqrtg * (inertia * gamma * gamma * v.y * vd - b.y * bd);
    f.s3 = sqrtg * (inertia * gamma * gamma * v.z * vd - b.z * bd);
    if (dir == 0u) { f.s1 += sqrtg * ptot; }
    if (dir == 1u) { f.s2 += sqrtg * ptot; }
    if (dir == 2u) { f.s3 += sqrtg * ptot; }
    f.tau = sqrtg * ((inertia * gamma * gamma - rho * gamma) * vd - bd * vdb);
    f.b1 = sqrtg * (b.x * vd - bd * v.x);
    f.b2 = sqrtg * (b.y * vd - bd * v.y);
    f.b3 = sqrtg * (b.z * vd - bd * v.z);
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

fn recovery_residual(w: f32, d: f32, tau: f32, s2: f32, bsq: f32) -> f32 {
    let q = max(w + bsq, 1e-8);
    let v2 = clamp(s2 / max(q * q, 1e-12), 0.0, 0.92);
    let gamma = inverseSqrt(max(1.0 - v2, 1e-6));
    let rho = max(d / gamma, params.rhoFloor);
    let uu = max((w / max(gamma * gamma, 1e-6) - rho) / 1.3333334, params.uFloor);
    let pg = 0.33333334 * uu;
    return w - pg + 0.5 * bsq + 0.5 * v2 * w - (tau + d);
}

fn cons_to_prim(c0: HarmCons, old: HarmPrim, r: f32, th: f32) -> HarmPrim {
    var c = c0;
    let sqrtg = sqrt_minus_g(r, th);
    c.d = max(c.d / sqrtg, params.rhoFloor);
    c.s1 = c.s1 / sqrtg;
    c.s2 = c.s2 / sqrtg;
    c.s3 = c.s3 / sqrtg;
    c.tau = max(c.tau / sqrtg, params.uFloor);
    let b = clamp(vec3f(c.b1, c.b2, c.b3) / sqrtg, vec3f(-0.60), vec3f(0.60));
    let bsq = dot(b, b);
    let mom = vec3f(c.s1, c.s2, c.s3);
    let s2 = dot(mom, mom);
    var w = max(c.d + c.tau + bsq + pressure(old), c.d + params.uFloor + bsq);
    var failed = 0.0;
    for (var iter = 0; iter < 20; iter = iter + 1) {
        let f = recovery_residual(w, c.d, c.tau, s2, bsq);
        let eps = max(1e-3 * w, 1e-6);
        let wm = max(w - eps, c.d + params.uFloor + bsq);
        let fp = recovery_residual(w + eps, c.d, c.tau, s2, bsq);
        let fm = recovery_residual(wm, c.d, c.tau, s2, bsq);
        let dfdw = (fp - fm) / max((w + eps) - wm, 1e-8);
        let step = f / select(dfdw, select(-1e-8, 1e-8, dfdw >= 0.0), abs(dfdw) < 1e-8);
        w = max(w - clamp(step, -0.35 * w, 0.35 * w), c.d + params.uFloor + bsq);
        if (abs(f) < 1e-6 * max(c.d + c.tau + bsq, 1.0)) {
            break;
        }
    }
    var v = mom / max(w + bsq, 1e-8);
    let v2 = dot(v, v);
    if (v2 > 0.92) {
        v *= sqrt(0.92 / v2);
    }
    let gamma = inverseSqrt(max(1.0 - clamp(dot(v, v), 0.0, 0.92), 1e-6));
    let rho = max(c.d / gamma, params.rhoFloor);
    var uu = max((w / max(gamma * gamma, 1e-6) - rho) / 1.3333334, params.uFloor);
    let sth = max(sin(th), 0.08);
    var p: HarmPrim;
    p.state0 = vec4f(rho, uu, v.x, v.y / max(r, 1.0));
    p.state1 = vec4f(v.z / max(r * sth, 1.0), b.x, b.y, b.z);
    p.state2 = vec4f(failed, 0.0, 0.0, 0.0);
    if (!sane4(p.state0) || !sane4(p.state1)) {
        p = old;
        p.state2.x = 1.0;
    }
    return sanitize(p, r, th);
}

fn kerr_lapse(r: f32, th: f32) -> f32 {
    let a = clamp(params.spin, -0.98, 0.98);
    let sth = max(sin(th), 0.08);
    let cth = cos(th);
    let sigma = r * r + a * a * cth * cth;
    let delta = max(r * r - 2.0 * r + a * a, 1e-4);
    let aa = (r * r + a * a) * (r * r + a * a) - a * a * delta * sth * sth;
    return sqrt(max(sigma * delta / max(aa, 1e-6), 1e-6));
}

fn kerr_frame_drag(r: f32, th: f32) -> f32 {
    let a = clamp(params.spin, -0.98, 0.98);
    let sth = max(sin(th), 0.08);
    let cth = cos(th);
    let sigma = r * r + a * a * cth * cth;
    let delta = max(r * r - 2.0 * r + a * a, 1e-4);
    let aa = (r * r + a * a) * (r * r + a * a) - a * a * delta * sth * sth;
    return 2.0 * a * r / max(aa, 1e-6);
}

fn gcov_terms(r: f32, th: f32) -> vec4f {
    let a = clamp(params.spin, -0.98, 0.98);
    let sth = max(sin(th), 0.08);
    let cth = cos(th);
    let sigma = r * r + a * a * cth * cth;
    let two = 2.0 * r / max(sigma, 1e-6);
    let gtt = -(1.0 - two);
    let gtr = two;
    let gtp = -two * a * sth * sth;
    let gpp = (r * r + a * a + two * a * a * sth * sth) * sth * sth;
    return vec4f(gtt, gtr, gtp, gpp);
}

fn gr_metric_source(p0: HarmPrim, r: f32, th: f32) -> vec3f {
    let p = sanitize(p0, r, th);
    let rho = max(p.state0.x, params.rhoFloor);
    let uu = max(p.state0.y, params.uFloor);
    let pg = pressure(p);
    let sth = max(sin(th), 0.08);
    let v = vec3f(p.state0.z, r * p.state0.w, r * sth * p.state1.x);
    let b = vec3f(p.state1.y, p.state1.z, p.state1.w);
    let bsq = dot(b, b);
    let gamma = inverseSqrt(max(1.0 - clamp(dot(v, v), 0.0, 0.92), 1e-6));
    let w = rho + uu + pg + bsq;
    let sqrtg = sqrt_minus_g(r, th);
    let drs = max(1e-3 * max(r, 1.0), 1e-4);
    let dts = 1e-4;
    let rp = gcov_terms(r + drs, th);
    let rm = gcov_terms(max(r - drs, params.rin * 1.0001), th);
    let tp = gcov_terms(r, min(th + dts, 3.1414926));
    let tm = gcov_terms(r, max(th - dts, 0.0001));
    let dgr = (rp - rm) / max((r + drs) - max(r - drs, params.rin * 1.0001), 1e-6);
    let dgt = (tp - tm) / max(min(th + dts, 3.1414926) - max(th - dts, 0.0001), 1e-6);
    let ut = gamma;
    let ur = gamma * v.x;
    let up = gamma * v.z / max(r * sth, 1.0);
    let ptot = pg + 0.5 * bsq;
    let Ttt = w * ut * ut - ptot;
    let Ttr = w * ut * ur;
    let Ttp = w * ut * up;
    let Tpp = w * up * up + ptot / max(r * r * sth * sth, 1.0);
    let sr = 0.5 * sqrtg * (Ttt * dgr.x + 2.0 * Ttr * dgr.y + 2.0 * Ttp * dgr.z + Tpp * dgr.w);
    let st = 0.5 * sqrtg * (Ttt * dgt.x + 2.0 * Ttr * dgt.y + 2.0 * Ttp * dgt.z + Tpp * dgt.w);
    return vec3f(sr, st, -0.02 * sqrtg * rho * v.z / max(r, 1.0));
}

fn kerr_metric_source(c: HarmPrim, r: f32, th: f32) -> vec3f {
    let rho = max(c.state0.x, params.rhoFloor);
    let uu = max(c.state0.y, params.uFloor);
    let pg = pressure(c);
    let sth = max(sin(th), 0.08);
    let v = vec3f(c.state0.z, r * c.state0.w, r * sth * c.state1.x);
    let b = vec3f(c.state1.y, c.state1.z, c.state1.w);
    let bsq = dot(b, b);
    let inertia = max(rho + 1.3333334 * uu + bsq, rho * 1.02);
    let lapse = kerr_lapse(r, th);
    let lp = kerr_lapse(r * 1.001 + 1e-4, th);
    let lm = kerr_lapse(max(r * 0.999 - 1e-4, params.rin * 1.01), th);
    let dlnAlpha = (log(max(lp, 1e-6)) - log(max(lm, 1e-6))) / max((r * 0.002 + 2e-4), 1e-4);
    let omega = kerr_frame_drag(r, th);
    let op = kerr_frame_drag(r * 1.001 + 1e-4, th);
    let om = kerr_frame_drag(max(r * 0.999 - 1e-4, params.rin * 1.01), th);
    let domega = (op - om) / max((r * 0.002 + 2e-4), 1e-4);
    let gravR = -inertia * dlnAlpha;
    let centrifugal = inertia * (v.y * v.y + v.z * v.z) / max(r, 1e-4);
    let polar = -inertia * v.z * v.z * cos(th) / max(r * sth, 0.1);
    let frameDrag = -inertia * v.z * r * sth * domega;
    let magneticHoop = -0.35 * (b.z * b.z + 0.25 * bsq) / max(r, 1.0);
    return vec3f(
        gravR + centrifugal - pg / max(r, 1e-4) + frameDrag + magneticHoop,
        polar - 0.20 * pg * cos(th) / max(r * sth, 0.1),
        -c.state0.z * inertia * v.z / max(r, 1.0) + 0.32 * (b.x * b.z + b.y * b.z));
}

fn velocity_phys(p: HarmPrim, r: f32, th: f32) -> vec3f {
    let sth = max(sin(th), 0.08);
    return vec3f(p.state0.z, r * p.state0.w, r * sth * p.state1.x);
}

fn magnetic_phys(p: HarmPrim) -> vec3f {
    return vec3f(p.state1.y, p.state1.z, p.state1.w);
}

fn electric_ideal(p: HarmPrim, r: f32, th: f32) -> vec3f {
    return -cross(velocity_phys(p, r, th), magnetic_phys(p));
}

fn ct_induction_update(
    center: HarmPrim,
    rm: HarmPrim,
    rp: HarmPrim,
    tm: HarmPrim,
    tp: HarmPrim,
    pm: HarmPrim,
    pp: HarmPrim,
    r: f32,
    th: f32,
    dr: f32,
    dth: f32,
    dph: f32) -> vec3f {
    let sth = max(sin(th), 0.08);
    let thm = theta(max(0, i32(floor((th / 3.141592653589793 - 0.08) / 0.84 * f32(params.n2))) - 1));
    let thp = theta(min(i32(params.n2) - 1, i32(floor((th / 3.141592653589793 - 0.08) / 0.84 * f32(params.n2))) + 1));
    let eRm = electric_ideal(rm, max(r - dr, params.rin), th);
    let eRp = electric_ideal(rp, r + dr, th);
    let eTm = electric_ideal(tm, r, thm);
    let eTp = electric_ideal(tp, r, thp);
    let ePm = electric_ideal(pm, r, th);
    let ePp = electric_ideal(pp, r, th);

    let sinTm = max(sin(thm), 0.08);
    let sinTp = max(sin(thp), 0.08);
    let d_sin_ephi_dth = (sinTp * eTp.z - sinTm * eTm.z) / max(2.0 * dth, 1e-4);
    let d_etheta_dphi = (ePp.y - ePm.y) / max(2.0 * dph, 1e-4);
    let d_er_dphi = (ePp.x - ePm.x) / max(2.0 * dph, 1e-4);
    let d_r_ephi_dr = ((r + dr) * eRp.z - max(r - dr, params.rin) * eRm.z) / max(2.0 * dr, 1e-4);
    let d_r_etheta_dr = ((r + dr) * eRp.y - max(r - dr, params.rin) * eRm.y) / max(2.0 * dr, 1e-4);
    let d_er_dth = (eTp.x - eTm.x) / max(2.0 * dth, 1e-4);

    let dbr = -(d_sin_ephi_dth - d_etheta_dphi) / max(r * sth, 1e-4);
    let dbt = -((d_er_dphi / sth) - d_r_ephi_dr) / max(r, 1e-4);
    let dbp = -(d_r_etheta_dr - d_er_dth) / max(r, 1e-4);
    return magnetic_phys(center) + params.dt * vec3f(dbr, dbt, dbp);
}

fn divb_spherical(center: HarmPrim, rm: HarmPrim, rp: HarmPrim, tm: HarmPrim, tp: HarmPrim, pm: HarmPrim, pp: HarmPrim, r: f32, th: f32, dr: f32, dth: f32, dph: f32) -> f32 {
    let sth = max(sin(th), 0.08);
    let brp = magnetic_phys(rp).x;
    let brm = magnetic_phys(rm).x;
    let btp = magnetic_phys(tp).y;
    let btm = magnetic_phys(tm).y;
    let bpp = magnetic_phys(pp).z;
    let bpm = magnetic_phys(pm).z;
    let thm = max(th - dth, 0.02);
    let thp = min(th + dth, 3.12159);
    let radial = (((r + dr) * (r + dr) * brp) - (max(r - dr, params.rin) * max(r - dr, params.rin) * brm)) / max(2.0 * dr, 1e-4);
    let polar = (sin(thp) * btp - sin(thm) * btm) / max(2.0 * dth, 1e-4);
    let az = (bpp - bpm) / max(2.0 * dph, 1e-4);
    return radial / max(r * r, 1e-4) + polar / max(r * sth, 1e-4) + az / max(r * sth, 1e-4);
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

    var flux_r_m: HarmCons;
    var flux_r_p: HarmCons;
    var flux_t_m: HarmCons;
    var flux_t_p: HarmCons;
    var flux_p_m: HarmCons;
    var flux_p_p: HarmCons;

    if (params.highOrder > 0.5) {
        let rmm = sanitize(primA[idx(ir - 2, it, ip)], radius(ir - 2), th);
        let rpp = sanitize(primA[idx(ir + 2, it, ip)], radius(ir + 2), th);
        let tmm = sanitize(primA[idx(ir, it - 2, ip)], r, theta(it - 2));
        let tpp = sanitize(primA[idx(ir, it + 2, ip)], r, theta(it + 2));
        let pmm = sanitize(primA[idx(ir, it, ip - 2)], r, th);
        let ppp = sanitize(primA[idx(ir, it, ip + 2)], r, th);

        let slope_rm = minmodPrim(subPrim(rm, rmm), subPrim(c, rm));
        let slope_c_r = minmodPrim(subPrim(c, rm), subPrim(rp, c));
        let slope_rp = minmodPrim(subPrim(rp, c), subPrim(rpp, rp));
        
        flux_r_m = hll_flux(addPrim(rm, slope_rm, 0.5), addPrim(c, slope_c_r, -0.5), r, th, 0u);
        flux_r_p = hll_flux(addPrim(c, slope_c_r, 0.5), addPrim(rp, slope_rp, -0.5), r, th, 0u);
        
        let slope_tm = minmodPrim(subPrim(tm, tmm), subPrim(c, tm));
        let slope_c_t = minmodPrim(subPrim(c, tm), subPrim(tp, c));
        let slope_tp = minmodPrim(subPrim(tp, c), subPrim(tpp, tp));
        
        flux_t_m = hll_flux(addPrim(tm, slope_tm, 0.5), addPrim(c, slope_c_t, -0.5), r, th, 1u);
        flux_t_p = hll_flux(addPrim(c, slope_c_t, 0.5), addPrim(tp, slope_tp, -0.5), r, th, 1u);
        
        let slope_pm = minmodPrim(subPrim(pm, pmm), subPrim(c, pm));
        let slope_c_p = minmodPrim(subPrim(c, pm), subPrim(pp, c));
        let slope_pp = minmodPrim(subPrim(pp, c), subPrim(ppp, pp));
        
        flux_p_m = hll_flux(addPrim(pm, slope_pm, 0.5), addPrim(c, slope_c_p, -0.5), r, th, 2u);
        flux_p_p = hll_flux(addPrim(c, slope_c_p, 0.5), addPrim(pp, slope_pp, -0.5), r, th, 2u);
    } else {
        flux_r_m = hll_flux(rm, c, r, th, 0u);
        flux_r_p = hll_flux(c, rp, r, th, 0u);
        flux_t_m = hll_flux(tm, c, r, th, 1u);
        flux_t_p = hll_flux(c, tp, r, th, 1u);
        flux_p_m = hll_flux(pm, c, r, th, 2u);
        flux_p_p = hll_flux(c, pp, r, th, 2u);
    }

    var u = prim_to_cons(c, r, th);
    u = cons_add(u, cons_sub(flux_r_m, flux_r_p), params.dt / dr);
    u = cons_add(u, cons_sub(flux_t_m, flux_t_p), params.dt / max(r * dth, 1e-4));
    u = cons_add(u, cons_sub(flux_p_m, flux_p_p), params.dt / max(r * sth * dph, 1e-4));

    let vph = r * sth * c.state1.x;
    let b = vec3f(c.state1.y, c.state1.z, c.state1.w);
    let magneticStress = b.x * b.z + 0.35 * b.y * b.z;
    let src = gr_metric_source(c, r, th);
    u.s1 += params.dt * src.x;
    u.s2 += params.dt * src.y;
    u.s3 += params.dt * src.z;
    u.tau += params.dt * (0.026 * abs(magneticStress) * max(abs(vph), 0.2));

    var out = cons_to_prim(u, c, r, th);
    if (ir < 3) {
        out.state0.z = min(out.state0.z, -0.05);
        out.state0.x *= 0.82;
        out.state0.y *= 0.82;
    }
    if (ir > n1 - 4) {
        out.state0.z = min(out.state0.z, 0.25);
        out.state0.x *= 0.92;
        out.state0.y *= 0.92;
    }
    if (min(it, n2 - 1 - it) < 2) {
        out.state0.x *= 0.80;
        out.state0.y *= 0.80;
        out.state0.w *= 0.5;
        out.state1.z *= 0.5;
    }
    let omega = out.state1.x;
    let shearWind = -1.5 * omega * out.state1.y;
    let ctB = ct_induction_update(c, rm, rp, tm, tp, pm, pp, r, th, dr, dth, dph);
    let divB = divb_spherical(c, rm, rp, tm, tp, pm, pp, r, th, dr, dth, dph);
    out.state1.y = ctB.x - params.dt * 0.10 * divB * dr;
    out.state1.z = ctB.y - params.dt * 0.10 * divB * r * dth;
    out.state1.w = ctB.z - params.dt * 0.05 * divB * r * sth * dph;
    out.state1.w += params.dt * 0.42 * shearWind;
    out.state0.z += params.dt * clamp(0.22 * magneticStress / max(out.state0.x, params.rhoFloor), -0.08, 0.05);
    out = sanitize(out, r, th);
    out.state0.x = max(out.state0.x, params.rhoFloor);
    out.state0.y = max(out.state0.y, params.uFloor);
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
