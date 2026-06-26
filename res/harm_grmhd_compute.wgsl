struct HarmParams {
    gridN: u32,
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
};

struct HarmPrim {
    state0: vec4f, // rho, u, U1, U3
    state1: vec4f, // B1, B3, fail, reserved
};

struct HarmCons {
    d: f32,
    sr: f32,
    sp: f32,
    tau: f32,
    br: f32,
    bp: f32,
};

@group(0) @binding(0) var<uniform> params: HarmParams;
@group(0) @binding(1) var<storage, read_write> primA: array<HarmPrim>;
@group(0) @binding(2) var<storage, read_write> primB: array<HarmPrim>;

fn idx(ir: i32, ip: i32, n: i32) -> u32 {
    let cr = clamp(ir, 0, n - 1);
    var cp = ip % n;
    if (cp < 0) {
        cp += n;
    }
    return u32(cp * n + cr);
}

fn radius(ir: i32, n: i32) -> f32 {
    let x = (f32(ir) + 0.5) / f32(max(n, 1));
    return exp(log(params.rin) + x * (log(params.rout) - log(params.rin)));
}

fn sanitize(p: HarmPrim, r: f32) -> HarmPrim {
    var q = p;
    q.state0.x = max(q.state0.x, params.rhoFloor);
    q.state0.y = max(q.state0.y, params.uFloor);
    let vphi = r * q.state0.w;
    let v2 = max(q.state0.z * q.state0.z + vphi * vphi, 1e-12);
    if (v2 > 0.72) {
        let s = sqrt(0.72 / v2);
        q.state0.z *= s;
        q.state0.w *= s;
    }
    q.state1.x = clamp(q.state1.x, -0.45, 0.45);
    q.state1.y = clamp(q.state1.y, -0.45, 0.45);
    return q;
}

fn pressure(p: HarmPrim) -> f32 {
    return 0.33333334 * max(p.state0.y, params.uFloor);
}

fn prim_to_cons(p0: HarmPrim, r: f32) -> HarmCons {
    let p = sanitize(p0, r);
    let rho = max(p.state0.x, params.rhoFloor);
    let uu = max(p.state0.y, params.uFloor);
    let vr = p.state0.z;
    let vp = r * p.state0.w;
    let br = p.state1.x;
    let bp = p.state1.y;
    let b2 = br * br + bp * bp;
    let gamU = 1.3333334 * uu;
    let inertia = max(rho + gamU + b2, rho * 1.02);
    let v2 = min(vr * vr + vp * vp, 0.72);
    var c: HarmCons;
    c.d = rho;
    c.sr = inertia * vr;
    c.sp = inertia * vp;
    c.tau = uu + 0.5 * rho * v2 + 0.5 * b2;
    c.br = br;
    c.bp = bp;
    return c;
}

fn cons_add(a: HarmCons, b: HarmCons, s: f32) -> HarmCons {
    var c: HarmCons;
    c.d = a.d + s * b.d;
    c.sr = a.sr + s * b.sr;
    c.sp = a.sp + s * b.sp;
    c.tau = a.tau + s * b.tau;
    c.br = a.br + s * b.br;
    c.bp = a.bp + s * b.bp;
    return c;
}

fn cons_sub(a: HarmCons, b: HarmCons) -> HarmCons {
    return cons_add(a, b, -1.0);
}

fn cons_scale(a: HarmCons, s: f32) -> HarmCons {
    var c: HarmCons;
    c.d = a.d * s;
    c.sr = a.sr * s;
    c.sp = a.sp * s;
    c.tau = a.tau * s;
    c.br = a.br * s;
    c.bp = a.bp * s;
    return c;
}

fn flux(p0: HarmPrim, r: f32, dir: u32) -> HarmCons {
    let p = sanitize(p0, r);
    let rho = max(p.state0.x, params.rhoFloor);
    let uu = max(p.state0.y, params.uFloor);
    let vr = p.state0.z;
    let vp = r * p.state0.w;
    let br = p.state1.x;
    let bp = p.state1.y;
    let b2 = br * br + bp * bp;
    let pg = pressure(p);
    let ptot = pg + 0.5 * b2;
    let gamU = 1.3333334 * uu;
    let inertia = max(rho + gamU + b2, rho * 1.02);
    let vdotb = vr * br + vp * bp;
    var f: HarmCons;
    if (dir == 0u) {
        f.d = rho * vr;
        f.sr = inertia * vr * vr + ptot - br * br;
        f.sp = inertia * vp * vr - bp * br;
        f.tau = (uu + 0.5 * rho * (vr * vr + vp * vp) + pg + b2) * vr - br * vdotb;
        f.br = 0.0;
        f.bp = bp * vr - br * vp;
    } else {
        f.d = rho * vp;
        f.sr = inertia * vr * vp - br * bp;
        f.sp = inertia * vp * vp + ptot - bp * bp;
        f.tau = (uu + 0.5 * rho * (vr * vr + vp * vp) + pg + b2) * vp - bp * vdotb;
        f.br = br * vp - bp * vr;
        f.bp = 0.0;
    }
    return f;
}

fn wavespeed(p0: HarmPrim, r: f32, dir: u32) -> f32 {
    let p = sanitize(p0, r);
    let rho = max(p.state0.x, params.rhoFloor);
    let uu = max(p.state0.y, params.uFloor);
    let pg = pressure(p);
    let b2 = p.state1.x * p.state1.x + p.state1.y * p.state1.y;
    let cs2 = clamp(1.3333334 * pg / max(rho + 1.3333334 * uu + b2, 1e-8), 0.0, 0.55);
    let va2 = clamp(b2 / max(rho + 1.3333334 * uu + b2, 1e-8), 0.0, 0.85);
    let cf = sqrt(clamp(cs2 + va2 - cs2 * va2, 0.0, 0.92));
    let v = select(r * p.state0.w, p.state0.z, dir == 0u);
    return clamp(abs(v) + cf, 0.02, 0.98);
}

fn sane4(v: vec4f) -> bool {
    return all(v == v) && all(abs(v) < vec4f(1.0e12));
}

fn hll_flux(left: HarmPrim, right: HarmPrim, r: f32, dir: u32) -> HarmCons {
    let ul = prim_to_cons(left, r);
    let ur = prim_to_cons(right, r);
    let fl = flux(left, r, dir);
    let fr = flux(right, r, dir);
    let al = wavespeed(left, r, dir);
    let ar = wavespeed(right, r, dir);
    let ap = max(al, ar);
    var out: HarmCons;
    out.d = 0.5 * (fl.d + fr.d) - 0.5 * ap * (ur.d - ul.d);
    out.sr = 0.5 * (fl.sr + fr.sr) - 0.5 * ap * (ur.sr - ul.sr);
    out.sp = 0.5 * (fl.sp + fr.sp) - 0.5 * ap * (ur.sp - ul.sp);
    out.tau = 0.5 * (fl.tau + fr.tau) - 0.5 * ap * (ur.tau - ul.tau);
    out.br = 0.5 * (fl.br + fr.br) - 0.5 * ap * (ur.br - ul.br);
    out.bp = 0.5 * (fl.bp + fr.bp) - 0.5 * ap * (ur.bp - ul.bp);
    return out;
}

fn cons_to_prim(c0: HarmCons, old: HarmPrim, r: f32) -> HarmPrim {
    var c = c0;
    c.d = max(c.d, params.rhoFloor);
    c.tau = max(c.tau, params.uFloor);
    c.br = clamp(c.br, -0.45, 0.45);
    c.bp = clamp(c.bp, -0.45, 0.45);

    let b2 = c.br * c.br + c.bp * c.bp;
    var rho = max(c.d, params.rhoFloor);
    var uu = max(c.tau - 0.5 * b2, params.uFloor);
    var inertia = max(rho + 1.3333334 * uu + b2, rho * 1.02);
    var vr = c.sr / inertia;
    var vp = c.sp / inertia;
    let v2 = vr * vr + vp * vp;
    if (v2 > 0.72) {
        let s = sqrt(0.72 / v2);
        vr *= s;
        vp *= s;
    }
    uu = max(c.tau - 0.5 * rho * (vr * vr + vp * vp) - 0.5 * b2, params.uFloor);

    var p: HarmPrim;
    p.state0 = vec4f(rho, uu, vr, vp / max(r, 1.0));
    p.state1 = vec4f(c.br, c.bp, 0.0, 0.0);
    if (!sane4(p.state0) || !sane4(p.state1)) {
        p = old;
        p.state1.z = 1.0;
    }
    return sanitize(p, r);
}

@compute @workgroup_size(8, 8, 1)
fn harm_step(@builtin(global_invocation_id) gid: vec3u) {
    let n = i32(params.gridN);
    let ir = i32(gid.x);
    let ip = i32(gid.y);
    if (ir >= n || ip >= n) {
        return;
    }

    let i = idx(ir, ip, n);
    let r = radius(ir, n);
    let rin = max(params.rin, 1.01);
    let rout = max(params.rout, rin + 1.0);
    let dlogr = (log(rout) - log(rin)) / f32(n);
    let dr = max(r * dlogr, 1e-4);
    let dphi = 6.283185307179586 / f32(n);

    let c = sanitize(primA[i], r);
    let rm = sanitize(primA[idx(ir - 1, ip, n)], radius(ir - 1, n));
    let rp = sanitize(primA[idx(ir + 1, ip, n)], radius(ir + 1, n));
    let pm = sanitize(primA[idx(ir, ip - 1, n)], r);
    let pp = sanitize(primA[idx(ir, ip + 1, n)], r);

    let dt = params.dt;
    var ucons = prim_to_cons(c, r);
    let frp = hll_flux(c, rp, r, 0u);
    let frm = hll_flux(rm, c, r, 0u);
    let fpp = hll_flux(c, pp, r, 1u);
    let fpm = hll_flux(pm, c, r, 1u);

    ucons = cons_add(ucons, cons_sub(frm, frp), dt / dr);
    ucons = cons_add(ucons, cons_sub(fpm, fpp), dt / max(r * dphi, 1e-4));

    let pgas = pressure(c);
    let rho = max(c.state0.x, params.rhoFloor);
    let uu = max(c.state0.y, params.uFloor);
    let vr = c.state0.z;
    let vp = r * c.state0.w;
    let b2 = c.state1.x * c.state1.x + c.state1.y * c.state1.y;
    let inertia = max(rho + 1.3333334 * uu + b2, rho * 1.02);
    let grav = -rho / max(r * r, 1e-4);
    let centrifugal = inertia * vp * vp / max(r, 1e-4);
    let pressureCurve = pgas / max(r, 1e-4);
    let magneticTension = -0.35 * c.state1.y * c.state1.y / max(r, 1e-4);
    ucons.sr += dt * (grav + centrifugal - pressureCurve + magneticTension);
    ucons.sp += dt * (-vr * ucons.sp / max(r, 1.0));
    ucons.tau += dt * (0.012 * b2 * abs(vr) / max(r, 1.0));

    var out = cons_to_prim(ucons, c, r);

    let edgeOuter = smoothstep(0.0, 6.0, f32(n - 1 - ir));
    let edgeInner = smoothstep(0.0, 4.0, f32(ir));
    out.state0.x = mix(params.rhoFloor, out.state0.x, edgeOuter);
    out.state0.y = mix(params.uFloor, out.state0.y, edgeOuter);
    if (ir < 3) {
        out.state0.z = min(out.state0.z, -0.10);
        out.state0.x *= 0.78;
        out.state0.y *= 0.78;
        out.state1.x *= 0.7;
        out.state1.y *= 0.7;
    }
    if (ir > n - 4) {
        let omegaK = 1.0 / pow(max(r, 1.0), 1.5) / (1.0 + params.spin / pow(max(r, 1.0), 1.5));
        out.state0.z = mix(-0.01 / sqrt(max(r, 1.0)), out.state0.z, edgeOuter);
        out.state0.w = mix(0.72 * omegaK, out.state0.w, edgeOuter);
    }
    let omega = out.state0.w;
    let shearWind = -1.5 * omega * out.state1.x;
    let magneticStress = out.state1.x * out.state1.y;
    out.state1.y += dt * 0.85 * shearWind;
    out.state0.z += dt * clamp(0.28 * magneticStress / max(out.state0.x, params.rhoFloor), -0.08, 0.05);
    out.state0.y += dt * 0.030 * abs(magneticStress) * max(r * abs(omega), 0.2);
    out = sanitize(out, r);
    out.state0.x = max(out.state0.x, params.rhoFloor * edgeInner);
    out.state0.y = max(out.state0.y, params.uFloor * edgeInner);
    primB[i] = out;
}

@compute @workgroup_size(8, 8, 1)
fn copy_b_to_a(@builtin(global_invocation_id) gid: vec3u) {
    let n = i32(params.gridN);
    let ir = i32(gid.x);
    let ip = i32(gid.y);
    if (ir >= n || ip >= n) {
        return;
    }
    let i = idx(ir, ip, n);
    primA[i] = primB[i];
}
