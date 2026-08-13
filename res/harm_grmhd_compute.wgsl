struct HarmParams {
    n1: u32, n2: u32, n3: u32, substeps: u32,
    dt: f32, rin: f32, rout: f32, spin: f32,
    rhoFloor: f32, uFloor: f32, magneticLoop: f32, time: f32,
    problem: u32, highOrder: f32, phiSplit: u32, tiledStorage: u32,
};

struct HarmPrim { state0: vec4f, state1: vec4f } // rho,u,v^r,v^theta | v^phi,B^r,B^theta,B^phi
struct HarmCons { d: f32, s1: f32, s2: f32, s3: f32, tau: f32, b1: f32, b2: f32, b3: f32 }
struct RecoverResult { prim: HarmPrim, failed: u32 }

@group(0) @binding(0) var<uniform> params: HarmParams;
// cflDt is a positive f32 encoded as u32; cflTime uses 1e-5 M integer ticks,
// covering more than 4e4 M before rollover.
struct StateA0 { cflDt: atomic<u32>, cflTime: atomic<u32>, recoveryFails: atomic<u32>, pad1: u32, data: array<vec4f> }
@group(0) @binding(1) var<storage, read_write> primA0: StateA0;
@group(0) @binding(2) var<storage, read_write> primA1_raw: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> primA2_raw: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> primA3_raw: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> primB0_raw: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> primB1_raw: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> primB2_raw: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> primB3_raw: array<vec4f>;

fn wrap(i: i32, n: i32) -> i32 { var q = i % n; if (q < 0) { q += n; } return q; }
fn slab_idx(ir: i32, it: i32, ip: i32) -> vec2u {
    let n1 = i32(params.n1); let n2 = i32(params.n2); let n3 = i32(params.n3);
    let slabPhi = clamp(i32(params.phiSplit), 1, n3);
    let p = wrap(ip, n3); let slab = clamp(p / slabPhi, 0, clamp(i32(params.tiledStorage), 1, 4) - 1);
    return vec2u(u32(slab), u32(((p - slab * slabPhi) * n2 + clamp(it, 0, n2 - 1)) * n1 + clamp(ir, 0, n1 - 1)));
}
fn readA(ir: i32, it: i32, ip: i32) -> HarmPrim {
    let s = slab_idx(ir,it,ip); let k = 2u*s.y; var p: HarmPrim;
    if (s.x==0u) { p.state0=primA0.data[k]; p.state1=primA0.data[k+1u]; }
    else if (s.x==1u) { p.state0=primA1_raw[k]; p.state1=primA1_raw[k+1u]; }
    else if (s.x==2u) { p.state0=primA2_raw[k]; p.state1=primA2_raw[k+1u]; }
    else { p.state0=primA3_raw[k]; p.state1=primA3_raw[k+1u]; } return p;
}
fn readB(ir: i32, it: i32, ip: i32) -> HarmPrim {
    let s = slab_idx(ir,it,ip); let k = 2u*s.y; var p: HarmPrim;
    if (s.x==0u) { p.state0=primB0_raw[k]; p.state1=primB0_raw[k+1u]; }
    else if (s.x==1u) { p.state0=primB1_raw[k]; p.state1=primB1_raw[k+1u]; }
    else if (s.x==2u) { p.state0=primB2_raw[k]; p.state1=primB2_raw[k+1u]; }
    else { p.state0=primB3_raw[k]; p.state1=primB3_raw[k+1u]; } return p;
}
fn writeA(ir:i32,it:i32,ip:i32,p:HarmPrim) { let s=slab_idx(ir,it,ip); let k=2u*s.y;
    if(s.x==0u){primA0.data[k]=p.state0;primA0.data[k+1u]=p.state1;} else if(s.x==1u){primA1_raw[k]=p.state0;primA1_raw[k+1u]=p.state1;}
    else if(s.x==2u){primA2_raw[k]=p.state0;primA2_raw[k+1u]=p.state1;} else{primA3_raw[k]=p.state0;primA3_raw[k+1u]=p.state1;} }
fn writeB(ir:i32,it:i32,ip:i32,p:HarmPrim) { let s=slab_idx(ir,it,ip); let k=2u*s.y;
    if(s.x==0u){primB0_raw[k]=p.state0;primB0_raw[k+1u]=p.state1;} else if(s.x==1u){primB1_raw[k]=p.state0;primB1_raw[k+1u]=p.state1;}
    else if(s.x==2u){primB2_raw[k]=p.state0;primB2_raw[k+1u]=p.state1;} else{primB3_raw[k]=p.state0;primB3_raw[k+1u]=p.state1;} }
fn sourcePrim(useB: bool, ir:i32,it:i32,ip:i32)->HarmPrim { if(useB){return readB(ir,it,ip);} return readA(ir,it,ip); }

fn radius(ir:i32)->f32 { let n=max(i32(params.n1),1); let x=(f32(clamp(ir,0,n-1))+.5)/f32(n); return exp(log(params.rin)+x*(log(params.rout)-log(params.rin))); }
fn theta(it:i32)->f32 { let n=max(i32(params.n2),1); return (f32(clamp(it,0,n-1))+.5)/f32(n)*3.141592653589793; }
fn dr_at(ir:i32)->f32 { return max(radius(ir)*(log(params.rout)-log(params.rin))/f32(max(params.n1,1u)),1e-4); }
fn dtheta()->f32 { return 3.141592653589793/f32(max(params.n2,1u)); }
fn dphi()->f32 { return 6.283185307179586/f32(max(params.n3,1u)); }
fn rho_floor(r:f32)->f32 { return params.rhoFloor*pow(max(r,1.0),-1.5); }
fn u_floor(r:f32)->f32 { return params.uFloor*pow(max(r,1.0),-2.5); }

struct Mat4 { r0:vec4f, r1:vec4f, r2:vec4f, r3:vec4f }
fn gcov(r:f32,th:f32)->Mat4 {
    let a=clamp(params.spin,-.999,.999); let s=max(sin(th),1e-4); let s2=s*s; let sig=r*r+a*a*cos(th)*cos(th); let q=2.0*r/max(sig,1e-8);
    var m:Mat4;m.r0=vec4f(-(1.0-q),q,0.0,-q*a*s2);m.r1=vec4f(q,1.0+q,0.0,-(1.0+q)*a*s2);
    m.r2=vec4f(0.0,0.0,sig,0.0);m.r3=vec4f(-q*a*s2,-(1.0+q)*a*s2,0.0,(r*r+a*a+q*a*a*s2)*s2);return m;
}
fn gcon(r:f32,th:f32)->Mat4 {
    let a=clamp(params.spin,-.999,.999); let s=max(sin(th),1e-4); let sig=r*r+a*a*cos(th)*cos(th); let q=2.0*r/max(sig,1e-8);
    var m:Mat4;m.r0=vec4f(-(1.0+q),q,0.0,0.0);m.r1=vec4f(q,(r*r-2.0*r+a*a)/sig,0.0,a/sig);
    m.r2=vec4f(0.0,0.0,1.0/sig,0.0);m.r3=vec4f(0.0,a/sig,0.0,1.0/(sig*s*s));return m;
}
fn sqrtg(r:f32,th:f32)->f32 { let a=clamp(params.spin,-.999,.999); return max((r*r+a*a*cos(th)*cos(th))*max(sin(th),1e-4),1e-8); }
fn pressure(p:HarmPrim,r:f32)->f32 { return .33333334*max(p.state0.y,u_floor(r)); }
fn vel(p:HarmPrim)->vec3f { return vec3f(p.state0.z,p.state0.w,p.state1.x); }
fn mag(p:HarmPrim)->vec3f { return p.state1.yzw; }

fn timelike(p0:HarmPrim,r:f32,th:f32)->HarmPrim {
    var p=p0; p.state0.x=max(p.state0.x,rho_floor(r)); p.state0.y=max(p.state0.y,u_floor(r));
    p.state0.z=clamp(p.state0.z,-.999,.999);p.state0.w=clamp(p.state0.w,-.999,.999); p.state1.x=clamp(p.state1.x,-.999,.999);
    let g=gcov(r,th);let gi=gcon(r,th);let alpha=inverseSqrt(max(-gi.r0.x,1e-8));let normal=-alpha*alpha*gi.r0.yzw;let delta=vel(p)-normal;
    var v=vel(p);var n=g.r0.x+2.0*(g.r0.y*v.x+g.r0.z*v.y+g.r0.w*v.z)+g.r1.y*v.x*v.x+g.r2.z*v.y*v.y+g.r3.w*v.z*v.z+2.0*(g.r1.z*v.x*v.y+g.r1.w*v.x*v.z+g.r2.w*v.y*v.z);
    let targetNorm=-alpha*alpha/2500.0;
    if(n>targetNorm){var lo=0.0;var hi=1.0;for(var k=0;k<32;k++){let mid=.5*(lo+hi);let vm=normal+mid*delta;let nm=g.r0.x+2.0*(g.r0.y*vm.x+g.r0.z*vm.y+g.r0.w*vm.z)+g.r1.y*vm.x*vm.x+g.r2.z*vm.y*vm.y+g.r3.w*vm.z*vm.z+2.0*(g.r1.z*vm.x*vm.y+g.r1.w*vm.x*vm.z+g.r2.w*vm.y*vm.z);if(nm<targetNorm){lo=mid;}else{hi=mid;}}v=normal+.995*lo*delta;p.state0.z=v.x;p.state0.w=v.y;p.state1.x=v.z;}
    return p;
}
fn ucon_of(p0:HarmPrim,r:f32,th:f32)->vec4f {
    let p=timelike(p0,r,th); let g=gcov(r,th); let v=vel(p); var n=g.r0.x+2.0*(g.r0.y*v.x+g.r0.z*v.y+g.r0.w*v.z)+g.r1.y*v.x*v.x+g.r2.z*v.y*v.y+g.r3.w*v.z*v.z+2.0*(g.r1.z*v.x*v.y+g.r1.w*v.x*v.z+g.r2.w*v.y*v.z);
    let ut=inverseSqrt(max(-n,1e-10)); return vec4f(ut,ut*v.x,ut*v.y,ut*v.z);
}
fn bcon_of(p0:HarmPrim,r:f32,th:f32,u:vec4f)->vec4f {
    let p=timelike(p0,r,th); let g=gcov(r,th); let B=mag(p);
    let uc1=dot(g.r1,u);let uc2=dot(g.r2,u);let uc3=dot(g.r3,u);let b0=B.x*uc1+B.y*uc2+B.z*uc3;
    return vec4f(b0,(B.x+b0*u.y)/u.x,(B.y+b0*u.z)/u.x,(B.z+b0*u.w)/u.x);
}
fn dot4(g:Mat4,a:vec4f,b:vec4f)->f32 { return a.x*dot(g.r0,b)+a.y*dot(g.r1,b)+a.z*dot(g.r2,b)+a.w*dot(g.r3,b); }
fn stress(p0:HarmPrim,r:f32,th:f32)->Mat4 {
    let p=timelike(p0,r,th); let g=gcov(r,th);let gi=gcon(r,th);let u=ucon_of(p,r,th);let b=bcon_of(p,r,th,u);
    let b2=max(dot4(g,b,b),0.0);let pg=pressure(p,r);let w=p.state0.x+p.state0.y+pg+b2;let pt=pg+.5*b2;var T:Mat4;
    T.r0=w*u.x*u+pt*gi.r0-b.x*b;T.r1=w*u.y*u+pt*gi.r1-b.y*b;T.r2=w*u.z*u+pt*gi.r2-b.z*b;T.r3=w*u.w*u+pt*gi.r3-b.w*b;return T;
}
fn prim_to_cons(p0:HarmPrim,r:f32,th:f32)->HarmCons {
    let p=timelike(p0,r,th);let g=gcov(r,th);let u=ucon_of(p,r,th);let T=stress(p,r,th);let sg=sqrtg(r,th);var c:HarmCons;
    c.d=sg*p.state0.x*u.x;c.s1=sg*dot(T.r0,vec4f(g.r0.y,g.r1.y,g.r2.y,g.r3.y));c.s2=sg*dot(T.r0,vec4f(g.r0.z,g.r1.z,g.r2.z,g.r3.z));c.s3=sg*dot(T.r0,vec4f(g.r0.w,g.r1.w,g.r2.w,g.r3.w));
    c.tau=sg*(-dot(T.r0,vec4f(g.r0.x,g.r1.x,g.r2.x,g.r3.x))-p.state0.x*u.x);c.b1=sg*p.state1.y;c.b2=sg*p.state1.z;c.b3=sg*p.state1.w;return c;
}
fn flux(p0:HarmPrim,r:f32,th:f32,dir:u32)->HarmCons {
    let p=timelike(p0,r,th);let g=gcov(r,th);let u=ucon_of(p,r,th);let b=bcon_of(p,r,th,u);let T=stress(p,r,th);let sg=sqrtg(r,th);var row=T.r1;var ui=u.y;var bi=b.y;if(dir==1u){row=T.r2;ui=u.z;bi=b.z;}if(dir==2u){row=T.r3;ui=u.w;bi=b.w;}var f:HarmCons;
    let rhoUi=p.state0.x*ui;f.d=sg*rhoUi;f.s1=sg*dot(row,vec4f(g.r0.y,g.r1.y,g.r2.y,g.r3.y));f.s2=sg*dot(row,vec4f(g.r0.z,g.r1.z,g.r2.z,g.r3.z));f.s3=sg*dot(row,vec4f(g.r0.w,g.r1.w,g.r2.w,g.r3.w));f.tau=sg*(-dot(row,vec4f(g.r0.x,g.r1.x,g.r2.x,g.r3.x))-rhoUi);
    let bf=sg*vec3f(b.y*ui-bi*u.y,b.z*ui-bi*u.z,b.w*ui-bi*u.w);f.b1=bf.x;f.b2=bf.y;f.b3=bf.z;
    if(dir==0u){f.b1=0.0;}if(dir==1u){f.b2=0.0;}if(dir==2u){f.b3=0.0;}return f;
}
fn addc(a:HarmCons,b:HarmCons,s:f32)->HarmCons { var c:HarmCons;c.d=a.d+s*b.d;c.s1=a.s1+s*b.s1;c.s2=a.s2+s*b.s2;c.s3=a.s3+s*b.s3;c.tau=a.tau+s*b.tau;c.b1=a.b1+s*b.b1;c.b2=a.b2+s*b.b2;c.b3=a.b3+s*b.b3;return c; }
fn subc(a:HarmCons,b:HarmCons)->HarmCons{return addc(a,b,-1.0);}
fn wavespeed(p:HarmPrim,r:f32,th:f32,dir:u32)->f32 {
    let g=gcov(r,th);let gi=gcon(r,th);let u=ucon_of(p,r,th);let b=bcon_of(p,r,th,u);let b2=max(dot4(g,b,b),0.0);let pg=pressure(p,r);
    let h=max(p.state0.x+p.state0.y+pg+b2,p.state0.x);let cs2=clamp(1.3333334*pg/h,0.0,.66);let va2=clamp(b2/h,0.0,.95);let cf=sqrt(clamp(cs2+va2-cs2*va2,0.0,.98));
    let alpha=inverseSqrt(max(-gi.r0.x,1e-8));var vd=vel(p).x;var gd=gi.r1.y;if(dir==1u){vd=vel(p).y;gd=gi.r2.z;}if(dir==2u){vd=vel(p).z;gd=gi.r3.w;}return clamp(abs(vd)+cf*alpha*sqrt(max(gd,1e-10)),1e-5,4.0);
}
fn local_cfl_dt(p:HarmPrim,r:f32,th:f32,ir:i32)->f32 {
    let ar=wavespeed(p,r,th,0u)/max(dr_at(ir),1e-8);
    let at=wavespeed(p,r,th,1u)/max(dtheta(),1e-8);
    let ap=wavespeed(p,r,th,2u)/max(dphi(),1e-8);
    // This implementation's cell-centred CT/recovery coupling requires a
    // considerably stricter safety factor than a production staggered-CT
    // scheme.  The strict 0.008 factor is calibrated by the three-minute
    // stress gate below and still contracts as characteristic speeds grow.
    return max(min(.008/max(ar+at+ap,1e-8),params.dt),1e-8);
}
fn adaptive_dt()->f32 { return bitcast<f32>(atomicLoad(&primA0.cflDt)); }
fn hll(l:HarmPrim,rp:HarmPrim,r:f32,th:f32,dir:u32)->HarmCons {let ul=prim_to_cons(l,r,th);let ur=prim_to_cons(rp,r,th);let fl=flux(l,r,th,dir);let fr=flux(rp,r,th,dir);let a=max(wavespeed(l,r,th,dir),wavespeed(rp,r,th,dir));var q=addc(fl,fr,1.0);q=addc(q,subc(ur,ul),-a);var z:HarmCons;z.d=0.;z.s1=0.;z.s2=0.;z.s3=0.;z.tau=0.;z.b1=0.;z.b2=0.;z.b3=0.;return addc(z,q,.5);}

fn objective(p:HarmPrim,goal:HarmCons,r:f32,th:f32)->f32 {let c=prim_to_cons(p,r,th);let den=abs(goal.d)+length(vec3f(goal.s1,goal.s2,goal.s3))+abs(goal.tau)+1e-20;return (abs(c.d-goal.d)+length(vec3f(c.s1-goal.s1,c.s2-goal.s2,c.s3-goal.s3))+abs(c.tau-goal.tau))/den;}
fn recover(goal:HarmCons,old0:HarmPrim,r:f32,th:f32)->RecoverResult {
    var p=timelike(old0,r,th);let sg=sqrtg(r,th);p.state1.y=goal.b1/sg;p.state1.z=goal.b2/sg;p.state1.w=goal.b3/sg;
    var best=objective(p,goal,r,th);
    for(var k=0;k<10;k++){let c=prim_to_cons(p,r,th);let dscale=max(abs(goal.d),1e-12);let escale=max(abs(goal.tau)+abs(goal.d),1e-12);
        var step=1.0;var accepted=false;
        for(var trial=0;trial<9;trial++){var q=p;
            let rr=clamp(goal.d/max(c.d,1e-20),.8,1.2);let er=clamp((goal.tau+goal.d)/max(c.tau+c.d,1e-20),.8,1.2);
            q.state0.x=max(q.state0.x*mix(1.0,rr,step),rho_floor(r));q.state0.y=max(q.state0.y*mix(1.0,er,step),u_floor(r));
            let momErr=vec3f(goal.s1-c.s1,goal.s2-c.s2,goal.s3-c.s3)/max(dscale+escale,1e-10);
            q.state0.z+=step*.04*momErr.x;q.state0.w+=step*.04*momErr.y;q.state1.x+=step*.04*momErr.z;q=timelike(q,r,th);
            let score=objective(q,goal,r,th);if(score<best){p=q;best=score;accepted=true;break;}step*=.5;
        }
        if(!accepted){break;}
    }
    var result:RecoverResult;result.prim=p;
    let resolved=old0.state0.x>8.0*rho_floor(r)||old0.state0.y>8.0*u_floor(r);
    result.failed=select(0u,1u,best>5e-4&&resolved);return result;
}
fn contract(a:Mat4,b:Mat4)->f32{return dot(a.r0,b.r0)+dot(a.r1,b.r1)+dot(a.r2,b.r2)+dot(a.r3,b.r3);}
fn subm(a:Mat4,b:Mat4,s:f32)->Mat4{var m:Mat4;m.r0=(a.r0-b.r0)*s;m.r1=(a.r1-b.r1)*s;m.r2=(a.r2-b.r2)*s;m.r3=(a.r3-b.r3)*s;return m;}
fn source(p:HarmPrim,r:f32,th:f32)->vec3f {let T=stress(p,r,th);let hr=max(1e-4,1e-3*max(r,1.0));let ht=1e-4;let dgdr=subm(gcov(r+hr,th),gcov(max(r-hr,1e-4),th),1.0/(2.0*hr));let dgdt=subm(gcov(r,min(th+ht,3.14149265)),gcov(r,max(th-ht,1e-4)),1.0/(2.0*ht));return .5*sqrtg(r,th)*vec3f(contract(T,dgdr),contract(T,dgdt),0.0);}

fn minmod(a:f32,b:f32)->f32{if(a*b<=0.0){return 0.0;}return select(max(a,b),min(a,b),a>0.0);}
fn mc(a:f32,b:f32)->f32{return minmod(.5*(a+b),minmod(2.0*a,2.0*b));}
fn recon(l:HarmPrim,c:HarmPrim,r:HarmPrim,side:f32)->HarmPrim{var p:HarmPrim;p.state0=c.state0+side*vec4f(mc(c.state0.x-l.state0.x,r.state0.x-c.state0.x),mc(c.state0.y-l.state0.y,r.state0.y-c.state0.y),mc(c.state0.z-l.state0.z,r.state0.z-c.state0.z),mc(c.state0.w-l.state0.w,r.state0.w-c.state0.w));p.state1=c.state1+side*vec4f(mc(c.state1.x-l.state1.x,r.state1.x-c.state1.x),mc(c.state1.y-l.state1.y,r.state1.y-c.state1.y),mc(c.state1.z-l.state1.z,r.state1.z-c.state1.z),mc(c.state1.w-l.state1.w,r.state1.w-c.state1.w));return p;}
fn face(useB:bool,ir:i32,it:i32,ip:i32,dir:u32,side:f32)->HarmPrim{let c=sourcePrim(useB,ir,it,ip);if(params.highOrder<.5){return c;}let d=select(select(vec3i(0,0,1),vec3i(0,1,0),dir==1u),vec3i(1,0,0),dir==0u);return recon(sourcePrim(useB,ir-d.x,it-d.y,ip-d.z),c,sourcePrim(useB,ir+d.x,it+d.y,ip+d.z),side);}
fn emf(useB:bool,ir:i32,it:i32,ip:i32)->vec3f{let p=sourcePrim(useB,ir,it,ip);return -sqrtg(radius(ir),theta(it))*cross(vel(p),mag(p));}
fn edgeEr(useB:bool,ir:i32,it:i32,ip:i32)->f32{return .25*(emf(useB,ir,it,ip).x+emf(useB,ir,it-1,ip).x+emf(useB,ir,it,ip-1).x+emf(useB,ir,it-1,ip-1).x);}
fn edgeEt(useB:bool,ir:i32,it:i32,ip:i32)->f32{return .25*(emf(useB,ir,it,ip).y+emf(useB,ir-1,it,ip).y+emf(useB,ir,it,ip-1).y+emf(useB,ir-1,it,ip-1).y);}
fn edgeEp(useB:bool,ir:i32,it:i32,ip:i32)->f32{return .25*(emf(useB,ir,it,ip).z+emf(useB,ir-1,it,ip).z+emf(useB,ir,it-1,ip).z+emf(useB,ir-1,it-1,ip).z);}
fn applyBoundary(p0:HarmPrim,ir:i32,it:i32)->HarmPrim{var p=p0;if(ir==0){p.state0.z=min(p.state0.z,0.0);}if(ir==i32(params.n1)-1){p.state0.z=max(p.state0.z,0.0);}if(min(it,i32(params.n2)-1-it)==0){p.state0.w=0.0;}return p;}

fn evolveCell(useB:bool,ir:i32,it:i32,ip:i32,dt:f32)->RecoverResult {
    let r=radius(ir);let th=theta(it);let dr=dr_at(ir);let dth=dtheta();let dph=dphi();let c=sourcePrim(useB,ir,it,ip);
    let frm=hll(face(useB,ir-1,it,ip,0u,.5),face(useB,ir,it,ip,0u,-.5),r,th,0u);let frp=hll(face(useB,ir,it,ip,0u,.5),face(useB,ir+1,it,ip,0u,-.5),r,th,0u);
    let ftm=hll(face(useB,ir,it-1,ip,1u,.5),face(useB,ir,it,ip,1u,-.5),r,th,1u);let ftp=hll(face(useB,ir,it,ip,1u,.5),face(useB,ir,it+1,ip,1u,-.5),r,th,1u);
    let fpm=hll(face(useB,ir,it,ip-1,2u,.5),face(useB,ir,it,ip,2u,-.5),r,th,2u);let fpp=hll(face(useB,ir,it,ip,2u,.5),face(useB,ir,it,ip+1,2u,-.5),r,th,2u);
    var u=prim_to_cons(readA(ir,it,ip),r,th);u=addc(u,subc(frm,frp),dt/dr);u=addc(u,subc(ftm,ftp),dt/dth);u=addc(u,subc(fpm,fpp),dt/dph);let s=source(c,r,th);u.s1+=dt*s.x;u.s2+=dt*s.y;
    let base=prim_to_cons(readA(ir,it,ip),r,th);var bd=vec3f(base.b1,base.b2,base.b3);let eptp=edgeEp(useB,ir,it+1,ip);let eptm=edgeEp(useB,ir,it,ip);let etpp=edgeEt(useB,ir,it,ip+1);let etpm=edgeEt(useB,ir,it,ip);let erpp=edgeEr(useB,ir,it,ip+1);let erpm=edgeEr(useB,ir,it,ip);let eprp=edgeEp(useB,ir+1,it,ip);let eprm=edgeEp(useB,ir,it,ip);let etrp=edgeEt(useB,ir+1,it,ip);let etrm=edgeEt(useB,ir,it,ip);let ertp=edgeEr(useB,ir,it+1,ip);let ertm=edgeEr(useB,ir,it,ip);
    bd.x-=dt*((eptp-eptm)/dth-(etpp-etpm)/dph);bd.y-=dt*((erpp-erpm)/dph-(eprp-eprm)/dr);bd.z-=dt*((etrp-etrm)/dr-(ertp-ertm)/dth);u.b1=bd.x;u.b2=bd.y;u.b3=bd.z;
    var result=recover(u,c,r,th);result.prim=applyBoundary(result.prim,ir,it);return result;
}

@compute @workgroup_size(1) fn reset_cfl(@builtin(global_invocation_id) gid:vec3u){if(gid.x==0u){atomicStore(&primA0.cflDt,bitcast<u32>(params.dt));atomicStore(&primA0.recoveryFails,0u);}}
var<workgroup> cflScratch: array<u32,256>;
@compute @workgroup_size(8,8,4) fn reduce_cfl(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lid:u32){
    var candidate=params.dt;
    if(all(gid<vec3u(params.n1,params.n2,params.n3))){let ir=i32(gid.x);let it=i32(gid.y);let ip=i32(gid.z);candidate=local_cfl_dt(readA(ir,it,ip),radius(ir),theta(it),ir);}
    cflScratch[lid]=bitcast<u32>(candidate);workgroupBarrier();
    var stride=128u;loop{if(lid<stride){cflScratch[lid]=min(cflScratch[lid],cflScratch[lid+stride]);}workgroupBarrier();if(stride==1u){break;}stride/=2u;}
    if(lid==0u){atomicMin(&primA0.cflDt,cflScratch[0]);}
}
@compute @workgroup_size(8,8,4) fn harm_step(@builtin(global_invocation_id) gid:vec3u){if(any(gid>=vec3u(params.n1,params.n2,params.n3))){return;}let result=evolveCell(false,i32(gid.x),i32(gid.y),i32(gid.z),.5*adaptive_dt());writeB(i32(gid.x),i32(gid.y),i32(gid.z),result.prim);}
@compute @workgroup_size(8,8,4) fn harm_midpoint(@builtin(global_invocation_id) gid:vec3u){if(any(gid>=vec3u(params.n1,params.n2,params.n3))){return;}let result=evolveCell(true,i32(gid.x),i32(gid.y),i32(gid.z),adaptive_dt());writeA(i32(gid.x),i32(gid.y),i32(gid.z),result.prim);if(result.failed!=0u){atomicAdd(&primA0.recoveryFails,1u);}}
@compute @workgroup_size(8,8,4) fn sync_a_to_b(@builtin(global_invocation_id) gid:vec3u){if(any(gid>=vec3u(params.n1,params.n2,params.n3))){return;}writeB(i32(gid.x),i32(gid.y),i32(gid.z),readA(i32(gid.x),i32(gid.y),i32(gid.z)));if(all(gid==vec3u(0u))){atomicAdd(&primA0.cflTime,u32(round(adaptive_dt()*1e5)));}}
