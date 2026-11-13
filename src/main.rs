use std::env;
use rayon::prelude::*;

const ALPHA: usize = 6;
const EOS: usize = 6;
const VOCAB: usize = 7;
const LMAX: usize = 40;
const W0_CAP: usize = 16;
const PROD_CAP: usize = 4;
const SLOT: usize = ALPHA + 1;
const LAYOUT: usize = 288;
const DIM: usize = 16;
const N_LENS: usize = 2;
const LENS_DIM: usize = N_LENS * DIM;
const N_COMP: usize = 6;
const COMP: usize = N_COMP * DIM;
const R: usize = COMP;
const STEP_BUDGET: usize = 512;
const RHO: f32 = 0.75;
const MAX_ATT: usize = 10000;
const N_CAL: usize = 8000;

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s.wrapping_add(0x9E3779B97F4A7C15)) }
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 { ((self.u64() >> 40) as f32) / ((1u64 << 24) as f32) }
    fn below(&mut self, n: usize) -> usize { (self.u64() % (n as u64)) as usize }
    fn bit(&mut self) -> u64 { self.u64() & 1 }
    fn geom(&mut self, c: usize) -> usize { let mut k = 1; while self.bit() == 0 && k < c { k += 1; } k }
}

fn matvec(o: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
    for r in 0..rows { let mut s = 0f32; let b = r * cols; for c in 0..cols { s += w[b + c] * x[c]; } o[r] = s; }
}
fn outer_add(gw: &mut [f32], gy: &[f32], x: &[f32], rows: usize, cols: usize) {
    for r in 0..rows { let g = gy[r]; if g == 0.0 { continue; } let b = r * cols; for c in 0..cols { gw[b + c] += g * x[c]; } }
}
fn transvec_add(o: &mut [f32], w: &[f32], gy: &[f32], rows: usize, cols: usize) {
    for r in 0..rows { let g = gy[r]; if g == 0.0 { continue; } let b = r * cols; for c in 0..cols { o[c] += w[b + c] * g; } }
}
fn tanh_ip(x: &mut [f32]) { for v in x.iter_mut() { *v = v.tanh(); } }
fn softmax(logits: &[f32], prob: &mut [f32]) {
    let mut mx = f32::NEG_INFINITY; for &v in logits { mx = mx.max(v); }
    let mut s = 0f32; for i in 0..logits.len() { prob[i] = (logits[i] - mx).exp(); s += prob[i]; }
    for p in prob.iter_mut() { *p /= s; }
}

// ---------- générateur 2-tag avec provenance ----------
#[derive(Clone)]
struct Program { w0: Vec<u8>, prod: Vec<Vec<u8>> }
struct World { prog: Program, effect: Vec<u8>, prov: Vec<u16>, depth: usize, dead: usize, logw: f32, mask: Vec<f32>, obs: Vec<bool> }

fn sample_program(rng: &mut Rng, lp: &mut f32) -> Program {
    let la = (ALPHA as f32).ln(); let l2 = 2f32.ln();
    let l0 = 4 + rng.geom(12); *lp -= (l0 as f32 - 4.0) * l2;
    let mut w0 = Vec::new(); for _ in 0..l0 { w0.push(rng.below(ALPHA) as u8); *lp -= la; }
    let mut prod = Vec::with_capacity(ALPHA);
    for _ in 0..ALPHA {
        let l = rng.geom(4); *lp -= (l as f32) * l2;
        let mut r = Vec::new(); for _ in 0..l { r.push(rng.below(ALPHA) as u8); *lp -= la; }
        prod.push(r);
    }
    Program { w0, prod }
}

fn execute(p: &Program, budget: usize) -> (Vec<u8>, Vec<u16>, bool) {
    let mut word: Vec<(u8, u16)> = p.w0.iter().map(|&s| (s, 0)).collect();
    let mut eff = Vec::new(); let mut prov = Vec::new(); let mut halted = false;
    let mut head = 0usize;
    for _ in 0..budget {
        if word.len() - head < 2 { halted = true; break; }
        let (b, kb) = word[head]; eff.push(b); prov.push(kb);
        head += 2;                                   // O(1) au lieu de drain O(n)
        for &x in &p.prod[b as usize] { word.push((x, kb + 1)); }
        if eff.len() >= LMAX { break; }
    }
    (eff, prov, halted)
}

fn mutate_surgery(p: &Program, rng: &mut Rng, lp: &mut f32) -> Program {
    let mut q = p.clone(); let mut sites: Vec<(usize, usize)> = Vec::new();
    for i in 0..q.w0.len() { sites.push((usize::MAX, i)); }
    for s in 0..ALPHA { for i in 0..q.prod[s].len() { sites.push((s, i)); } }
    if sites.is_empty() { return q; }
    let (s, i) = sites[rng.below(sites.len())];
    *lp -= ((sites.len() * (ALPHA - 1)) as f32).ln();
    let cur = if s == usize::MAX { q.w0[i] } else { q.prod[s][i] };
    let mut ns = rng.below(ALPHA - 1) as u8; if ns >= cur { ns += 1; }
    if s == usize::MAX { q.w0[i] = ns; } else { q.prod[s][i] = ns; }
    q
}
fn mutate_null(p: &Program, rng: &mut Rng, lp: &mut f32) -> Program {
    let mut perm: Vec<u8> = (0..ALPHA as u8).collect();
    for i in (1..ALPHA).rev() { let j = rng.below(i + 1); perm.swap(i, j); }
    let mut lf = 0f32; for k in 2..=ALPHA { lf += (k as f32).ln(); } *lp -= lf;
    let ap = |w: &Vec<u8>| -> Vec<u8> { w.iter().map(|&x| perm[x as usize]).collect() };
    let mut np = vec![Vec::new(); ALPHA];
    for s in 0..ALPHA { np[perm[s as usize] as usize] = ap(&p.prod[s]); }
    Program { w0: ap(&p.w0), prod: np }
}

fn make_view(rng: &mut Rng, depth: usize, lp: &mut f32) -> (Vec<f32>, Vec<bool>) {
    let full = rng.f32() < 0.2;
    let mut mask = vec![0f32; LMAX]; let mut obs = vec![false; LMAX];
    if full {
        *lp += 0.2f32.ln();
        for p in 0..LMAX { mask[p] = 1.0; obs[p] = p <= depth; }
    } else {
        *lp += 0.8f32.ln();
        for p in 0..LMAX {
            let keep = rng.f32() < RHO; mask[p] = if keep { 1.0 } else { 0.0 };
            obs[p] = keep && p <= depth;
            *lp += if keep { RHO.ln() } else { (1.0 - RHO).ln() };
        }
    }
    (mask, obs)
}

fn build_world(rng: &mut Rng, prog: Program, effect: Vec<u8>, prov: Vec<u16>, depth: usize, lp: &mut f32) -> World {
    let mut fired = vec![false; ALPHA]; for &b in &effect { fired[b as usize] = true; }
    let dead = fired.iter().filter(|&&b| !b).count();
    let (mask, obs) = make_view(rng, depth, lp);
    World { prog, effect, prov, depth, dead, logw: *lp, mask, obs }
}
fn sample_world(rng: &mut Rng, kind: u8, lo: usize, hi: usize, p_acc: f32) -> Option<World> {
    if p_acc <= 0.0 { return None; }
    for _ in 0..MAX_ATT {
        let mut lp = 0f32; let base = sample_program(rng, &mut lp);
        let prog = match kind { 1 => mutate_surgery(&base, rng, &mut lp), 2 => mutate_null(&base, rng, &mut lp), _ => base };
        let (eff, prov, halted) = execute(&prog, STEP_BUDGET); let d = eff.len();
        if halted && d >= lo && d <= hi && d > 0 && d < LMAX {
            lp -= p_acc.ln(); return Some(build_world(rng, prog, eff, prov, d, &mut lp));
        }
    }
    None
}
fn calibrate(rng: &mut Rng, bands: &[(usize, usize); 4]) -> [[f32; 4]; 3] {
    let mut p = [[0f32; 4]; 3];
    for kind in 0..3u8 {
        let mut cnt = [0usize; 4];
        for _ in 0..N_CAL {
            let mut lp = 0f32; let base = sample_program(rng, &mut lp);
            let prog = match kind { 1 => mutate_surgery(&base, rng, &mut lp), 2 => mutate_null(&base, rng, &mut lp), _ => base };
            let (e, _, h) = execute(&prog, STEP_BUDGET); let d = e.len();
            if h && d > 0 && d < LMAX { for (bi, &(lo, hi)) in bands.iter().enumerate() { if d >= lo && d <= hi { cnt[bi] += 1; } } }
        }
        for bi in 0..4 { p[kind as usize][bi] = cnt[bi] as f32 / N_CAL as f32; }
    }
    p
}

// ---------- binding lossless gelé ----------
fn bind(p: &Program) -> Vec<f32> {
    let mut v = vec![0f32; LAYOUT]; let mut o = 0;
    for i in 0..W0_CAP { if i < p.w0.len() { let s = p.w0[i] as usize; v[o + s] = 1.0; v[o + ALPHA] = 1.0; } o += SLOT; }
    for s in 0..ALPHA { for j in 0..PROD_CAP { if j < p.prod[s].len() { let x = p.prod[s][j] as usize; v[o + x] = 1.0; v[o + ALPHA] = 1.0; } o += SLOT; } }
    v
}

#[derive(Clone, Copy, PartialEq)]
enum Link { Add, Mul }
struct Cfg { link: Link, fb: bool }

struct Layout {
    read: usize, lens: usize, rec: usize, bcomp: usize,
    ma: usize, mb: usize, mc: usize, wv: usize, wout: usize, bout: usize, fb: usize, total: usize,
}
fn layout() -> Layout {
    let mut o = 0; let mut take = |n: usize| { let s = o; o += n; s };
    let read = take(COMP * LAYOUT); let lens = take(COMP * LENS_DIM); let rec = take(COMP * COMP); let bcomp = take(COMP);
    let ma = take(R * LAYOUT); let mb = take(R * COMP); let mc = take(COMP * R);
    let wv = take(LENS_DIM * LMAX); let wout = take(VOCAB * COMP); let bout = take(VOCAB); let fb = take(COMP * VOCAB);
    Layout { read, lens, rec, bcomp, ma, mb, mc, wv, wout, bout, fb, total: o }
}
fn init_theta(rng: &mut Rng, l: &Layout) -> Vec<f32> {
    let mut th = vec![0f32; l.total];
    let mut fill = |off: usize, n: usize, s: f32| { for v in th[off..off + n].iter_mut() { *v = (rng.f32() - 0.5) * 2.0 * s; } };
    fill(l.read, COMP * LAYOUT, 1.0 / (LAYOUT as f32).sqrt());
    fill(l.lens, COMP * LENS_DIM, 1.0 / (LENS_DIM as f32).sqrt());
    fill(l.rec, COMP * COMP, 1.0 / (COMP as f32).sqrt());
    fill(l.ma, R * LAYOUT, 1.0 / (LAYOUT as f32).sqrt());
    fill(l.mb, R * COMP, 1.0 / (COMP as f32).sqrt());
    fill(l.mc, COMP * R, 1.0 / (R as f32).sqrt());
    fill(l.wv, LENS_DIM * LMAX, 1.0 / (LMAX as f32).sqrt());
    fill(l.wout, VOCAB * COMP, 1.0 / (COMP as f32).sqrt());
    fill(l.fb, COMP * VOCAB, 1.0 / (VOCAB as f32).sqrt());
    th
}

struct Fwd {
    cause: Vec<f32>, lens: Vec<f32>, h: Vec<f32>, prob: Vec<f32>, logits: Vec<f32>,
    bh: Vec<f32>, m: Vec<f32>, moda: Vec<f32>,
}
fn forward(th: &[f32], l: &Layout, cfg: &Cfg, p: &Program, mask: &[f32], ts: usize) -> Fwd {
    let cause = bind(p);
    let mut lens = vec![0f32; LENS_DIM];
    matvec(&mut lens, &th[l.wv..l.wv + LENS_DIM * LMAX], mask, LENS_DIM, LMAX); tanh_ip(&mut lens);
    let mut xc = vec![0f32; COMP]; matvec(&mut xc, &th[l.read..l.read + COMP * LAYOUT], &cause, COMP, LAYOUT);
    let mut xl = vec![0f32; COMP]; matvec(&mut xl, &th[l.lens..l.lens + COMP * LENS_DIM], &lens, COMP, LENS_DIM);
    let mut moda = vec![0f32; R];
    if cfg.link == Link::Mul { matvec(&mut moda, &th[l.ma..l.ma + R * LAYOUT], &cause, R, LAYOUT); }
    let mut h = vec![0f32; ts * COMP]; let mut prob = vec![0f32; ts * VOCAB]; let mut logits = vec![0f32; ts * VOCAB];
    let mut bh = vec![0f32; if cfg.link == Link::Mul { ts * R } else { 0 }];
    let mut m = vec![0f32; if cfg.link == Link::Mul { ts * R } else { 0 }];
    let mut cur = vec![0f32; COMP]; let mut pre = vec![0f32; COMP]; let mut tmp = vec![0f32; COMP]; let mut fbv = vec![0f32; COMP];
    let wrec = &th[l.rec..l.rec + COMP * COMP]; let bc = &th[l.bcomp..l.bcomp + COMP];
    let wout = &th[l.wout..l.wout + VOCAB * COMP]; let bout = &th[l.bout..l.bout + VOCAB];
    let mb = &th[l.mb..l.mb + R * COMP]; let mc = &th[l.mc..l.mc + COMP * R]; let wfb = &th[l.fb..l.fb + COMP * VOCAB];
    for t in 0..ts {
        pre.copy_from_slice(bc);
        for i in 0..COMP { pre[i] += xc[i] + xl[i]; }
        if cfg.fb && t > 0 { matvec(&mut fbv, wfb, &prob[(t - 1) * VOCAB..t * VOCAB], COMP, VOCAB); for i in 0..COMP { pre[i] += fbv[i]; } }
        matvec(&mut tmp, wrec, &cur, COMP, COMP); for i in 0..COMP { pre[i] += tmp[i]; }
        if cfg.link == Link::Mul {
            let bhs = &mut bh[t * R..(t + 1) * R]; matvec(bhs, mb, &cur, R, COMP);
            let ms = &mut m[t * R..(t + 1) * R]; for i in 0..R { ms[i] = moda[i] * bhs[i]; }
            matvec(&mut tmp, mc, ms, COMP, R); for i in 0..COMP { pre[i] += tmp[i]; }
        }
        cur.copy_from_slice(&pre); tanh_ip(&mut cur);
        h[t * COMP..(t + 1) * COMP].copy_from_slice(&cur);
        let lg = &mut logits[t * VOCAB..(t + 1) * VOCAB]; matvec(lg, wout, &cur, VOCAB, COMP);
        for v in 0..VOCAB { lg[v] += bout[v]; }
        softmax(lg, &mut prob[t * VOCAB..(t + 1) * VOCAB]);
    }
    Fwd { cause, lens, h, prob, logits, bh, m, moda }
}

fn backward(th: &[f32], l: &Layout, cfg: &Cfg, f: &Fwd, mask: &[f32], tg: &[usize], obs: &[bool], ts: usize, g: &mut [f32]) {
    let n_obs = obs.iter().filter(|&&b| b).count().max(1) as f32;
    let wrec = &th[l.rec..l.rec + COMP * COMP]; let wout = &th[l.wout..l.wout + VOCAB * COMP];
    let mb = &th[l.mb..l.mb + R * COMP]; let mc = &th[l.mc..l.mc + COMP * R]; let wfb = &th[l.fb..l.fb + COMP * VOCAB];
    let wlens = &th[l.lens..l.lens + COMP * LENS_DIM];
    let mut sum_gpre = vec![0f32; COMP]; let mut gmoda = vec![0f32; R];
    let mut gh_carry = vec![0f32; COMP]; let mut gprob_carry = vec![0f32; VOCAB];
    let mut gpre = vec![0f32; COMP]; let mut gh = vec![0f32; COMP]; let mut tmpc = vec![0f32; COMP];
    let mut gm = vec![0f32; R]; let mut gbh = vec![0f32; R]; let mut glog = [0f32; VOCAB]; let mut ng = vec![0f32; COMP];
    for t in (0..ts).rev() {
        let pt = &f.prob[t * VOCAB..(t + 1) * VOCAB];
        for v in 0..VOCAB { glog[v] = 0.0; }
        if obs[t] { for v in 0..VOCAB { glog[v] += (pt[v] - if v == tg[t] { 1.0 } else { 0.0 }) / n_obs; } }
        if cfg.fb {
            let mut dot = 0f32; for v in 0..VOCAB { dot += pt[v] * gprob_carry[v]; }
            for v in 0..VOCAB { glog[v] += pt[v] * (gprob_carry[v] - dot); }
        }
        let ht = &f.h[t * COMP..(t + 1) * COMP];
        outer_add(&mut g[l.wout..l.wout + VOCAB * COMP], &glog, ht, VOCAB, COMP);
        for v in 0..VOCAB { g[l.bout + v] += glog[v]; }
        for i in 0..COMP { gh[i] = gh_carry[i]; }
        transvec_add(&mut gh, wout, &glog, VOCAB, COMP);
        for i in 0..COMP { gpre[i] = gh[i] * (1.0 - ht[i] * ht[i]); }
        for i in 0..COMP { sum_gpre[i] += gpre[i]; }
        for i in 0..COMP { ng[i] = 0.0; }
        transvec_add(&mut ng, wrec, &gpre, COMP, COMP);
        if t > 0 { outer_add(&mut g[l.rec..l.rec + COMP * COMP], &gpre, &f.h[(t - 1) * COMP..t * COMP], COMP, COMP); }
        if cfg.link == Link::Mul {
            let mt = &f.m[t * R..(t + 1) * R]; let bht = &f.bh[t * R..(t + 1) * R];
            outer_add(&mut g[l.mc..l.mc + COMP * R], &gpre, mt, COMP, R);
            for i in 0..R { gm[i] = 0.0; } transvec_add(&mut gm, mc, &gpre, COMP, R);
            for i in 0..R { gmoda[i] += gm[i] * bht[i]; gbh[i] = gm[i] * f.moda[i]; }
            if t > 0 { outer_add(&mut g[l.mb..l.mb + R * COMP], &gbh, &f.h[(t - 1) * COMP..t * COMP], R, COMP); }
            transvec_add(&mut ng, mb, &gbh, R, COMP);
        }
        if cfg.fb {
            if t > 0 {
                outer_add(&mut g[l.fb..l.fb + COMP * VOCAB], &gpre, &f.prob[(t - 1) * VOCAB..t * VOCAB], COMP, VOCAB);
                for v in 0..VOCAB { gprob_carry[v] = 0.0; } transvec_add(&mut gprob_carry, wfb, &gpre, COMP, VOCAB);
            } else { for v in 0..VOCAB { gprob_carry[v] = 0.0; } }
        }
        gh_carry.copy_from_slice(&ng);
    }
    outer_add(&mut g[l.read..l.read + COMP * LAYOUT], &sum_gpre, &f.cause, COMP, LAYOUT);
    outer_add(&mut g[l.lens..l.lens + COMP * LENS_DIM], &sum_gpre, &f.lens, COMP, LENS_DIM);
    for i in 0..COMP { g[l.bcomp + i] += sum_gpre[i]; }
    let mut glens = vec![0f32; LENS_DIM]; transvec_add(&mut glens, wlens, &sum_gpre, COMP, LENS_DIM);
    let mut gpl = vec![0f32; LENS_DIM]; for i in 0..LENS_DIM { gpl[i] = glens[i] * (1.0 - f.lens[i] * f.lens[i]); }
    outer_add(&mut g[l.wv..l.wv + LENS_DIM * LMAX], &gpl, mask, LENS_DIM, LMAX);
    if cfg.link == Link::Mul { outer_add(&mut g[l.ma..l.ma + R * LAYOUT], &gmoda, &f.cause, R, LAYOUT); }
}

fn clip_adam(th: &mut [f32], g: &mut [f32], m: &mut [f32], v: &mut [f32], step: i32, lr: f32, clip: f32) {
    let mut nrm = 0f32; for &x in g.iter() { nrm += x * x; } nrm = nrm.sqrt();
    let sc = if nrm > clip { clip / nrm } else { 1.0 };
    let b1 = 0.9f32; let b2 = 0.999f32; let e = 1e-8f32;
    let c1 = 1.0 - b1.powi(step); let c2 = 1.0 - b2.powi(step);
    for i in 0..th.len() {
        let gi = g[i] * sc;
        m[i] = b1 * m[i] + (1.0 - b1) * gi; v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
        th[i] -= lr * (m[i] / c1) / ((v[i] / c2).sqrt() + e);
    }
}
fn targets_of(w: &World) -> Vec<usize> {
    let mut t = vec![EOS; LMAX]; for (p, &b) in w.effect.iter().enumerate() { t[p] = b as usize; }
    if w.depth < LMAX { t[w.depth] = EOS; } t
}
fn ce_loss(f: &Fwd, tg: &[usize], obs: &[bool], ts: usize) -> f32 {
    let mut s = 0f32; let mut n = 0u32;
    for t in 0..ts { if !obs[t] { continue; } s -= f.prob[t * VOCAB + tg[t]].max(1e-12).ln(); n += 1; }
    s / n.max(1) as f32
}

struct Met { n: usize, exact: u64, c_ok: u64, c_n: u64, l_ok: u64, l_n: u64, e_ok: u64, e_n: u64, base_acc: f64, base_ex: f64 }
impl Met {
    fn new() -> Self { Met { n: 0, exact: 0, c_ok: 0, c_n: 0, l_ok: 0, l_n: 0, e_ok: 0, e_n: 0, base_acc: 0.0, base_ex: 0.0 } }
    fn add(&mut self, w: &World, f: &Fwd, ts: usize) {
        let mut pred = vec![0usize; ts]; let mut that = ts;
        for t in 0..ts {
            let lg = &f.logits[t * VOCAB..(t + 1) * VOCAB];
            let mut b = 0; let mut bv = f32::NEG_INFINITY; for v in 0..VOCAB { if lg[v] > bv { bv = lg[v]; b = v; } }
            pred[t] = b; if b == EOS && that == ts { that = t; }
        }
        let mut all = that == w.depth;
        for t in 0..w.depth { if pred[t] != w.effect[t] as usize { all = false; } }
        if all { self.exact += 1; }
        let mut ncopy = 0u64; let mut nother = 0u64;
        for t in 0..=w.depth {
            if t >= ts { break; }
            let target = if t < w.depth { w.effect[t] as usize } else { EOS };
            let ok = pred[t] == target;
            if t == w.depth { self.e_n += 1; if ok { self.e_ok += 1; } }
            else if w.prov[t] == 0 { self.c_n += 1; if ok { self.c_ok += 1; } ncopy += 1; }
            else { self.l_n += 1; if ok { self.l_ok += 1; } nother += 1; }
        }
        let pos = (w.depth + 1) as f64;
        self.base_acc += (ncopy as f64 + 1.0 + nother as f64 / VOCAB as f64) / pos;
        self.base_ex += (1.0 / VOCAB as f64).powi(nother as i32);
        self.n += 1;
    }
    fn report(&self, name: &str) {
        let ex = self.exact as f64 / self.n.max(1) as f64;
        let ca = self.c_ok as f64 / self.c_n.max(1) as f64;
        let la = self.l_ok as f64 / self.l_n.max(1) as f64;
        let ea = self.e_ok as f64 / self.e_n.max(1) as f64;
        let ba = self.base_acc / self.n.max(1) as f64; let be = self.base_ex / self.n.max(1) as f64;
        println!("{name:10} n={:4} exact {:.3} | copie {:.3} lookup {:.3} (méca {:+.3}) EOS {:.3} | base acc {:.3} ex {:.3}",
            self.n, ex, ca, la, la - 1.0 / VOCAB as f64, ea, ba, be);
    }
}
fn eval_band(th: &[f32], l: &Layout, cfg: &Cfg, rng: &mut Rng, lo: usize, hi: usize, want: usize, p: f32, ts: usize) -> Met {
    let mut met = Met::new();
    while met.n < want {
        let w = match sample_world(rng, 0, lo, hi, p) { Some(w) => w, None => break };
        let mask = vec![1f32; LMAX]; // request-all, aucune info de longueur
        let f = forward(th, l, cfg, &w.prog, &mask, ts);
        met.add(&w, &f, ts);
    }
    met
}

fn main() {
    let mut worlds = 2_000_000usize; let mut batch = 64usize; let mut d = 8usize;
    let mut seeds = vec![1u64, 2, 3]; let mut lrs = vec![1e-3f32, 3e-3];
    let mut cfg = Cfg { link: Link::Mul, fb: false };
    let mut eval_n = 300usize; let mut clip = 1.0f32; let mut quick = false;
    let args: Vec<String> = env::args().collect(); let mut i = 1;
    while i < args.len() {
        let k = args[i].as_str(); let val = args.get(i + 1).cloned().unwrap_or_default();
        match k {
            "--worlds" => { worlds = val.parse().unwrap_or(worlds); i += 2; }
            "--batch" => { batch = val.parse().unwrap_or(batch); i += 2; }
            "--depth" => { d = val.parse().unwrap_or(d); i += 2; }
            "--eval-n" => { eval_n = val.parse().unwrap_or(eval_n); i += 2; }
            "--clip" => { clip = val.parse().unwrap_or(clip); i += 2; }
            "--link" => { cfg.link = if val == "add" { Link::Add } else { Link::Mul }; i += 2; }
            "--feedback" => { cfg.fb = val == "soft"; i += 2; }
            "--seeds" => { seeds = val.split(',').filter_map(|s| s.parse().ok()).collect(); i += 2; }
            "--lrs" => { lrs = val.split(',').filter_map(|s| s.parse().ok()).collect(); i += 2; }
            "--quick" => { quick = true; i += 1; }
            _ => { i += 1; }
        }
    }
    if quick { worlds = 200_000; seeds = vec![1]; lrs = vec![3e-3]; }
    let t_train = 2 * d + 8; let t_eval = 4 * d + 8;
    let bands = [(1, d), (d + 1, 2 * d), (2 * d + 1, 4 * d), (1, 2 * d)];
    let l = layout();
    eprintln!("M0 durci — params {} link={:?} fb={} d={d} T_train={t_train} T_eval={t_eval} worlds={worlds} batch={batch}", l.total, cfg.link as u8, cfg.fb);
    let mut agg_il = Vec::new(); let mut agg_ie = Vec::new(); let mut agg_ic = Vec::new();
let mut agg_ol = Vec::new(); let mut agg_iex = Vec::new(); let mut agg_oex = Vec::new();

    for &lr in &lrs {
        for &seed in &seeds {
            let mut rng = Rng::new(seed);
            let p_acc = calibrate(&mut rng, &bands);
            let mut th = init_theta(&mut rng, &l);
            let mut m = vec![0f32; th.len()]; let mut v = vec![0f32; th.len()]; let mut g = vec![0f32; th.len()];
            let steps = worlds / batch; let log_every = (steps / 10).max(1); let eval_every = (steps / 4).max(1);
            eprintln!("--- lr={lr} seed={seed} steps={steps} p_train={:.3}", p_acc[0][3]);
            for step in 1..=steps {
    let prog = step as f32 / steps as f32;
    let lr_t = lr * (0.1 + 0.45 * (1.0 + (std::f32::consts::PI * prog).cos()));

    // --- 1) échantillonnage séquentiel (utilise rng, non thread-safe) ---
    let mut batch_worlds: Vec<World> = Vec::with_capacity(batch);
    let mut att = 0;
    while batch_worlds.len() < batch && att < batch * 40 {
        att += 1;
        let kind = rng.below(3) as u8;
        if let Some(w) = sample_world(&mut rng, kind, 1, 2 * d,
                                       p_acc[kind as usize][3]) {
            batch_worlds.push(w);
        }
    }
    let n_done = batch_worlds.len().max(1) as f32;

    // --- 2) forward+backward en parallèle, grad accumulé par thread ---
    // backward est purement additif (+=), donc on peut accumuler
    // plusieurs mondes dans le même buffer local par thread.
    let (total_loss, mut total_g) = batch_worlds
        .par_iter()
        .fold(
            || (0f32, vec![0f32; th.len()]),
            |(mut acc_l, mut acc_g), w| {
                // gain x2.5 : on n'unroll que jusqu'à EOS
                let ts = (w.depth + 1).min(t_train);
                let tg = targets_of(w);
                let f  = forward(&th, &l, &cfg, &w.prog, &w.mask, ts);
                acc_l += ce_loss(&f, &tg, &w.obs, ts);
                backward(&th, &l, &cfg, &f, &w.mask, &tg, &w.obs, ts,
                         &mut acc_g);
                (acc_l, acc_g)
            },
        )
        .reduce(
            || (0f32, vec![0f32; th.len()]),
            |(l1, mut g1), (l2, g2)| {
                for i in 0..g1.len() { g1[i] += g2[i]; }
                (l1 + l2, g1)
            },
        );

    clip_adam(&mut th, &mut total_g, &mut m, &mut v, step as i32, lr_t, clip);

    if step % log_every == 0 || step == steps {
        eprintln!("step {step:6} loss {:.4} lr {:.2e}",
                  total_loss / n_done, lr_t);
    }
    if step % eval_every == 0 || step == steps {
        let mt = eval_band(&th, &l, &cfg, &mut rng, 1, 2 * d, 64,
                           p_acc[0][3], t_eval);
        eprintln!("   in-dist(1..{}) exact {:.3} lookup {:.3}",
                  2 * d,
                  mt.exact as f64 / mt.n.max(1) as f64,
                  mt.l_ok as f64 / mt.l_n.max(1) as f64);
    }
}


            println!("\n=== lr={lr} seed={seed} ===");
            let m0 = eval_band(&th, &l, &cfg, &mut rng, 1, d, eval_n, p_acc[0][0], t_eval); m0.report("1..d");
            let m1 = eval_band(&th, &l, &cfg, &mut rng, d + 1, 2 * d, eval_n, p_acc[0][1], t_eval); m1.report("d+1..2d");
            let m2 = eval_band(&th, &l, &cfg, &mut rng, 2 * d + 1, 4 * d, eval_n, p_acc[0][2], t_eval); m2.report("2d+1..4d");

            let il = (m0.l_ok + m1.l_ok) as f64 / (m0.l_n + m1.l_n).max(1) as f64;
let ie = (m0.e_ok + m1.e_ok) as f64 / (m0.e_n + m1.e_n).max(1) as f64;
let ic = (m0.c_ok + m1.c_ok) as f64 / (m0.c_n + m1.c_n).max(1) as f64;
let ol = m2.l_ok as f64 / m2.l_n.max(1) as f64;
let iex = (m0.exact + m1.exact) as f64 / (m0.n + m1.n).max(1) as f64;
let oex = m2.exact as f64 / m2.n.max(1) as f64;
agg_il.push(il); agg_ie.push(ie); agg_ic.push(ic); agg_ol.push(ol); agg_iex.push(iex); agg_oex.push(oex);

        }
    }
    let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len().max(1) as f64;
let (il, ie, ic, ol) = (mean(&agg_il), mean(&agg_ie), mean(&agg_ic), mean(&agg_ol));
println!("\nAGGÉGAT ({} runs): in-dist lookup {:.3} EOS {:.3} copie {:.3} exact {:.3} | OOD lookup {:.3} exact {:.3}",
    agg_il.len(), il, ie, ic, mean(&agg_iex), ol, mean(&agg_oex));
if il < 0.90 || ie < 0.95 || ic < 0.99 {
    println!("VERDICT: M0a' NON ATTEINT (lookup {:.3} / EOS {:.3} / copie {:.3}). D n'est pas un instrument fidèle.", il, ie, ic);
} else if ol < 0.50 || ol < 0.5 * il {
    println!("VERDICT: M0a' OK mais M0b' NON ATTEINT (OOD lookup {:.3} vs in-dist {:.3}). D fit mais ne compose pas en profondeur.", ol, il);
} else {
    println!("VERDICT: M0a' et M0b' OK. D est un banc fidèle et compose. Continuer M1.");
}
}