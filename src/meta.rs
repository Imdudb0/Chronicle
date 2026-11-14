// Méta-générateur : sampler + mutateur + traceur.
// Base : système normal de Post (tag system, m ∈ {1,2}) écrit dans l'alphabet de E.
use crate::rng::Rng;
use std::collections::VecDeque;

pub const K_MAX: usize = 6; // symboles d'alphabet max
pub const W_MAX: usize = 3; // longueur max d'une production
pub const Q_MAX: usize = 3; // longueur max de la file initiale
pub const NSLOT: usize = K_MAX + 1; // one-hot : indice de symbole ou « vide »
pub const NONE: usize = K_MAX;
pub const ROW: usize = 8 + 1 + W_MAX * NSLOT; // 30
pub const H_DIM: usize = 2 + K_MAX * ROW + Q_MAX * NSLOT; // 203
pub const V_DIM: usize = 3 + 4; // 7
pub const Z_DIM: usize = H_DIM + V_DIM; // 210

/// Cause C : m, octet de chaque symbole, production de chaque symbole, file initiale.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub m: usize,
    pub bytes: Vec<u8>,
    pub prod: Vec<Vec<u8>>,
    pub q0: Vec<u8>,
}

fn glyph(b: u8) -> char {
    if (0x20..0x7f).contains(&b) { b as char } else { '?' }
}

pub fn text(b: &[u8]) -> String {
    b.iter().map(|&x| glyph(x)).collect()
}

impl Program {
    pub fn k(&self) -> usize {
        self.bytes.len()
    }
    pub fn show(&self) -> String {
        let s = |v: &[u8]| -> String { v.iter().map(|&j| glyph(self.bytes[j as usize])).collect() };
        let rules: Vec<String> = (0..self.k())
            .map(|j| {
                let w = s(&self.prod[j]);
                format!("{}→{}", glyph(self.bytes[j]), if w.is_empty() { "ε".to_string() } else { w })
            })
            .collect();
        format!("m={} q0={} [{}]", self.m, s(&self.q0), rules.join(" "))
    }
}

/// Vue = crible E-aveugle : positions p < T avec p ≥ o et (p − o) % s == 0.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub o: usize, // 0..=3
    pub s: usize, // 1..=3
}

/// Prior du méta-générateur : tables q + filtres (tous comptabilisés).
#[derive(Clone, Debug)]
pub struct Prior {
    pub pool: usize,              // octets b'a'.. (modulo 256)
    pub k_max: usize,             // alphabet 1..=k_max
    pub q0_max: usize,            // file initiale 1..=q0_max
    pub m2: f64,                  // P(m = 2)
    pub w1: [f64; W_MAX + 1],     // P(|production| = 0..=3 | m = 1)
    pub w2: [f64; W_MAX + 1],     // idem | m = 2
    pub t_lo: usize,              // budget T ~ U[t_lo, t_hi]
    pub t_hi: usize,
    pub q_lo: usize,              // filtre : qmax ∈ [q_lo, q_hi]
    pub q_hi: usize,
    pub need_all_fired: bool,     // filtre de minimalité : toute règle tire dans l'horizon
    pub trunc: bool, // true : horizon coupé au dernier tick dont la file lue est ≤ q_hi, au lieu de rejeter le monde
}

impl Prior {
    pub fn demo() -> Prior {
        Prior {
            pool: 26, k_max: 4, q0_max: 3, m2: 0.3,
            w1: [0.1, 0.6, 0.2, 0.1], w2: [0.1, 0.2, 0.5, 0.2],
            t_lo: 4, t_hi: 16, q_lo: 1, q_hi: 6, need_all_fired: true,
            trunc: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub prog: Program,
    pub view: View,
    pub t: usize,
    pub trace: Vec<u8>, // règle (indice du symbole de tête) tirée à chaque pas
    pub emit: Vec<u8>,  // octet émis à chaque pas
    pub flag: Vec<bool>, // l'émission i est-elle dans la vue ?
    pub qmax: usize,
    pub bits_p: f64, // |p| sous le sampler
    pub bits_v: f64, // |V|
    pub bits_t: f64, // choix du budget T
    pub attempts: u32, // tentatives (coût du rejet)
}

// ---------- sampler ----------

pub fn sample_program(rng: &mut Rng, pr: &Prior, bits: &mut f64) -> Program {
    let k_cap = pr.k_max.min(K_MAX).min(pr.pool);
    let k = rng.range(1, k_cap);
    *bits += (k_cap as f64).log2();
    let mut pool: Vec<u8> = (0..pr.pool).map(|i| ((b'a' as usize + i) % 256) as u8).collect();
    let mut bytes = Vec::with_capacity(k);
    for _ in 0..k {
        let i = rng.below(pool.len());
        *bits += (pool.len() as f64).log2();
        bytes.push(pool.swap_remove(i));
    }
    let m = if rng.bernoulli(pr.m2) { 2 } else { 1 };
    *bits += if m == 2 { -pr.m2.log2() } else { -(1.0 - pr.m2).log2() };
    let ql_cap = pr.q0_max.min(Q_MAX);
    let ql = rng.range(1, ql_cap);
    *bits += (ql_cap as f64).log2();
    let mut q0 = Vec::with_capacity(ql);
    for _ in 0..ql {
        q0.push(rng.below(k) as u8);
        *bits += (k as f64).log2();
    }
    let wp = if m == 2 { &pr.w2 } else { &pr.w1 };
    let wtot: f64 = wp.iter().sum();
    let mut prod = Vec::with_capacity(k);
    for _ in 0..k {
        let l = rng.categorical(wp);
        *bits += -(wp[l] / wtot).log2();
        let mut w = Vec::with_capacity(l);
        for _ in 0..l {
            w.push(rng.below(k) as u8);
            *bits += (k as f64).log2();
        }
        prod.push(w);
    }
    Program { m, bytes, prod, q0 }
}

pub fn sample_view(rng: &mut Rng, bits: &mut f64) -> View {
    let o = rng.below(4);
    let s = 1 + rng.below(3);
    *bits += 2.0 + 3f64.log2();
    View { o, s }
}

// ---------- traceur ----------

pub struct Run {
    pub trace: Vec<u8>,
    pub qmax: usize,
    pub fired: u32, // masque des règles tirées
}

/// Exécute t pas. None si la file s'épuise (non productif).
pub fn run(p: &Program, t: usize) -> Option<Run> {
    let mut q: VecDeque<u8> = p.q0.iter().copied().collect();
    let mut qmax = q.len();
    let mut trace = Vec::with_capacity(t);
    let mut fired: u32 = 0;
    for _ in 0..t {
        if q.len() < p.m {
            return None;
        }
        let j = q.pop_front()?;
        for _ in 1..p.m {
            let _ = q.pop_front();
        }
        trace.push(j);
        fired |= 1u32 << (j as u32);
        for &x in &p.prod[j as usize] {
            q.push_back(x);
        }
        if q.len() > qmax {
            qmax = q.len();
        }
    }
    Some(Run { trace, qmax, fired })
}

/// Comme `run`, mais l'horizon est coupé (sans rejet) au premier tick dont la file lue dépasse `cap`.
/// Ticks gardés = ceux dont la file avant pop est ≤ cap. None si la file s'épuise avant.
pub fn run_cap(p: &Program, t: usize, cap: usize) -> Option<Run> {
    let mut q: VecDeque<u8> = p.q0.iter().copied().collect();
    let mut qmax = q.len();
    let mut trace = Vec::with_capacity(t);
    let mut fired: u32 = 0;
    for _ in 0..t {
        if q.len() > cap {
            break;
        }
        if q.len() < p.m {
            return None;
        }
        let j = q.pop_front()?;
        for _ in 1..p.m {
            let _ = q.pop_front();
        }
        trace.push(j);
        fired |= 1u32 << (j as u32);
        for &x in &p.prod[j as usize] {
            q.push_back(x);
        }
        if q.len() > qmax {
            qmax = q.len();
        }
    }
    Some(Run { trace, qmax, fired })
}

pub fn emit_of(p: &Program, trace: &[u8]) -> Vec<u8> {
    trace.iter().map(|&j| p.bytes[j as usize]).collect()
}

#[allow(clippy::too_many_arguments)]
fn finish(p: Program, view: View, t: usize, r: Run, bp: f64, bv: f64, bt: f64, att: u32) -> Sample {
    let emit = emit_of(&p, &r.trace);
    let flag: Vec<bool> = (0..t).map(|i| i >= view.o && (i - view.o) % view.s == 0).collect();
    Sample { prog: p, view, t, trace: r.trace, emit, flag, qmax: r.qmax, bits_p: bp, bits_v: bv, bits_t: bt, attempts: att }
}

/// Monde = (C, V, T) tiré du prior. Filtres : productif, minimalité (horizon complet) ; puis, selon `trunc`,
/// rejet par qmax (v1) ou troncature de l'horizon à file ≤ q_hi (v2).
pub fn sample_world(rng: &mut Rng, pr: &Prior, max_attempts: u32) -> Option<Sample> {
    for a in 1..=max_attempts {
        let (mut bp, mut bv) = (0.0f64, 0.0f64);
        let p = sample_program(rng, pr, &mut bp);
        let t = rng.range(pr.t_lo, pr.t_hi);
        let bt = ((pr.t_hi - pr.t_lo + 1) as f64).log2();
        let view = sample_view(rng, &mut bv);
        let r = match run(&p, t) {
            Some(r) => r,
            None => continue,
        };
        if pr.need_all_fired && r.fired != (1u32 << p.k()) - 1 {
            continue;
        }
        if pr.trunc {
            let rc = match run_cap(&p, t, pr.q_hi) {
                Some(rc) => rc,
                None => continue,
            };
            let te = rc.trace.len();
            if te == 0 || te <= view.o {
                continue; // aucun tick observé par la vue
            }
            return Some(finish(p, view, te, rc, bp, bv, bt, a));
        }
        if r.qmax < pr.q_lo || r.qmax > pr.q_hi {
            continue;
        }
        return Some(finish(p, view, t, r, bp, bv, bt, a));
    }
    None
}

// ---------- mutateur (do() local sur le code) ----------

enum Edit {
    Q0Sub(usize, u8),
    ProdSub(usize, usize, u8),
    ProdIns(usize, usize, u8),
    ProdDel(usize, usize),
    FlipM,
}

/// Une édition uniforme dans le voisinage à un site ; coût log2(|voisinage|).
pub fn mutate(rng: &mut Rng, p: &Program, bits: &mut f64) -> Program {
    let k = p.k();
    let mut ed: Vec<Edit> = Vec::new();
    for i in 0..p.q0.len() {
        for x in 0..k as u8 {
            if x != p.q0[i] {
                ed.push(Edit::Q0Sub(i, x));
            }
        }
    }
    for j in 0..k {
        for i in 0..p.prod[j].len() {
            for x in 0..k as u8 {
                if x != p.prod[j][i] {
                    ed.push(Edit::ProdSub(j, i, x));
                }
            }
            ed.push(Edit::ProdDel(j, i));
        }
        if p.prod[j].len() < W_MAX {
            for i in 0..=p.prod[j].len() {
                for x in 0..k as u8 {
                    ed.push(Edit::ProdIns(j, i, x));
                }
            }
        }
    }
    ed.push(Edit::FlipM);
    *bits += (ed.len() as f64).log2();
    let mut np = p.clone();
    match ed[rng.below(ed.len())] {
        Edit::Q0Sub(i, x) => np.q0[i] = x,
        Edit::ProdSub(j, i, x) => np.prod[j][i] = x,
        Edit::ProdIns(j, i, x) => np.prod[j].insert(i, x),
        Edit::ProdDel(j, i) => {
            np.prod[j].remove(i);
        }
        Edit::FlipM => np.m = 3 - np.m,
    }
    np
}

/// Enfant muté du parent, même vue, horizon ≤ celui du parent. Productif exigé ; qmax ≤ q_hi exigé (v1)
/// ou horizon coupé à file ≤ q_hi (v2). Pas de filtre de minimalité : D doit rester total sur les voisins.
pub fn mutate_sample(rng: &mut Rng, par: &Sample, pr: &Prior, max_attempts: u32) -> Option<Sample> {
    for a in 1..=max_attempts {
        let mut bp = par.bits_p;
        let np = mutate(rng, &par.prog, &mut bp);
        if pr.trunc {
            if let Some(r) = run_cap(&np, par.t, pr.q_hi) {
                let te = r.trace.len();
                if te > 0 && te > par.view.o {
                    return Some(finish(np, par.view, te, r, bp, par.bits_v, par.bits_t, a));
                }
            }
            continue;
        }
        if let Some(r) = run(&np, par.t) {
            if r.qmax > pr.q_hi {
                continue;
            }
            return Some(finish(np, par.view, par.t, r, bp, par.bits_v, par.bits_t, a));
        }
    }
    None
}

// ---------- encodeur gelé C,V → z (jamais appris) ----------

pub fn encode(p: &Program, v: &View, z: &mut [f32]) {
    assert_eq!(z.len(), Z_DIM);
    for x in z.iter_mut() {
        *x = 0.0;
    }
    z[p.m - 1] = 1.0;
    for j in 0..p.k() {
        let base = 2 + j * ROW;
        let b = p.bytes[j];
        for i in 0..8 {
            z[base + i] = if (b >> i) & 1 == 1 { 1.0 } else { -1.0 };
        }
        z[base + 8] = 1.0;
        for s in 0..W_MAX {
            let idx = if s < p.prod[j].len() { p.prod[j][s] as usize } else { NONE };
            z[base + 9 + s * NSLOT + idx] = 1.0;
        }
    }
    let qb = 2 + K_MAX * ROW;
    for s in 0..Q_MAX {
        let idx = if s < p.q0.len() { p.q0[s] as usize } else { NONE };
        z[qb + s * NSLOT + idx] = 1.0;
    }
    z[H_DIM + v.s - 1] = 1.0;
    z[H_DIM + 3 + v.o] = 1.0;
}