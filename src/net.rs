// Substrat de D (simulateur interne) : (H, spec de vue) → Ê, jamais E en entrée.
//
// • 12 nœuds × 16 unités. Entre nœuds, le couplage (W et U) suit un TOURNOI : pour toute paire
//   {i,j} une seule des deux arêtes existe (les poids absents n'existent pas).
//   Les blocs diagonaux (dynamique interne d'un nœud) ne sont pas une paire.
// • Binding activationnel : p = U·s + u0 ; q = G·z + g0 ; b = p ⊙ q.
// • Tick : s' = tanh(W·s + b + c). Lecture : y = Wo·s + Vo·b + bo (8 bits d'octet + K_MAX logits de règle).
// • s0 = tanh(E·z + e0). Aucune entrée par tick : D est autonome, y_k ne dépend pas de l'horizon.
// • BPTT complet à la main, Adam + écrêtage global, gradcheck. f64 partout, zéro dépendance.
#![allow(dead_code)]

use crate::meta::{encode, sample_world, Prior, Sample, K_MAX, Z_DIM};
use crate::rng::Rng;
use std::time::Instant;

pub const ND: usize = 16; // dimension d'un nœud
pub const NN: usize = 12; // nombre de nœuds
pub const NU: usize = ND * NN; // 192 unités
const BB: usize = ND * ND;
pub const NB: usize = 8; // bits d'un octet
pub const NOUT: usize = NB + K_MAX; // 8 logits d'octet + K_MAX logits de règle
pub const SELF_LOOPS: bool = true;

// ---------- graphe : tournoi circulaire ----------

/// Arêtes (dst, src). Pour a < b : a→b si b−a ≤ NN/2, sinon b→a (une seule arête par paire).
pub fn edges() -> Vec<(usize, usize)> {
    let mut e = Vec::new();
    for a in 0..NN {
        for b in a + 1..NN {
            if b - a <= NN / 2 {
                e.push((b, a));
            } else {
                e.push((a, b));
            }
        }
    }
    if SELF_LOOPS {
        for i in 0..NN {
            e.push((i, i));
        }
    }
    e
}

fn check_graph() {
    let mut has = vec![false; NN * NN]; // has[d*NN+s] : s → d
    for (d, s) in edges() {
        if d != s {
            assert!(!has[d * NN + s], "arête dupliquée");
            has[d * NN + s] = true;
        }
    }
    for a in 0..NN {
        for b in a + 1..NN {
            assert!(has[b * NN + a] ^ has[a * NN + b], "paire ({a},{b}) : tournoi violé");
        }
    }
    for i in 0..NN {
        assert!(has[((i + 1) % NN) * NN + i], "cycle hamiltonien rompu en {i}");
    }
}

// ---------- disposition du vecteur de paramètres ----------

#[derive(Clone, Copy, Debug)]
pub struct Lay {
    pub ne: usize,
    pub w: usize,  // ne blocs ND×ND (colonnes) : W
    pub u: usize,  // ne blocs : U
    pub ub: usize, // NU : u0
    pub g: usize,  // NU×Z_DIM (colonnes) : G
    pub gb: usize, // NU : g0
    pub c: usize,  // NU
    pub e: usize,  // NU×Z_DIM (colonnes) : E
    pub e0: usize, // NU
    pub wo: usize, // NOUT×NU (lignes)
    pub vo: usize, // NOUT×NU (lignes)
    pub bo: usize, // NOUT
    pub total: usize,
}

impl Lay {
    pub fn new(ne: usize) -> Lay {
        let mut o = 0usize;
        let mut take = |n: usize| -> usize {
            let r = o;
            o += n;
            r
        };
        let w = take(ne * BB);
        let u = take(ne * BB);
        let ub = take(NU);
        let g = take(NU * Z_DIM);
        let gb = take(NU);
        let c = take(NU);
        let e = take(NU * Z_DIM);
        let e0 = take(NU);
        let wo = take(NOUT * NU);
        let vo = take(NOUT * NU);
        let bo = take(NOUT);
        Lay { ne, w, u, ub, g, gb, c, e, e0, wo, vo, bo, total: o }
    }

    pub fn groups(&self) -> Vec<(&'static str, usize, usize)> {
        let ne = self.ne;
        vec![
            ("W", self.w, ne * BB),
            ("U", self.u, ne * BB),
            ("u0", self.ub, NU),
            ("G", self.g, NU * Z_DIM),
            ("g0", self.gb, NU),
            ("c", self.c, NU),
            ("E", self.e, NU * Z_DIM),
            ("e0", self.e0, NU),
            ("Wo", self.wo, NOUT * NU),
            ("Vo", self.vo, NOUT * NU),
            ("bo", self.bo, NOUT),
        ]
    }
}

// ---------- espace de travail (un par fil) ----------

pub struct Work {
    z: Vec<f64>,
    q: Vec<f64>,
    b: Vec<f64>,
    s: Vec<f64>,  // s[k] = état avant le tick k
    p: Vec<f64>,  // p[k] = U·s[k] + u0
    dy: Vec<f64>, // dérivée de la perte par rapport aux sorties
    pub y: Vec<f64>, // logits de sortie, NOUT par tick
    ds: Vec<f64>,
    dc: Vec<f64>,
    db: Vec<f64>,
    da: Vec<f64>,
    dp: Vec<f64>,
    dq: Vec<f64>,
}

impl Work {
    pub fn new() -> Work {
        let v = |n: usize| vec![0.0f64; n];
        Work {
            z: v(Z_DIM), q: v(NU), b: v(NU), s: v(0), p: v(0), dy: v(0), y: v(0),
            ds: v(NU), dc: v(NU), db: v(NU), da: v(NU), dp: v(NU), dq: v(NU),
        }
    }

    fn reserve(&mut self, t: usize) {
        let t = t.max(1);
        if self.s.len() < t * NU {
            self.s.resize(t * NU, 0.0);
            self.p.resize(t * NU, 0.0);
        }
        if self.y.len() < t * NOUT {
            self.y.resize(t * NOUT, 0.0);
            self.dy.resize(t * NOUT, 0.0);
        }
    }
}

// ---------- noyaux ----------
// Blocs ND×ND stockés par colonnes : W[r][c] = blk[c*ND + r].

fn bs_mul_add(edges: &[(usize, usize)], blk: &[f64], x: &[f64], out: &mut [f64]) {
    for (e, &(d, s)) in edges.iter().enumerate() {
        let b = &blk[e * BB..(e + 1) * BB];
        let xs = &x[s * ND..(s + 1) * ND];
        let o = &mut out[d * ND..(d + 1) * ND];
        for c in 0..ND {
            let xc = xs[c];
            let col = &b[c * ND..(c + 1) * ND];
            for r in 0..ND {
                o[r] += col[r] * xc;
            }
        }
    }
}

fn bs_mul_t(edges: &[(usize, usize)], blk: &[f64], dd: &[f64], dx: &mut [f64]) {
    for (e, &(d, s)) in edges.iter().enumerate() {
        let b = &blk[e * BB..(e + 1) * BB];
        let dv = &dd[d * ND..(d + 1) * ND];
        let o = &mut dx[s * ND..(s + 1) * ND];
        for c in 0..ND {
            let col = &b[c * ND..(c + 1) * ND];
            let mut acc = 0.0;
            for r in 0..ND {
                acc += col[r] * dv[r];
            }
            o[c] += acc;
        }
    }
}

fn bs_outer(edges: &[(usize, usize)], gblk: &mut [f64], dd: &[f64], x: &[f64]) {
    for (e, &(d, s)) in edges.iter().enumerate() {
        let gb = &mut gblk[e * BB..(e + 1) * BB];
        let dv = &dd[d * ND..(d + 1) * ND];
        let xs = &x[s * ND..(s + 1) * ND];
        for c in 0..ND {
            let xc = xs[c];
            let col = &mut gb[c * ND..(c + 1) * ND];
            for r in 0..ND {
                col[r] += dv[r] * xc;
            }
        }
    }
}

// Matrices NU×Z_DIM stockées par colonnes ; z est creux, on saute les zéros.
fn cm_mul_add(m: &[f64], x: &[f64], out: &mut [f64]) {
    for c in 0..Z_DIM {
        let xc = x[c];
        if xc == 0.0 {
            continue;
        }
        let col = &m[c * NU..(c + 1) * NU];
        for r in 0..NU {
            out[r] += col[r] * xc;
        }
    }
}

fn cm_outer(gm: &mut [f64], d: &[f64], x: &[f64]) {
    for c in 0..Z_DIM {
        let xc = x[c];
        if xc == 0.0 {
            continue;
        }
        let col = &mut gm[c * NU..(c + 1) * NU];
        for r in 0..NU {
            col[r] += d[r] * xc;
        }
    }
}

// ---------- le réseau ----------

pub struct Net {
    pub lay: Lay,
    pub edges: Vec<(usize, usize)>,
    pub p: Vec<f64>,
}

impl Net {
    pub fn new(rng: &mut Rng) -> Net {
        let ed = edges();
        let lay = Lay::new(ed.len());
        let mut indeg = [0usize; NN];
        for &(d, _) in &ed {
            indeg[d] += 1;
        }
        let mut p = vec![0.0f64; lay.total];
        for (e, &(d, _)) in ed.iter().enumerate() {
            let sd = 1.0 / ((ND * indeg[d]) as f64).sqrt();
            for i in 0..BB {
                p[lay.w + e * BB + i] = sd * rng.normal() as f64;
                p[lay.u + e * BB + i] = sd * rng.normal() as f64;
            }
        }
        for i in 0..NU * Z_DIM {
            p[lay.g + i] = 0.15 * rng.normal() as f64;
            p[lay.e + i] = 0.15 * rng.normal() as f64;
        }
        for i in 0..NOUT * NU {
            p[lay.wo + i] = 0.03 * rng.normal() as f64;
            p[lay.vo + i] = 0.03 * rng.normal() as f64;
        }
        Net { lay, edges: ed, p }
    }

    /// Exécute t ticks à partir de z (= C,V encodés). Sorties : ws.y[k*NOUT..].
    pub fn forward(&self, z32: &[f32], t: usize, ws: &mut Work) {
        let l = &self.lay;
        let ne = l.ne;
        ws.reserve(t);
        for i in 0..Z_DIM {
            ws.z[i] = z32[i] as f64;
        }
        // q = G·z + g0 (constant sur l'horizon)
        ws.q.copy_from_slice(&self.p[l.gb..l.gb + NU]);
        cm_mul_add(&self.p[l.g..l.g + NU * Z_DIM], &ws.z, &mut ws.q);
        // s0 = tanh(E·z + e0)
        {
            let s0 = &mut ws.s[0..NU];
            s0.copy_from_slice(&self.p[l.e0..l.e0 + NU]);
            cm_mul_add(&self.p[l.e..l.e + NU * Z_DIM], &ws.z, s0);
            for x in s0.iter_mut() {
                *x = x.tanh();
            }
        }
        let wblk = &self.p[l.w..l.w + ne * BB];
        let ublk = &self.p[l.u..l.u + ne * BB];
        for k in 0..t {
            let (sa, sb) = ws.s.split_at_mut((k + 1) * NU);
            let sk = &sa[k * NU..];
            let pk = &mut ws.p[k * NU..(k + 1) * NU];
            pk.copy_from_slice(&self.p[l.ub..l.ub + NU]);
            bs_mul_add(&self.edges, ublk, sk, pk);
            for a in 0..NU {
                ws.b[a] = pk[a] * ws.q[a];
            }
            let yk = &mut ws.y[k * NOUT..(k + 1) * NOUT];
            for o in 0..NOUT {
                let wr = &self.p[l.wo + o * NU..l.wo + (o + 1) * NU];
                let vr = &self.p[l.vo + o * NU..l.vo + (o + 1) * NU];
                let mut acc = self.p[l.bo + o];
                for a in 0..NU {
                    acc += wr[a] * sk[a] + vr[a] * ws.b[a];
                }
                yk[o] = acc;
            }
            if k + 1 < t {
                let sn = &mut sb[0..NU];
                sn.copy_from_slice(&self.p[l.c..l.c + NU]);
                for a in 0..NU {
                    sn[a] += ws.b[a];
                }
                bs_mul_add(&self.edges, wblk, sk, sn);
                for x in sn.iter_mut() {
                    *x = x.tanh();
                }
            }
        }
    }

    /// BPTT complet. Lit ws.dy (rempli par la perte), accumule dans g (pas de remise à zéro).
    pub fn backward(&self, t: usize, ws: &mut Work, g: &mut [f64]) {
        let l = &self.lay;
        let ne = l.ne;
        let wblk = &self.p[l.w..l.w + ne * BB];
        let ublk = &self.p[l.u..l.u + ne * BB];
        for x in ws.ds.iter_mut() {
            *x = 0.0;
        }
        for x in ws.dq.iter_mut() {
            *x = 0.0;
        }
        for k in (0..t).rev() {
            let sk = &ws.s[k * NU..(k + 1) * NU];
            let pk = &ws.p[k * NU..(k + 1) * NU];
            for x in ws.dc.iter_mut() {
                *x = 0.0;
            }
            for x in ws.db.iter_mut() {
                *x = 0.0;
            }
            // lecture au tick k
            let dyk = &ws.dy[k * NOUT..(k + 1) * NOUT];
            for o in 0..NOUT {
                let d = dyk[o];
                if d == 0.0 {
                    continue;
                }
                g[l.bo + o] += d;
                let wr = &self.p[l.wo + o * NU..l.wo + (o + 1) * NU];
                let vr = &self.p[l.vo + o * NU..l.vo + (o + 1) * NU];
                for a in 0..NU {
                    let ba = pk[a] * ws.q[a];
                    g[l.wo + o * NU + a] += d * sk[a];
                    g[l.vo + o * NU + a] += d * ba;
                    ws.dc[a] += d * wr[a];
                    ws.db[a] += d * vr[a];
                }
            }
            // mise à jour s[k+1] = tanh(W·s[k] + b[k] + c)
            if k + 1 < t {
                let sn = &ws.s[(k + 1) * NU..(k + 2) * NU];
                for a in 0..NU {
                    let da = ws.ds[a] * (1.0 - sn[a] * sn[a]);
                    ws.da[a] = da;
                    g[l.c + a] += da;
                    ws.db[a] += da;
                }
                bs_outer(&self.edges, &mut g[l.w..l.w + ne * BB], &ws.da, sk);
                bs_mul_t(&self.edges, wblk, &ws.da, &mut ws.dc);
            }
            // binding b = p ⊙ q, p = U·s + u0
            for a in 0..NU {
                let db = ws.db[a];
                let dp = db * ws.q[a];
                ws.dp[a] = dp;
                ws.dq[a] += db * pk[a];
                g[l.ub + a] += dp;
            }
            bs_outer(&self.edges, &mut g[l.u..l.u + ne * BB], &ws.dp, sk);
            bs_mul_t(&self.edges, ublk, &ws.dp, &mut ws.dc);
            std::mem::swap(&mut ws.ds, &mut ws.dc); // ds = dL/ds[k]
        }
        // s0 = tanh(E·z + e0)
        let s0 = &ws.s[0..NU];
        for a in 0..NU {
            let da = ws.ds[a] * (1.0 - s0[a] * s0[a]);
            ws.da[a] = da;
            g[l.e0 + a] += da;
        }
        cm_outer(&mut g[l.e..l.e + NU * Z_DIM], &ws.da, &ws.z);
        // q = G·z + g0
        for a in 0..NU {
            g[l.gb + a] += ws.dq[a];
        }
        cm_outer(&mut g[l.g..l.g + NU * Z_DIM], &ws.dq, &ws.z);
    }

    /// Un monde : forward, perte, et (si g) gradient accumulé dans g.
    pub fn run_sample(&self, s: &Sample, w_trace: f64, ws: &mut Work, g: Option<&mut [f64]>) -> Stat {
        let mut z = [0f32; Z_DIM];
        encode(&s.prog, &s.view, &mut z);
        self.forward(&z, s.t, ws);
        let st = loss_raw(s, w_trace, &ws.y, &mut ws.dy);
        if let Some(g) = g {
            self.backward(s.t, ws, g);
        }
        st
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut buf = Vec::with_capacity(12 + self.p.len() * 8);
        buf.extend_from_slice(b"M0D1");
        buf.extend_from_slice(&(self.p.len() as u64).to_le_bytes());
        for x in &self.p {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(path, buf)
    }

    pub fn load(path: &str) -> std::io::Result<Net> {
        let buf = std::fs::read(path)?;
        let mut net = Net::new(&mut Rng::new(0));
        if buf.len() != 12 + net.p.len() * 8 || buf[..4] != b"M0D1"[..] {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "fichier modèle invalide"));
        }
        for (i, c) in buf[12..].chunks_exact(8).enumerate() {
            net.p[i] = f64::from_le_bytes(c.try_into().unwrap());
        }
        Ok(net)
    }
}

// ---------- perte et lecture ----------

#[derive(Clone, Copy, Debug, Default)]
pub struct Stat {
    pub t: usize,
    pub loss: f64,      // perte normalisée par T (celle qui est rétropropagée)
    pub bits_ce: f64,   // CE moyenne par bit, sur les ticks observés
    pub trace_ce: f64,  // CE moyenne de la règle, par tick
    pub n_flag: usize,  // ticks observés
    pub flag_ok: usize, // ticks observés dont l'octet est exact
    pub trace_ok: usize,
    pub regen: bool,     // régénération exacte sous la vue
    pub trace_all: bool, // trace exacte sur tout l'horizon
}

fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Lecture d'un tick : (octet, règle) depuis les NOUT logits.
pub fn decode(y: &[f64]) -> (u8, usize) {
    let mut b = 0u8;
    for i in 0..NB {
        if y[i] > 0.0 {
            b |= 1 << i;
        }
    }
    let mut r = 0usize;
    for j in 1..K_MAX {
        if y[NB + j] > y[NB + r] {
            r = j;
        }
    }
    (b, r)
}

/// Perte = (1/T)·Σ_k [ flag_k·(1/8)·Σ_i BCE(bit_i) + w_trace·CE(règle) ] ; écrit ws.dy.
pub fn loss_raw(s: &Sample, w_trace: f64, ys: &[f64], dys: &mut [f64]) -> Stat {
    let t = s.t;
    let inv_t = 1.0 / t as f64;
    let mut st = Stat { t, regen: true, trace_all: true, ..Stat::default() };
    let (mut lb, mut lt) = (0.0f64, 0.0f64);
    for k in 0..t {
        let y = &ys[k * NOUT..(k + 1) * NOUT];
        let dy = &mut dys[k * NOUT..(k + 1) * NOUT];
        for d in dy.iter_mut() {
            *d = 0.0;
        }
        if s.flag[k] {
            let byte = s.emit[k];
            let mut l = 0.0;
            for i in 0..NB {
                let tg = ((byte >> i) & 1) as f64;
                l += softplus(y[i]) - tg * y[i];
                dy[i] = (sigmoid(y[i]) - tg) * inv_t / NB as f64;
            }
            lb += l / NB as f64;
            st.n_flag += 1;
            if decode(y).0 == byte {
                st.flag_ok += 1;
            } else {
                st.regen = false;
            }
        }
        let tg = s.trace[k] as usize;
        let lo = &y[NB..];
        let mx = lo.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = lo.iter().map(|v| (v - mx).exp()).sum();
        lt += mx + sum.ln() - lo[tg];
        for j in 0..K_MAX {
            let pj = (lo[j] - mx).exp() / sum;
            dy[NB + j] = w_trace * (pj - if j == tg { 1.0 } else { 0.0 }) * inv_t;
        }
        if decode(y).1 == tg {
            st.trace_ok += 1;
        } else {
            st.trace_all = false;
        }
    }
    st.bits_ce = lb / st.n_flag.max(1) as f64;
    st.trace_ce = lt * inv_t;
    st.loss = (lb + w_trace * lt) * inv_t;
    st
}

// ---------- Adam + écrêtage global ----------

pub struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
    b1: f64,
    b2: f64,
    eps: f64,
}

impl Adam {
    pub fn new(n: usize) -> Adam {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0, b1: 0.9, b2: 0.999, eps: 1e-8 }
    }

    /// Un pas ; g = gradient moyen du lot. Renvoie la norme avant écrêtage. Aucun pas si non fini.
    pub fn step(&mut self, p: &mut [f64], g: &[f64], lr: f64, clip: f64) -> f64 {
        let gn = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !gn.is_finite() {
            return gn;
        }
        let sc = if clip > 0.0 && gn > clip { clip / gn } else { 1.0 };
        self.t += 1;
        let c1 = 1.0 - self.b1.powi(self.t);
        let c2 = 1.0 - self.b2.powi(self.t);
        for i in 0..p.len() {
            let gi = g[i] * sc;
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * gi;
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * gi * gi;
            p[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + self.eps);
        }
        gn
    }
}

// ---------- gradcheck ----------

pub fn gradcheck(seed: u64) -> bool {
    check_graph();
    println!("graphe : tournoi sur {} nœuds, une arête par paire, cycle hamiltonien : ok", NN);
    let pr = Prior::demo();
    let mut rng = Rng::new(seed);
    let mut net = Net::new(&mut rng);
    let lay = net.lay;
    assert_eq!(lay.groups().iter().map(|g| g.2).sum::<usize>(), lay.total);
    println!("D : {} paramètres, {} arêtes, {} unités", net.p.len(), net.edges.len(), NU);
    // biais non nuls : tous les chemins portent du gradient
    for &(off, len) in &[(lay.ub, NU), (lay.gb, NU), (lay.c, NU), (lay.e0, NU), (lay.bo, NOUT)] {
        for i in 0..len {
            net.p[off + i] = 0.1 * rng.normal() as f64;
        }
    }
    let samples: Vec<Sample> = (0..6).map(|_| sample_world(&mut rng, &pr, 100_000).expect("monde")).collect();
    let mut ws = Work::new();

    // causalité : y_k ne dépend pas de l'horizon
    let mut z = [0f32; Z_DIM];
    encode(&samples[0].prog, &samples[0].view, &mut z);
    net.forward(&z, 16, &mut ws);
    let long: Vec<f64> = ws.y[..8 * NOUT].to_vec();
    net.forward(&z, 8, &mut ws);
    let causal = ws.y[..8 * NOUT] == long[..];
    println!("causalité (y_k indépendant de l'horizon) : {}", if causal { "ok" } else { "ÉCHEC" });

    let total = |n: &Net, w: &mut Work| -> f64 {
        samples.iter().map(|s| n.run_sample(s, 1.0, w, None).loss).sum::<f64>()
    };
    let mut ga = vec![0.0f64; net.p.len()];
    for s in &samples {
        net.run_sample(s, 1.0, &mut ws, Some(&mut ga[..]));
    }
    let eps = 1e-5;
    let mut ok = causal;
    println!("{:<4} {:>10} {:>10} {:>10}", "grp", "|g| max", "err abs", "err rel*");
    for (name, off, len) in lay.groups() {
        let mut idx: Vec<usize> = (0..len).collect();
        idx.sort_by(|&a, &b| ga[off + b].abs().total_cmp(&ga[off + a].abs()));
        let gmax = ga[off + idx[0]].abs();
        let mut probes: Vec<usize> = idx[..8].to_vec(); // les 8 plus grands
        for _ in 0..8 {
            probes.push(rng.below(len)); // + 8 au hasard
        }
        let (mut eabs, mut erel) = (0.0f64, 0.0f64);
        for (n, &i) in probes.iter().enumerate() {
            let j = off + i;
            let x0 = net.p[j];
            net.p[j] = x0 + eps;
            let lp = total(&net, &mut ws);
            net.p[j] = x0 - eps;
            let lm = total(&net, &mut ws);
            net.p[j] = x0;
            let gn = (lp - lm) / (2.0 * eps);
            let d = (ga[j] - gn).abs();
            eabs = eabs.max(d);
            if n < 8 {
                erel = erel.max(d / ga[j].abs().max(gn.abs()));
            }
        }
        let good = gmax > 0.0 && erel < 1e-4 && eabs < 1e-6;
        ok &= good;
        println!("{:<4} {:>10.3e} {:>10.2e} {:>10.2e}  {}", name, gmax, eabs, erel, if good { "ok" } else { "ÉCHEC" });
    }
    println!("(* erreur relative sur les 8 plus grands gradients du groupe ; seuils : rel < 1e-4, abs < 1e-6)");
    println!("{}", if ok { "GRADCHECK OK" } else { "GRADCHECK ÉCHEC" });
    ok
}

// ---------- surapprentissage d'un lot fixe : test de bout en bout du pipeline ----------

pub fn overfit(seed: u64) {
    let pr = Prior::demo();
    let mut rng = Rng::new(seed);
    let mut net = Net::new(&mut rng);
    println!(
        "D : {} paramètres, {} arêtes ({} paires + {} diagonales), {} unités",
        net.p.len(), net.edges.len(), NN * (NN - 1) / 2, net.edges.len() - NN * (NN - 1) / 2, NU
    );
    let set: Vec<Sample> = (0..16).map(|_| sample_world(&mut rng, &pr, 100_000).expect("monde")).collect();
    let n = set.len();
    let mut ws = Work::new();
    let mut g = vec![0.0f64; net.p.len()];
    let mut adam = Adam::new(net.p.len());
    let steps: usize = 600;
    let t0 = Instant::now();
    for it in 0..=steps {
        for x in g.iter_mut() {
            *x = 0.0;
        }
        let mut loss = 0.0f64;
        let (mut fok, mut nf, mut tok, mut nt, mut rg, mut ta) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for s in &set {
            let st = net.run_sample(s, 1.0, &mut ws, Some(&mut g[..]));
            loss += st.loss;
            fok += st.flag_ok;
            nf += st.n_flag;
            tok += st.trace_ok;
            nt += st.t;
            rg += st.regen as usize;
            ta += st.trace_all as usize;
        }
        let inv = 1.0 / n as f64;
        for x in g.iter_mut() {
            *x *= inv;
        }
        let gn = adam.step(&mut net.p, &g, 2e-3, 1.0);
        if it % 50 == 0 {
            println!(
                "{:>4}  L={:.4}  octets {}/{}  règles {}/{}  régén. exacte {}/{}  trace exacte {}/{}  |g|={:.3}",
                it, loss * inv, fok, nf, tok, nt, rg, n, ta, n, gn
            );
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("{:.0} échantillons/s (forward+backward, 1 fil)", ((steps + 1) * n) as f64 / dt);
}