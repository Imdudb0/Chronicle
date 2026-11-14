// Substrat de D, version « bande » (M0 bis) : mémoire à poids partagés entre positions, taille = paramètre d'exécution.
//
// • n = 3T + 6 cellules de DC unités, mêmes poids partout. Le contenu ne bouge jamais ; la tête et la queue de la file
//   sont des marqueurs latents qui n'avancent que vers la droite.
// • Couplage non réciproque : la cellule i reçoit de i, i−1, i−2, i−3 (jamais de i+1…). Les cellules alimentent le bus
//   de lecture R (attention), R alimente le bus d'écriture W (liaison avec z), W alimente les cellules.
//   Aucune paire de nœuds n'est couplée dans les deux sens ; la diagonale (i → i) est la dynamique interne.
// • Tick k : R = Σ_i softmax_i(ua·s_i)·(Uv s_i + uv) ; W = (Vw R + vb) ⊙ (Gw z + gb) ; G_o = Γ_o W + γ_o ;
//   s_i' = tanh(bc + Σ_o G_o ⊙ (A_o s_{i−o})) ; y = Wo R + Vo W + bo.
// • Chargement : s_i(0) = tanh(L1 x_i + L2 x_{i−1} + l0), x_i = i-ème symbole de q0 (lu dans z) ou « vide ».
// • L'attention se limite aux cellules i ≤ 3k (cône de lumière des marqueurs) : y_k ne dépend ni de T ni de n.
// • BPTT à la main, f64, zéro dépendance, même interface que net.rs.
#![allow(dead_code)]

use crate::meta::{encode, sample_world, Prior, Sample, K_MAX, NONE, NSLOT, Q_MAX, ROW, Z_DIM};
use crate::net::loss_raw;
pub use crate::net::{decode, Adam, Stat, NB, NOUT};
use crate::rng::Rng;

pub const DC: usize = 16; // unités par cellule
pub const DR: usize = 16; // bus de lecture
pub const DW: usize = 192; // bus d'écriture (≥ 6 lignes × 29 champs de table = 174)
pub const NO: usize = 4; // la cellule i reçoit de i−o, o = 0..3
const QB: usize = 2 + K_MAX * ROW; // début de q0 dans z

/// Budget d'exécution : nombre de cellules. Un marqueur avance d'au plus 3 cellules par tick.
pub fn cells(t: usize) -> usize {
    3 * t + 6
}

#[derive(Clone, Copy, Debug)]
pub struct Lay {
    pub l1: usize,
    pub l2: usize,
    pub l0: usize,
    pub a: usize,
    pub gam: usize,
    pub gg: usize,
    pub bc: usize,
    pub ua: usize,
    pub uv: usize,
    pub uvb: usize,
    pub vw: usize,
    pub vb: usize,
    pub gw: usize,
    pub gb: usize,
    pub wo: usize,
    pub vo: usize,
    pub bo: usize,
    pub total: usize,
}

impl Lay {
    pub fn new() -> Lay {
        let mut o = 0usize;
        let mut take = |n: usize| -> usize {
            let r = o;
            o += n;
            r
        };
        let l1 = take(DC * NSLOT);
        let l2 = take(DC * NSLOT);
        let l0 = take(DC);
        let a = take(NO * DC * DC);
        let gam = take(NO * DC);
        let gg = take(NO * DC * DW);
        let bc = take(DC);
        let ua = take(DC);
        let uv = take(DR * DC);
        let uvb = take(DR);
        let vw = take(DW * DR);
        let vb = take(DW);
        let gw = take(DW * Z_DIM);
        let gb = take(DW);
        let wo = take(NOUT * DR);
        let vo = take(NOUT * DW);
        let bo = take(NOUT);
        Lay { l1, l2, l0, a, gam, gg, bc, ua, uv, uvb, vw, vb, gw, gb, wo, vo, bo, total: o }
    }

    pub fn groups(&self) -> Vec<(&'static str, usize, usize)> {
        vec![
            ("L1", self.l1, DC * NSLOT),
            ("L2", self.l2, DC * NSLOT),
            ("l0", self.l0, DC),
            ("A", self.a, NO * DC * DC),
            ("gam", self.gam, NO * DC),
            ("Gam", self.gg, NO * DC * DW),
            ("bc", self.bc, DC),
            ("ua", self.ua, DC),
            ("Uv", self.uv, DR * DC),
            ("uvb", self.uvb, DR),
            ("Vw", self.vw, DW * DR),
            ("vb", self.vb, DW),
            ("Gw", self.gw, DW * Z_DIM),
            ("gb", self.gb, DW),
            ("Wo", self.wo, NOUT * DR),
            ("Vo", self.vo, NOUT * DW),
            ("bo", self.bo, NOUT),
        ]
    }
}

// ---------- espace de travail (un par fil) ----------

pub struct Work {
    n: usize,
    z: Vec<f64>,
    qz: Vec<f64>,
    s: Vec<f64>,   // s[k][i][d]
    att: Vec<f64>, // logits puis poids d'attention, [k][i] (pas n)
    v: Vec<f64>,   // valeurs, [k][i][j]
    r: Vec<f64>,   // R[k]
    pw: Vec<f64>,  // Vw R + vb
    w: Vec<f64>,   // W[k]
    gt: Vec<f64>,  // portes G_o[k]
    pub y: Vec<f64>,
    dy: Vec<f64>,
    ds: Vec<f64>,
    dsn: Vec<f64>,
    dl: Vec<f64>,
}

fn grow(v: &mut Vec<f64>, len: usize) {
    if v.len() < len {
        v.resize(len, 0.0);
    }
}

impl Work {
    pub fn new() -> Work {
        Work {
            n: 0,
            z: vec![0.0; Z_DIM],
            qz: vec![0.0; DW],
            s: Vec::new(),
            att: Vec::new(),
            v: Vec::new(),
            r: Vec::new(),
            pw: Vec::new(),
            w: Vec::new(),
            gt: Vec::new(),
            y: Vec::new(),
            dy: Vec::new(),
            ds: Vec::new(),
            dsn: Vec::new(),
            dl: Vec::new(),
        }
    }

    fn reserve(&mut self, t: usize, n: usize) {
        let t = t.max(1);
        grow(&mut self.s, t * n * DC);
        grow(&mut self.att, t * n);
        grow(&mut self.v, t * n * DR);
        grow(&mut self.r, t * DR);
        grow(&mut self.pw, t * DW);
        grow(&mut self.w, t * DW);
        grow(&mut self.gt, t * NO * DC);
        grow(&mut self.y, t * NOUT);
        grow(&mut self.dy, t * NOUT);
        grow(&mut self.ds, n * DC);
        grow(&mut self.dsn, n * DC);
        grow(&mut self.dl, n);
    }
}

/// i-ème symbole de la file initiale (one-hot sur NSLOT, « vide » au-delà de Q_MAX).
fn xs(z: &[f64], i: usize) -> [f64; NSLOT] {
    let mut x = [0.0f64; NSLOT];
    if i < Q_MAX {
        x.copy_from_slice(&z[QB + i * NSLOT..QB + (i + 1) * NSLOT]);
    } else {
        x[NONE] = 1.0;
    }
    x
}

fn fill(rng: &mut Rng, v: &mut [f64], sd: f64) {
    for x in v.iter_mut() {
        *x = sd * rng.normal() as f64;
    }
}

// ---------- le réseau ----------

pub struct Net {
    pub lay: Lay,
    pub p: Vec<f64>,
}

impl Net {
    pub fn new(rng: &mut Rng) -> Net {
        let lay = Lay::new();
        let l = lay;
        let mut p = vec![0.0f64; lay.total];
        fill(rng, &mut p[l.l1..l.l1 + DC * NSLOT], 0.5);
        fill(rng, &mut p[l.l2..l.l2 + DC * NSLOT], 0.5);
        fill(rng, &mut p[l.a..l.a + NO * DC * DC], 0.15);
        for d in 0..DC {
            p[l.a + d * DC + d] += 1.0; // A_0 ≈ I : le contenu persiste (biais d'architecture déclaré)
            p[l.gam + d] = 2.0; // porte de persistance : tanh(2x) = x en ±0,96
        }
        for i in DC..NO * DC {
            p[l.gam + i] = 0.3;
        }
        fill(rng, &mut p[l.gg..l.gg + NO * DC * DW], 0.05);
        fill(rng, &mut p[l.ua..l.ua + DC], 0.5);
        fill(rng, &mut p[l.uv..l.uv + DR * DC], 0.25);
        fill(rng, &mut p[l.vw..l.vw + DW * DR], 0.25);
        for i in 0..DW {
            p[l.vb + i] = 0.5;
        }
        fill(rng, &mut p[l.gw..l.gw + DW * Z_DIM], 0.15);
        fill(rng, &mut p[l.wo..l.wo + NOUT * DR], 0.03);
        fill(rng, &mut p[l.vo..l.vo + NOUT * DW], 0.03);
        Net { lay, p }
    }

    /// t ticks à partir de z (= C,V encodés). Sorties : ws.y[k*NOUT..].
    pub fn forward(&self, z32: &[f32], t: usize, ws: &mut Work) {
        let (l, p) = (&self.lay, &self.p);
        let n = cells(t);
        ws.reserve(t, n);
        ws.n = n;
        for i in 0..Z_DIM {
            ws.z[i] = z32[i] as f64;
        }
        for u in 0..DW {
            let row = &p[l.gw + u * Z_DIM..l.gw + (u + 1) * Z_DIM];
            let mut a = p[l.gb + u];
            for c in 0..Z_DIM {
                a += row[c] * ws.z[c];
            }
            ws.qz[u] = a;
        }
        for i in 0..n {
            let xi = xs(&ws.z, i);
            let xp = if i > 0 { xs(&ws.z, i - 1) } else { [0.0; NSLOT] };
            for d in 0..DC {
                let mut a = p[l.l0 + d];
                for c in 0..NSLOT {
                    a += p[l.l1 + d * NSLOT + c] * xi[c] + p[l.l2 + d * NSLOT + c] * xp[c];
                }
                ws.s[i * DC + d] = a.tanh();
            }
        }
        for k in 0..t {
            let na = n.min(3 * k + 1);
            let (sa, sb) = ws.s.split_at_mut((k + 1) * n * DC);
            let sk = &sa[k * n * DC..];
            // lecture : attention sur les cellules atteignables par un marqueur
            let mut mx = f64::NEG_INFINITY;
            for i in 0..na {
                let mut lg = 0.0;
                for d in 0..DC {
                    lg += p[l.ua + d] * sk[i * DC + d];
                }
                ws.att[k * n + i] = lg;
                if lg > mx {
                    mx = lg;
                }
            }
            let mut sum = 0.0;
            for i in 0..na {
                let e = (ws.att[k * n + i] - mx).exp();
                ws.att[k * n + i] = e;
                sum += e;
            }
            let rk = &mut ws.r[k * DR..(k + 1) * DR];
            for x in rk.iter_mut() {
                *x = 0.0;
            }
            for i in 0..na {
                let a = ws.att[k * n + i] / sum;
                ws.att[k * n + i] = a;
                for j in 0..DR {
                    let mut vj = p[l.uvb + j];
                    for d in 0..DC {
                        vj += p[l.uv + j * DC + d] * sk[i * DC + d];
                    }
                    ws.v[(k * n + i) * DR + j] = vj;
                    rk[j] += a * vj;
                }
            }
            // écriture : W = (Vw R + vb) ⊙ qz
            for u in 0..DW {
                let mut a = p[l.vb + u];
                for j in 0..DR {
                    a += p[l.vw + u * DR + j] * rk[j];
                }
                ws.pw[k * DW + u] = a;
                ws.w[k * DW + u] = a * ws.qz[u];
            }
            // portes G_o = Γ_o W + γ_o
            for o in 0..NO {
                for d in 0..DC {
                    let row = &p[l.gg + (o * DC + d) * DW..l.gg + (o * DC + d + 1) * DW];
                    let mut g = p[l.gam + o * DC + d];
                    for u in 0..DW {
                        g += row[u] * ws.w[k * DW + u];
                    }
                    ws.gt[(k * NO + o) * DC + d] = g;
                }
            }
            // sortie
            for o in 0..NOUT {
                let mut a = p[l.bo + o];
                for j in 0..DR {
                    a += p[l.wo + o * DR + j] * rk[j];
                }
                for u in 0..DW {
                    a += p[l.vo + o * DW + u] * ws.w[k * DW + u];
                }
                ws.y[k * NOUT + o] = a;
            }
            // mise à jour des cellules (flux de gauche à droite seulement)
            if k + 1 < t {
                let sn = &mut sb[0..n * DC];
                for i in 0..n {
                    for d in 0..DC {
                        let mut a = p[l.bc + d];
                        for o in 0..NO.min(i + 1) {
                            let src = &sk[(i - o) * DC..(i - o + 1) * DC];
                            let row = &p[l.a + (o * DC + d) * DC..l.a + (o * DC + d + 1) * DC];
                            let mut u = 0.0;
                            for c in 0..DC {
                                u += row[c] * src[c];
                            }
                            a += ws.gt[(k * NO + o) * DC + d] * u;
                        }
                        sn[i * DC + d] = a.tanh();
                    }
                }
            }
        }
    }

    /// BPTT complet. Lit ws.dy, accumule dans g (pas de remise à zéro).
    pub fn backward(&self, t: usize, ws: &mut Work, g: &mut [f64]) {
        let (l, p) = (&self.lay, &self.p);
        let n = ws.n;
        let mut dqz = [0.0f64; DW];
        let mut dw = [0.0f64; DW];
        let mut dr = [0.0f64; DR];
        let mut dg = [0.0f64; NO * DC];
        for x in ws.ds[..n * DC].iter_mut() {
            *x = 0.0;
        }
        for k in (0..t).rev() {
            let na = n.min(3 * k + 1);
            for x in dg.iter_mut() {
                *x = 0.0;
            }
            for x in dw.iter_mut() {
                *x = 0.0;
            }
            for x in dr.iter_mut() {
                *x = 0.0;
            }
            // 1. mise à jour s[k+1] = tanh(bc + Σ_o G_o ⊙ (A_o s[k]_{i−o}))
            if k + 1 < t {
                let sk = &ws.s[k * n * DC..(k + 1) * n * DC];
                let sn = &ws.s[(k + 1) * n * DC..(k + 2) * n * DC];
                for i in 0..n {
                    for d in 0..DC {
                        let sv = sn[i * DC + d];
                        let da = ws.dsn[i * DC + d] * (1.0 - sv * sv);
                        if da == 0.0 {
                            continue;
                        }
                        g[l.bc + d] += da;
                        for o in 0..NO.min(i + 1) {
                            let src = &sk[(i - o) * DC..(i - o + 1) * DC];
                            let ro = l.a + (o * DC + d) * DC;
                            let mut u = 0.0;
                            for c in 0..DC {
                                u += p[ro + c] * src[c];
                            }
                            dg[o * DC + d] += da * u;
                            let du = da * ws.gt[(k * NO + o) * DC + d];
                            for c in 0..DC {
                                g[ro + c] += du * src[c];
                                ws.ds[(i - o) * DC + c] += p[ro + c] * du;
                            }
                        }
                    }
                }
            }
            // 2. portes G_o = Γ_o W + γ_o
            for o in 0..NO {
                for d in 0..DC {
                    let v = dg[o * DC + d];
                    if v == 0.0 {
                        continue;
                    }
                    g[l.gam + o * DC + d] += v;
                    let ro = l.gg + (o * DC + d) * DW;
                    for u in 0..DW {
                        g[ro + u] += v * ws.w[k * DW + u];
                        dw[u] += p[ro + u] * v;
                    }
                }
            }
            // 3. sortie y = Wo R + Vo W + bo
            for o in 0..NOUT {
                let d = ws.dy[k * NOUT + o];
                if d == 0.0 {
                    continue;
                }
                g[l.bo + o] += d;
                for j in 0..DR {
                    g[l.wo + o * DR + j] += d * ws.r[k * DR + j];
                    dr[j] += p[l.wo + o * DR + j] * d;
                }
                for u in 0..DW {
                    g[l.vo + o * DW + u] += d * ws.w[k * DW + u];
                    dw[u] += p[l.vo + o * DW + u] * d;
                }
            }
            // 4. W = (Vw R + vb) ⊙ qz
            for u in 0..DW {
                let dwu = dw[u];
                if dwu == 0.0 {
                    continue;
                }
                let dpw = dwu * ws.qz[u];
                dqz[u] += dwu * ws.pw[k * DW + u];
                g[l.vb + u] += dpw;
                for j in 0..DR {
                    g[l.vw + u * DR + j] += dpw * ws.r[k * DR + j];
                    dr[j] += p[l.vw + u * DR + j] * dpw;
                }
            }
            // 5. attention : R = Σ_i a_i v_i, a = softmax(ua·s_i)
            let mut ssum = 0.0;
            for i in 0..na {
                let mut da = 0.0;
                for j in 0..DR {
                    da += dr[j] * ws.v[(k * n + i) * DR + j];
                }
                ws.dl[i] = da;
                ssum += ws.att[k * n + i] * da;
            }
            for i in 0..na {
                let a = ws.att[k * n + i];
                let dl = a * (ws.dl[i] - ssum);
                let si = &ws.s[(k * n + i) * DC..(k * n + i + 1) * DC];
                for j in 0..DR {
                    let dv = a * dr[j];
                    g[l.uvb + j] += dv;
                    for d in 0..DC {
                        g[l.uv + j * DC + d] += dv * si[d];
                        ws.ds[i * DC + d] += p[l.uv + j * DC + d] * dv;
                    }
                }
                for d in 0..DC {
                    g[l.ua + d] += dl * si[d];
                    ws.ds[i * DC + d] += dl * p[l.ua + d];
                }
            }
            // dL/ds[k] devient le « futur » du tick k−1
            std::mem::swap(&mut ws.ds, &mut ws.dsn);
            for x in ws.ds[..n * DC].iter_mut() {
                *x = 0.0;
            }
        }
        // chargement s_i(0) = tanh(L1 x_i + L2 x_{i−1} + l0)
        for i in 0..n {
            let xi = xs(&ws.z, i);
            let xp = if i > 0 { xs(&ws.z, i - 1) } else { [0.0; NSLOT] };
            for d in 0..DC {
                let s0 = ws.s[i * DC + d];
                let da = ws.dsn[i * DC + d] * (1.0 - s0 * s0);
                g[l.l0 + d] += da;
                for c in 0..NSLOT {
                    g[l.l1 + d * NSLOT + c] += da * xi[c];
                    g[l.l2 + d * NSLOT + c] += da * xp[c];
                }
            }
        }
        // qz = Gw z + gb
        for u in 0..DW {
            g[l.gb + u] += dqz[u];
            if dqz[u] == 0.0 {
                continue;
            }
            for c in 0..Z_DIM {
                g[l.gw + u * Z_DIM + c] += dqz[u] * ws.z[c];
            }
        }
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
        buf.extend_from_slice(b"M0T1");
        buf.extend_from_slice(&(self.p.len() as u64).to_le_bytes());
        for x in &self.p {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(path, buf)
    }

    pub fn load(path: &str) -> std::io::Result<Net> {
        let buf = std::fs::read(path)?;
        let mut net = Net::new(&mut Rng::new(0));
        if buf.len() != 12 + net.p.len() * 8 || buf[..4] != b"M0T1"[..] {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "fichier modèle invalide (substrat bande attendu)",
            ));
        }
        for (i, c) in buf[12..].chunks_exact(8).enumerate() {
            net.p[i] = f64::from_le_bytes(c.try_into().unwrap());
        }
        Ok(net)
    }
}

// ---------- gradcheck ----------

pub fn gradcheck(seed: u64) -> bool {
    let pr = Prior::demo();
    let mut rng = Rng::new(seed);
    let mut net = Net::new(&mut rng);
    let lay = net.lay;
    assert_eq!(lay.groups().iter().map(|g| g.2).sum::<usize>(), lay.total);
    println!(
        "D (bande) : {} paramètres ; cellule {} unités, bus de lecture {}, bus d'écriture {}",
        net.p.len(),
        DC,
        DR,
        DW
    );
    for &(off, len) in &[(lay.l0, DC), (lay.bc, DC), (lay.uvb, DR), (lay.vb, DW), (lay.gb, DW), (lay.bo, NOUT)] {
        for i in 0..len {
            net.p[off + i] += 0.1 * rng.normal() as f64;
        }
    }
    let samples: Vec<Sample> = (0..6).map(|_| sample_world(&mut rng, &pr, 100_000).expect("monde")).collect();
    let mut ws = Work::new();

    // causalité : y_k ne dépend ni de T ni de n
    let mut z = [0f32; Z_DIM];
    encode(&samples[0].prog, &samples[0].view, &mut z);
    net.forward(&z, 16, &mut ws);
    let long: Vec<f64> = ws.y[..8 * NOUT].to_vec();
    net.forward(&z, 8, &mut ws);
    let causal = ws.y[..8 * NOUT] == long[..];
    println!("causalité (y_k indépendant de T et de n) : {}", if causal { "ok" } else { "ÉCHEC" });

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
        let nt = 8.min(len);
        let mut probes: Vec<usize> = idx[..nt].to_vec();
        for _ in 0..8 {
            probes.push(rng.below(len));
        }
        let (mut eabs, mut erel) = (0.0f64, 0.0f64);
        for (q, &i) in probes.iter().enumerate() {
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
            if q < nt {
                erel = erel.max(d / ga[j].abs().max(gn.abs()).max(1e-300));
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