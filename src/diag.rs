// Anatomie des erreurs de D : OÙ (tick), QUOI (règle / octet), SUR QUELLES CLASSES, COMMENT (hésitant / confiant).
// Lecture seule : aucun entraînement.
#![allow(dead_code)]

use crate::eval::{draw_world, nontrivial, rehorizon, Part};
use crate::meta::{Prior, Sample, K_MAX, W_MAX};
use crate::tape::{decode, Net, Work, NB, NOUT};
use crate::rng::Rng;

#[derive(Clone, Copy, Default)]
struct Tab {
    n: [usize; 16],
    ok: [usize; 16],
}

impl Tab {
    fn add(&mut self, i: usize, ok: bool) {
        let i = i.min(15);
        self.n[i] += 1;
        self.ok[i] += ok as usize;
    }

    fn show(&self, title: &str, lo: usize, hi: usize) {
        print!("{:<26}", title);
        for i in lo..=hi.min(15) {
            if self.n[i] > 0 {
                print!(" | {}: n={} fid {:.1}%", i, self.n[i], 100.0 * self.ok[i] as f64 / self.n[i] as f64);
            }
        }
        println!();
    }
}

fn pct(a: usize, b: usize) -> f64 {
    100.0 * a as f64 / b.max(1) as f64
}

/// n mondes non vus du prior `pr` (triviaux compris). horizon > 0 : mêmes mondes prolongés à cet horizon
/// (qmax ≤ pr.q_hi conservé) : le risque par tick montre où l'extrapolation en temps casse.
pub fn run(net: &Net, pr: &Prior, n: usize, seed: u64, horizon: usize) {
    let mut rng = Rng::new(seed ^ 0xD1A6);
    let mut ws = Work::new();
    let tmax = horizon.max(pr.t_hi).max(1);
    let mut hz_n = vec![0usize; tmax];
    let mut hz_rule = vec![0usize; tmax];
    let mut hz_byte = vec![0usize; tmax];
    let (mut by_nt, mut by_m, mut by_k, mut by_pl, mut by_q) =
        (Tab::default(), Tab::default(), Tab::default(), Tab::default(), Tab::default());
    let (mut got, mut n_ok, mut mismatch) = (0usize, 0usize, 0usize);
    let (mut r_tot, mut r_ghost, mut r_rep, mut r_lead) = (0usize, 0usize, 0usize, 0usize);
    let (mut r_near, mut r_conf, mut b_tot, mut b_near) = (0usize, 0usize, 0usize, 0usize);
    let mut b_bits = [0usize; 4];
    let mut guard = 0usize;
    while got < n && guard < 200 * n.max(1) {
        guard += 1;
        let s0 = match draw_world(&mut rng, pr, Part::Held, 1_000_000) {
            Some(s) => s,
            None => continue,
        };
        let s: Sample = if horizon > 0 {
            match rehorizon(&s0, horizon) {
                Some(x) if x.qmax <= pr.q_hi => x,
                _ => continue,
            }
        } else {
            s0
        };
        let st = net.run_sample(&s, 1.0, &mut ws, None);
        let mut failed = false;
        for k in 0..s.t {
            let y = &ws.y[k * NOUT..(k + 1) * NOUT];
            let (b, r) = decode(y);
            let tr = s.trace[k] as usize;
            hz_n[k] += 1;
            if r != tr {
                failed = true;
                hz_rule[k] += 1;
                r_tot += 1;
                if r >= s.prog.k() {
                    r_ghost += 1;
                }
                if k > 0 && r == s.trace[k - 1] as usize {
                    r_rep += 1;
                } else if k + 1 < s.t && r == s.trace[k + 1] as usize {
                    r_lead += 1;
                }
                let margin = y[NB + r] - y[NB + tr];
                if margin < 1.0 {
                    r_near += 1;
                } else if margin > 4.0 {
                    r_conf += 1;
                }
                break;
            }
            if s.flag[k] && b != s.emit[k] {
                failed = true;
                hz_byte[k] += 1;
                b_tot += 1;
                let diff = b ^ s.emit[k];
                b_bits[(diff.count_ones() as usize - 1).min(3)] += 1;
                let mut mn = f64::INFINITY;
                for i in 0..NB {
                    if (diff >> i) & 1 == 1 {
                        mn = mn.min(y[i].abs());
                    }
                }
                if mn < 1.0 {
                    b_near += 1;
                }
                break;
            }
        }
        let ok = !failed;
        if ok {
            n_ok += 1;
        }
        if ok != (st.regen && st.trace_all) {
            mismatch += 1;
        }
        let maxpl = s.prog.prod.iter().map(|w| w.len()).max().unwrap_or(0);
        by_nt.add(nontrivial(&s) as usize, ok);
        by_m.add(s.prog.m, ok);
        by_k.add(s.prog.k(), ok);
        by_pl.add(maxpl, ok);
        by_q.add(s.qmax, ok);
        got += 1;
    }

    println!(
        "=== DIAG : D sur {} mondes non vus (T ∈ [{}, {}], qmax ≤ {}{}) ===",
        got,
        pr.t_lo,
        pr.t_hi,
        pr.q_hi,
        if horizon > 0 { format!(", horizon forcé à {}", horizon) } else { String::new() }
    );
    println!(
        "fid (tous mondes, triviaux compris) : {:.1}%   écarts avec la perte d'entraînement : {} (doit être 0)",
        pct(n_ok, got),
        mismatch
    );

    println!("\nRISQUE PAR TICK : P(première erreur au tick k | tout juste avant), en %");
    println!("{:>4}{:>10}{:>11}{:>11}", "k", "à risque", "règle", "octet");
    for k in 0..tmax {
        if hz_n[k] > 0 {
            println!("{:>4}{:>10}{:>11.2}{:>11.2}", k, hz_n[k], pct(hz_rule[k], hz_n[k]), pct(hz_byte[k], hz_n[k]));
        }
    }

    println!("\nPAR CLASSE (fid exacte)");
    by_nt.show("E constant (0) / non (1)", 0, 1);
    by_m.show("m (pops par tick)", 1, 2);
    by_k.show("k (nombre de règles)", 1, K_MAX);
    by_pl.show("|production| max", 0, W_MAX);
    by_q.show("qmax", 1, 15);

    println!("\nANATOMIE de la première erreur des mondes ratés ({} de règle, {} d'octet)", r_tot, b_tot);
    println!(
        "règle fausse : inexistante {:.0}% | répète le tick précédent {:.0}% | prend celle du tick suivant {:.0}% | marge < 1 (hésitant) {:.0}% | marge > 4 (confiant) {:.0}%",
        pct(r_ghost, r_tot),
        pct(r_rep, r_tot),
        pct(r_lead, r_tot),
        pct(r_near, r_tot),
        pct(r_conf, r_tot)
    );
    println!(
        "octet faux (règle juste) : 1 bit {:.0}% | 2 bits {:.0}% | 3 bits {:.0}% | 4+ bits {:.0}% | bit fautif proche de 0 (|logit| < 1) {:.0}%",
        pct(b_bits[0], b_tot),
        pct(b_bits[1], b_tot),
        pct(b_bits[2], b_tot),
        pct(b_bits[3], b_tot),
        pct(b_near, b_tot)
    );
    println!("lecture : marges < 1 dominantes = bruit d'optimisation ou de précision ; marges > 4 concentrées sur une classe = algorithme absent.");
}