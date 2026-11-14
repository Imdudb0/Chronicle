// Taille de file contre erreurs de D (lecture seule, aucun entraînement).
//
// Trois explications de l'effondrement de l'axe profondeur :
//   • TAILLE UTILE : D ne porte pas plus de symboles de file qu'à l'entraînement ;
//   • QUEUE NON LUE : D se dégrade dès que la file dépasse q, même si le surplus ne sera jamais lu ;
//   • CLASSE : D ne reconnaît pas les programmes à croissance (exclus de l'entraînement par qmax ≤ q),
//     même quand leur file est encore courte.
// Pour chaque tick d'un monde non vu : file = longueur avant le pop ; utile = min(file, m·(T−k)),
// les symboles encore lus avant la fin de la vue. Référence : le risque par tick des mondes « peu profonds »
// (file ≤ q à tous les ticks), indice de tick par indice de tick, car le risque dépend aussi du tick.
// SMR = erreurs observées / erreurs attendues à ce risque de référence.
#![allow(dead_code)]

use crate::eval::{draw_world, nontrivial, Part};
use crate::meta::{Prior, Program};
use crate::net::{decode, Net, Work, NOUT};
use crate::tape::{decode, Net, Work, NOUT};
use crate::rng::Rng;
use std::collections::VecDeque;

const NG: usize = 4;

/// Longueur de file AVANT le pop de chaque tick.
pub fn queue_lens(p: &Program, t: usize) -> Vec<usize> {
    let mut q: VecDeque<u8> = p.q0.iter().copied().collect();
    let mut out = Vec::with_capacity(t);
    for _ in 0..t {
        out.push(q.len());
        let j = match q.pop_front() {
            Some(j) => j as usize,
            None => break,
        };
        for _ in 1..p.m {
            let _ = q.pop_front();
        }
        for &x in &p.prod[j] {
            q.push_back(x);
        }
    }
    out
}

fn pc(a: usize, b: usize) -> f64 {
    100.0 * a as f64 / b.max(1) as f64
}

pub fn run(net: &Net, pr: &Prior, q: usize, n: usize, seed: u64) {
    let mut rng = Rng::new(seed ^ 0x517E);
    let mut ws = Work::new();
    let tm = pr.t_hi.max(1);
    // mondes peu profonds (référence) : à risque et erreurs par indice de tick
    let (mut risk_s, mut err_s) = (vec![0usize; tm], vec![0usize; tm]);
    let (mut rule_s, mut byte_s) = (0usize, 0usize);
    // mondes profonds : à risque par [groupe][tick], erreurs par groupe
    let mut risk_d = vec![vec![0usize; tm]; NG];
    let (mut rule_d, mut byte_d) = ([0usize; NG], [0usize; NG]);
    let (mut nw, mut ok) = ([0usize; 2], [0usize; 2]);
    let (mut before, mut after) = (0usize, 0usize);
    let (mut got, mut guard) = (0usize, 0usize);
    while got < n && guard < 400 * n.max(1) {
        guard += 1;
        let s = match draw_world(&mut rng, pr, Part::Held, 1_000_000) {
            Some(s) if nontrivial(&s) => s,
            _ => continue,
        };
        net.run_sample(&s, 1.0, &mut ws, None);
        let ql = queue_lens(&s.prog, s.t);
        let first_deep = ql.iter().position(|&x| x > q);
        let deep = first_deep.is_some();
        let mut err: Option<usize> = None;
        for k in 0..s.t {
            let (b, r) = decode(&ws.y[k * NOUT..(k + 1) * NOUT]);
            let u = ql[k].min(s.prog.m * (s.t - k));
            // 0 : file ≤ q ; 1 : file > q mais utile ≤ q ; 2 : utile ∈ ]q, 1,5q] ; 3 : utile > 1,5q
            let g = if ql[k] <= q {
                0
            } else if u <= q {
                1
            } else if u <= q + q / 2 {
                2
            } else {
                3
            };
            if deep {
                risk_d[g][k] += 1;
            } else {
                risk_s[k] += 1;
            }
            let rule_bad = r != s.trace[k] as usize;
            let byte_bad = !rule_bad && s.flag[k] && b != s.emit[k];
            if rule_bad || byte_bad {
                if deep {
                    if rule_bad {
                        rule_d[g] += 1;
                    } else {
                        byte_d[g] += 1;
                    }
                } else {
                    err_s[k] += 1;
                    if rule_bad {
                        rule_s += 1;
                    } else {
                        byte_s += 1;
                    }
                }
                err = Some(k);
                break;
            }
        }
        nw[deep as usize] += 1;
        match (err, first_deep) {
            (None, _) => ok[deep as usize] += 1,
            (Some(e), Some(f)) => {
                if e < f {
                    before += 1;
                } else {
                    after += 1;
                }
            }
            _ => {}
        }
        got += 1;
    }

    // risque de référence par indice de tick (repli sur le tick précédent sous 30 mondes à risque)
    let mut hz = vec![0.0f64; tm];
    for k in 0..tm {
        hz[k] = if risk_s[k] >= 30 {
            err_s[k] as f64 / risk_s[k] as f64
        } else if k > 0 {
            hz[k - 1]
        } else {
            0.0
        };
    }

    println!(
        "=== TAILLE DE FILE : D entraîné pour une file ≤ {} ; {} mondes non vus non triviaux, T ∈ [{}, {}], file jusqu'à {} ===",
        q, got, pr.t_lo, pr.t_hi, pr.q_hi
    );
    println!(
        "peu profonds (file ≤ {} à tous les ticks) : {} mondes, fid {:.1}%  |  profonds : {} mondes, fid {:.1}%",
        q,
        nw[0],
        pc(ok[0], nw[0]),
        nw[1],
        pc(ok[1], nw[1])
    );
    println!("\n{:<26}{:>10}{:>8}{:>8}{:>11}{:>8}", "ticks", "à risque", "règle", "octet", "attendu*", "SMR");
    println!(
        "{:<26}{:>10}{:>8}{:>8}{:>11}{:>8}",
        "référence (peu profonds)",
        risk_s.iter().sum::<usize>(),
        rule_s,
        byte_s,
        "—",
        "1.00"
    );
    let names = [
        format!("profonds, file ≤ {}", q),
        format!("file > {}, utile ≤ {}", q, q),
        format!("utile ∈ ]{}, {}]", q, q + q / 2),
        format!("utile > {}", q + q / 2),
    ];
    let mut smr = [0.0f64; NG];
    for g in 0..NG {
        let risk: usize = risk_d[g].iter().sum();
        let exp: f64 = (0..tm).map(|k| risk_d[g][k] as f64 * hz[k]).sum();
        let obs = (rule_d[g] + byte_d[g]) as f64;
        let cell = if exp >= 5.0 {
            smr[g] = obs / exp;
            format!("{:.2}", smr[g])
        } else {
            "n/a".to_string()
        };
        println!("{:<26}{:>10}{:>8}{:>8}{:>11.1}{:>8}", names[g], risk, rule_d[g], byte_d[g], exp, cell);
    }
    println!("* attendu = erreurs qu'auraient ces mêmes ticks avec le risque des mondes peu profonds au même indice de tick.");
    println!("  SMR = observé / attendu (≥ 3 : excès net ; < 1,5 : nul ; n/a : attendu < 5).");
    println!(
        "mondes profonds ratés : {} avant le premier tick où la file dépasse {}, {} à partir de ce tick",
        before, q, after
    );
    let verdict = if nw[1] < 100 {
        "ÉCHANTILLON INSUFFISANT (moins de 100 mondes profonds) : augmenter --n ou --qhi"
    } else if smr[0] >= 3.0 {
        "CLASSE : D rate déjà les mondes profonds quand leur file est courte ; l'axe profondeur mesurait la classe de programme, pas la taille"
    } else if smr[2] >= 3.0 || smr[3] >= 3.0 {
        "TAILLE UTILE : l'excès apparaît quand D doit porter plus de symboles qu'à l'entraînement"
    } else if smr[1] >= 3.0 {
        "QUEUE NON LUE : D se dégrade dès que la file dépasse q, même si le surplus ne sera jamais lu"
    } else {
        "AUCUN EXCÈS NET : ni classe, ni taille utile, ni queue non lue n'expliquent l'effondrement ; regarder l'indice de tick et le type de production"
    };
    println!(
        "règle (fixée avant mesure) : CLASSE si SMR[file ≤ q] ≥ 3 ; sinon TAILLE UTILE si un des deux SMR « utile > q » ≥ 3 ; sinon QUEUE NON LUE si SMR[file > q, utile ≤ q] ≥ 3 ; sinon AUCUN EXCÈS NET."
    );
    println!("verdict : {}", verdict);
}