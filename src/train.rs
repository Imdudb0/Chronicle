// Entraînement de D (v3) : minibatchs parallèles (threads std), Adam, curriculum piloté par la fidélité,
// ou plage fixe (--thi / --qhi) pour le contrôle de plage d'entraînement.
//
// • Programmes d'entraînement : jamais le sous-ensemble « non vu » (hash(C) mod 8 = 0), voisins mutés compris.
// • lr en cycles cosinus dans chaque étage : crête → lr_min ; crête × peak_decay à chaque cycle de plus.
// • Porte jugée en fin de cycle (lr minimal) sur n_gate mondes non vus non triviaux : étage intermédiaire,
//   fid ≥ up → on monte ; étage final, fid ≥ gate → on s'arrête ; max_cycles cycles sans la franchir → arrêt.
// • --kmax : taille d'alphabet maximale des causes (bouton du prior, au grand livre).
// • <out>.ref garde la plage d'entraînement « t_hi q_hi k_max » : eval, diag et cmp la lisent par défaut.
// • Un Rng, un Work et un tampon de gradient par fil ; déterministe pour (seed, nombre de fils).
#![allow(dead_code)]

use crate::eval::{draw_world, in_part, nontrivial, validate, Agg, Part};
use crate::meta::{mutate_sample, Prior, Sample};
use crate::net::{Adam, Net, Work};
use crate::rng::Rng;
use std::time::Instant;

pub struct Cfg {
    pub seed: u64,
    pub steps: usize,
    pub batch: usize,
    pub threads: usize,
    pub lr: f64,           // crête du premier cycle de chaque étage
    pub lr_min: f64,       // lr en fin de cycle
    pub cycle: usize,      // pas par cycle
    pub max_cycles: usize, // cycles max par étage
    pub peak_decay: f64,   // crête × peak_decay à chaque cycle de plus dans un même étage
    pub clip: f64,
    pub w_trace: f64,
    pub mut_frac: f64,
    pub up: f64,           // porte des étages intermédiaires
    pub gate: f64,         // porte de l'étage final
    pub every: usize,      // pas entre deux validations de suivi
    pub n_val: usize,
    pub n_gate: usize,     // taille de la validation de porte
    pub secs: f64,         // plafond de temps (0 = aucun)
    pub from: usize,       // étage de départ (0-based)
    pub kmax: usize,       // alphabet max des causes
    pub fixed: Option<(usize, usize)>, // plage fixe (t_hi, q_hi) : un seul étage, sans curriculum
    pub out: String,
    pub init: String,
}

/// Rampe de budgets : (T ≤ 6, q ≤ 3) → … → prior final.
pub fn stages(fin: &Prior) -> Vec<Prior> {
    let t0 = 6usize.max(fin.t_lo);
    let q0 = 3usize.max(fin.q_lo);
    let nt = (fin.t_hi.saturating_sub(t0) + 1) / 2;
    let nq = fin.q_hi.saturating_sub(q0);
    let n = nt.max(nq) + 1;
    (0..n)
        .map(|i| {
            let mut p = fin.clone();
            p.t_hi = (t0 + 2 * i).min(fin.t_hi);
            p.q_hi = (q0 + i).min(fin.q_hi);
            p
        })
        .collect()
}

pub fn save_ref(path: &str, p: &Prior) {
    let _ = std::fs::write(format!("{}.ref", path), format!("{} {} {}", p.t_hi, p.q_hi, p.k_max));
}

/// (t_hi, q_hi, k_max) de la plage d'entraînement du modèle, si le sidecar existe.
pub fn load_ref(path: &str) -> Option<(usize, usize, usize)> {
    let s = std::fs::read_to_string(format!("{}.ref", path)).ok()?;
    let mut it = s.split_whitespace().map(|x| x.parse::<usize>());
    let t = it.next()?.ok()?;
    let q = it.next()?.ok()?;
    let k = it.next()?.ok()?;
    Some((t, q, k))
}

fn checkpoint(net: &Net, out: &str, p: &Prior) {
    if let Err(e) = net.save(out) {
        eprintln!("sauvegarde impossible ({}) : {}", out, e);
    }
    save_ref(out, p);
}

/// Un échantillon d'entraînement : monde propre, ou voisin muté avec probabilité mut_frac. Jamais « non vu ».
pub fn draw_train(rng: &mut Rng, pr: &Prior, mut_frac: f64) -> Sample {
    let par = draw_world(rng, pr, Part::Train, 1_000_000).expect("prior d'entraînement vide");
    if rng.bernoulli(mut_frac) {
        if let Some(c) = mutate_sample(rng, &par, pr, 200) {
            if in_part(&c.prog, Part::Train) {
                return c;
            }
        }
    }
    par
}

struct Worker {
    rng: Rng,
    ws: Work,
    g: Vec<f64>,
}

impl Worker {
    fn new(rng: Rng, np: usize) -> Worker {
        Worker { rng, ws: Work::new(), g: vec![0.0; np] }
    }

    /// Gradient sommé sur n échantillons (triviaux inclus). Les statistiques ne comptent que les non triviaux.
    fn run(&mut self, net: &Net, pr: &Prior, n: usize, mut_frac: f64, w_trace: f64) -> Agg {
        for x in self.g.iter_mut() {
            *x = 0.0;
        }
        let mut a = Agg::default();
        for _ in 0..n {
            let s = draw_train(&mut self.rng, pr, mut_frac);
            let st = net.run_sample(&s, w_trace, &mut self.ws, Some(&mut self.g[..]));
            if nontrivial(&s) {
                a.add(&s, &st);
            }
        }
        a
    }
}

/// Renvoie (réseau, prior de la dernière plage entraînée).
pub fn train(cfg: &Cfg) -> (Net, Prior) {
    let mut fin = Prior::demo();
    fin.k_max = cfg.kmax;
    let sts: Vec<Prior> = match cfg.fixed {
        Some((t, q)) => {
            let mut p = fin.clone();
            p.t_hi = t.max(p.t_lo);
            p.q_hi = q.max(p.q_lo);
            vec![p]
        }
        None => stages(&fin),
    };
    let nst = sts.len();
    let mut rng = Rng::new(cfg.seed);
    let mut net = Net::new(&mut rng);
    if !cfg.init.is_empty() {
        match Net::load(&cfg.init) {
            Ok(n) => {
                net = n;
                println!("poids initiaux : {}", cfg.init);
            }
            Err(e) => {
                eprintln!("--init {} : {}", cfg.init, e);
                std::process::exit(1);
            }
        }
    }
    let np = net.p.len();
    let nth = cfg.threads.clamp(1, cfg.batch.max(1));
    let mut workers: Vec<Worker> = (0..nth).map(|_| Worker::new(rng.fork(), np)).collect();
    let mut vrng = rng.fork();
    let mut vws = Work::new();
    let mut adam = Adam::new(np);
    let mut g = vec![0.0f64; np];
    let cycle = cfg.cycle.max(1);
    let every = cfg.every.max(1);
    let mut stage = cfg.from.min(nst - 1);
    println!(
        "entraînement de D (v3) : {} paramètres, {} fil(s), lot {}, {} étage(s){}, départ étage {}, k ≤ {}, mutés {:.2}",
        np,
        nth,
        cfg.batch,
        nst,
        if cfg.fixed.is_some() { " (plage fixe)" } else { "" },
        stage + 1,
        fin.k_max,
        cfg.mut_frac
    );
    println!(
        "lr {:.1e} → {:.1e} par cycle de {} pas (crête ×{:.2} par cycle de plus), ≤ {} cycles/étage ; portes : up {:.0}% / final {:.0}% (fid sur mondes non triviaux non vus, n = {})",
        cfg.lr, cfg.lr_min, cycle, cfg.peak_decay, cfg.max_cycles, 100.0 * cfg.up, 100.0 * cfg.gate, cfg.n_gate
    );
    println!("étage {}/{} : T ≤ {}, qmax ≤ {}", stage + 1, nst, sts[stage].t_hi, sts[stage].q_hi);

    let clock = Instant::now();
    let (mut cyc, mut k, mut step, mut done) = (0usize, 0usize, 0usize, 0usize);
    let mut win = Agg::default();
    let (mut gsum, mut gcnt, mut bad) = (0.0f64, 0usize, 0usize);
    let mut why = "plafond de pas";
    let mut last_stage = stage;

    while step < cfg.steps {
        let pr = &sts[stage];
        last_stage = stage;
        let peak = (cfg.lr * cfg.peak_decay.powi(cyc as i32)).max(cfg.lr_min);
        let lr = cfg.lr_min
            + 0.5 * (peak - cfg.lr_min) * (1.0 + (std::f64::consts::PI * k as f64 / cycle as f64).cos());

        let (base, extra) = (cfg.batch / nth, cfg.batch % nth);
        let mut aggs: Vec<Agg> = Vec::with_capacity(nth);
        std::thread::scope(|sc| {
            let mut hs = Vec::with_capacity(nth);
            for (i, w) in workers.iter_mut().enumerate() {
                let n = base + usize::from(i < extra);
                let netr = &net;
                hs.push(sc.spawn(move || w.run(netr, pr, n, cfg.mut_frac, cfg.w_trace)));
            }
            for h in hs {
                aggs.push(h.join().expect("fil d'entraînement"));
            }
        });

        for x in g.iter_mut() {
            *x = 0.0;
        }
        for w in &workers {
            for (a, b) in g.iter_mut().zip(&w.g) {
                *a += *b;
            }
        }
        let inv = 1.0 / cfg.batch as f64;
        for x in g.iter_mut() {
            *x *= inv;
        }
        let gn = adam.step(&mut net.p, &g, lr, cfg.clip);
        if gn.is_finite() {
            gsum += gn;
            gcnt += 1;
        } else {
            bad += 1;
        }
        for a in &aggs {
            win.merge(a);
        }
        done += cfg.batch;
        step += 1;
        k += 1;

        let timeout = cfg.secs > 0.0 && clock.elapsed().as_secs_f64() >= cfg.secs;
        let cyc_end = k >= cycle;
        if step % every == 0 || cyc_end || timeout {
            let nv = if cyc_end { cfg.n_gate } else { cfg.n_val };
            let v = validate(&net, pr, nv, &mut vrng, &mut vws);
            let el = clock.elapsed().as_secs_f64();
            println!(
                "pas {:>6}  étage {}/{} (T≤{}, q≤{})  cycle {}/{}  lr {:.1e}  L {:.4}  train {:>5.1}%  val {:>5.1}% ±{:.1}{}  |g| {:.3}  {:.0} éch/s  {:.0} s",
                step,
                stage + 1,
                nst,
                pr.t_hi,
                pr.q_hi,
                cyc + 1,
                cfg.max_cycles,
                lr,
                win.mean_loss(),
                100.0 * win.fid(),
                100.0 * v.fid(),
                100.0 * v.ci(),
                if cyc_end { " [porte]" } else { "" },
                gsum / gcnt.max(1) as f64,
                done as f64 / el.max(1e-9),
                el
            );
            if bad > 0 {
                println!("  ! {} pas ignorés (gradient non fini)", bad);
            }
            win = Agg::default();
            gsum = 0.0;
            gcnt = 0;
            bad = 0;
            checkpoint(&net, &cfg.out, pr);
            if cyc_end {
                let is_final = stage + 1 == nst;
                let thr = if is_final { cfg.gate } else { cfg.up };
                if v.fid() >= thr {
                    if is_final {
                        why = "étage final maîtrisé";
                        break;
                    }
                    println!(
                        "  ↑ porte franchie ({:.1}% ≥ {:.0}%) → étage {}/{} : T ≤ {}, qmax ≤ {}",
                        100.0 * v.fid(),
                        100.0 * thr,
                        stage + 2,
                        nst,
                        sts[stage + 1].t_hi,
                        sts[stage + 1].q_hi
                    );
                    stage += 1;
                    cyc = 0;
                } else {
                    cyc += 1;
                    println!(
                        "  ✗ porte close ({:.1}% < {:.0}%) : {} cycle(s) sur {} dans l'étage {}/{}",
                        100.0 * v.fid(),
                        100.0 * thr,
                        cyc,
                        cfg.max_cycles,
                        stage + 1,
                        nst
                    );
                    if cyc >= cfg.max_cycles {
                        why = if cfg.fixed.is_some() {
                            "plan fixe terminé (cycles épuisés)"
                        } else {
                            "étage BLOQUÉ (porte jamais franchie)"
                        };
                        break;
                    }
                }
                k = 0;
            }
            if timeout {
                why = "plafond de temps";
                break;
            }
        }
    }
    checkpoint(&net, &cfg.out, &sts[last_stage]);
    println!(
        "fin : {} — dernier étage entraîné {}/{} ; {} pas, {} échantillons, {:.0} s",
        why,
        last_stage + 1,
        nst,
        step,
        done,
        clock.elapsed().as_secs_f64()
    );
    let p = sts[last_stage].clone();
    (net, p)
}