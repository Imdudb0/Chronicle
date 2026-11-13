// Entraînement de D : minibatchs parallèles (threads std), Adam, curriculum piloté par la fidélité.
//
// • Les programmes d'entraînement excluent le sous-ensemble « non vu » (hash(C) mod 8 = 0), voisins mutés compris.
// • Chaque fil a son Rng (fork du Rng principal), son Work et son tampon de gradient ; Net est partagé en lecture.
//   Résultat déterministe pour (seed, nombre de fils).
// • Curriculum : l'étage i borne T ≤ t_hi(i) et qmax ≤ q_hi(i). On monte d'un étage quand la fidélité
//   (régénération exacte ∧ trace exacte) mesurée sur programmes non vus de l'étage atteint `up`.
//   Pas d'avance forcée : un étage bloqué est un résultat (D ne sait pas juger plus profond).
// • Le seuil `up` et la rampe sont des boutons du grand livre, au même titre que τ et les vues.
#![allow(dead_code)]

use crate::eval::{draw_world, in_part, validate, Agg, Part};
use crate::meta::{mutate_sample, Prior, Sample};
use crate::net::{Adam, Net, Work};
use crate::rng::Rng;
use std::time::Instant;

pub struct Cfg {
    pub seed: u64,
    pub steps: usize,
    pub batch: usize,
    pub threads: usize,
    pub lr: f64,
    pub clip: f64,
    pub w_trace: f64,
    pub mut_frac: f64, // part du lot formée de voisins mutés
    pub up: f64,       // fid(val) pour monter d'un étage
    pub every: usize,  // pas entre deux validations
    pub n_val: usize,
    pub secs: f64,     // plafond de temps (0 = aucun)
    pub out: String,
    pub init: String,  // poids initiaux ("" = aucun)
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

/// Un fil de calcul : son flux aléatoire, son espace de travail, son tampon de gradient.
struct Worker {
    rng: Rng,
    ws: Work,
    g: Vec<f64>,
}

impl Worker {
    fn new(rng: Rng, np: usize) -> Worker {
        Worker { rng, ws: Work::new(), g: vec![0.0; np] }
    }

    fn run(&mut self, net: &Net, pr: &Prior, n: usize, mut_frac: f64, w_trace: f64) -> Agg {
        for x in self.g.iter_mut() {
            *x = 0.0;
        }
        let mut a = Agg::default();
        for _ in 0..n {
            let s = draw_train(&mut self.rng, pr, mut_frac);
            let st = net.run_sample(&s, w_trace, &mut self.ws, Some(&mut self.g[..]));
            a.add(&s, &st);
        }
        a
    }
}

pub fn train(cfg: &Cfg) -> Net {
    let fin = Prior::demo();
    let sts = stages(&fin);
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
    println!(
        "entraînement de D : {} paramètres, {} fil(s), lot {}, {} étages, voisins mutés {:.2}, lr {:.1e}, {} pas max{}",
        np,
        nth,
        cfg.batch,
        sts.len(),
        cfg.mut_frac,
        cfg.lr,
        cfg.steps,
        if cfg.secs > 0.0 { format!(", {:.0} s max", cfg.secs) } else { String::new() }
    );
    println!("étage 1/{} : T ≤ {}, qmax ≤ {}", sts.len(), sts[0].t_hi, sts[0].q_hi);

    let every = cfg.every.max(1);
    let clock = Instant::now();
    let mut stage = 0usize;
    // (pas, temps) d'entrée dans l'étage final : le lr ne décroît qu'à partir de là
    let mut final_at: Option<(usize, f64)> = if sts.len() == 1 { Some((0, 0.0)) } else { None };
    let mut win = Agg::default();
    let (mut gsum, mut gcnt, mut bad) = (0.0f64, 0usize, 0usize);
    let mut done = 0usize;

    for step in 1..=cfg.steps {
        let pr = &sts[stage];
        let lr = match final_at {
            None => cfg.lr,
            Some((s0, t0)) => {
                let fs = (step - s0) as f64 / cfg.steps.saturating_sub(s0).max(1) as f64;
                let ft = if cfg.secs > 0.0 {
                    (clock.elapsed().as_secs_f64() - t0) / (cfg.secs - t0).max(1.0)
                } else {
                    0.0
                };
                let f = fs.max(ft).clamp(0.0, 1.0);
                cfg.lr * (0.1 + 0.45 * (1.0 + (std::f64::consts::PI * f).cos()))
            }
        };

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
        let mut tot = Agg::default();
        for a in &aggs {
            tot.merge(a);
        }
        done += tot.n;
        win.merge(&tot);

        let timeout = cfg.secs > 0.0 && clock.elapsed().as_secs_f64() >= cfg.secs;
        if step % every == 0 || step == cfg.steps || timeout {
            let v = validate(&net, pr, cfg.n_val, &mut vrng, &mut vws);
            let el = clock.elapsed().as_secs_f64();
            println!(
                "pas {:>6}  étage {}/{} (T≤{}, q≤{})  lr {:.1e}  L {:.4}  fid(train) {:>5.1}%  fid(val) {:>5.1}% ±{:.1}  |g| {:.3}  {:.0} éch/s  {:.0} s",
                step,
                stage + 1,
                sts.len(),
                pr.t_hi,
                pr.q_hi,
                lr,
                win.mean_loss(),
                100.0 * win.fid(),
                100.0 * v.fid(),
                100.0 * v.ci(),
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
            if let Err(e) = net.save(&cfg.out) {
                eprintln!("sauvegarde impossible ({}) : {}", cfg.out, e);
            }
            if v.fid() >= cfg.up && stage + 1 < sts.len() {
                stage += 1;
                println!(
                    "  ↑ étage {}/{} : T ≤ {}, qmax ≤ {}   (fid(val) {:.1}% ≥ {:.0}%)",
                    stage + 1,
                    sts.len(),
                    sts[stage].t_hi,
                    sts[stage].q_hi,
                    100.0 * v.fid(),
                    100.0 * cfg.up
                );
                if stage + 1 == sts.len() {
                    final_at = Some((step, clock.elapsed().as_secs_f64()));
                }
            }
            if timeout {
                println!("plafond de temps atteint");
                break;
            }
        }
    }
    println!(
        "fin de l'entraînement : étage {}/{} atteint, {} échantillons, {:.0} s",
        stage + 1,
        sts.len(),
        done,
        clock.elapsed().as_secs_f64()
    );
    net
}