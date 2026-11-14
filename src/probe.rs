#![allow(dead_code)]
// Sondes sur D : réécritures neutres (equiv), hasard par longueur de file (hazard), lecture (reads).
use crate::eval::{canon_hash, draw_world, nontrivial, with_prog, Agg, Part};
use crate::meta::{mutate, rebyte, rename_idx, sample_world, Prior, Sample};
use crate::rng::Rng;
use crate::sizes::queue_lens;
use crate::tape::{decode, Net, Work, NOUT};
use std::collections::{HashMap, HashSet};

fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.below(i + 1);
        v.swap(i, j);
    }
}

/// Monde non vu (selon le split actif) et non trivial (E non constant).
fn draw(rng: &mut Rng, pr: &Prior) -> Option<Sample> {
    for _ in 0..1000 {
        if let Some(s) = draw_world(rng, pr, Part::Held, 1_000_000) {
            if nontrivial(&s) {
                return Some(s);
            }
        }
    }
    None
}

pub fn selfcheck() {
    let pr = Prior::demo();
    let mut rng = Rng::new(21);
    let alpha: Vec<u8> = (b'a'..=b'z').collect();
    let n = 3000usize;
    let mut diff = 0usize;
    for _ in 0..n {
        let s = sample_world(&mut rng, &pr, 100_000).expect("monde");
        let k = s.prog.k();
        let mut pi: Vec<u8> = (0..k as u8).collect();
        shuffle(&mut pi, &mut rng);
        let p1 = rename_idx(&s.prog, &pi);
        let x1 = with_prog(&s, p1.clone()).expect("renommage des règles");
        assert_eq!(x1.emit, s.emit, "E doit rester identique");
        assert_eq!(x1.qmax, s.qmax);
        for t in 0..s.t {
            assert_eq!(x1.trace[t], pi[s.trace[t] as usize], "la trace doit être relabellisée par pi");
        }
        assert_eq!(canon_hash(&p1), canon_hash(&s.prog), "canon invariant au renommage des règles");
        let p2 = rebyte(&s.prog, &mut rng, &alpha);
        assert_eq!(canon_hash(&p2), canon_hash(&s.prog), "canon aveugle aux octets");
        let x2 = with_prog(&s, p2).expect("renommage des octets");
        assert_eq!(x2.trace, s.trace, "la trace ne dépend pas des octets");
        let mut bits = 0.0;
        let c = mutate(&mut rng, &s.prog, &mut bits);
        diff += (canon_hash(&c) != canon_hash(&s.prog)) as usize;
    }
    println!("ok 1 : renommages (règles, octets) : E, trace et forme canonique cohérents sur {} mondes", n);
    println!("ok 2 : une mutation locale change la forme canonique dans {:.1} % des cas", 100.0 * diff as f64 / n as f64);
    assert!(diff * 10 >= n * 8, "le hash canonique sépare trop peu de mécanismes");
    println!("PCHECK OK");
}

pub fn equiv(net: &Net, pr: &Prior, n: usize, seed: u64) {
    let mut rng = Rng::new(seed ^ 0xE0E0);
    let mut ws = Work::new();
    let a26: Vec<u8> = (b'a'..=b'z').collect();
    let a256: Vec<u8> = (0..=255u8).collect();
    let names = ["identité", "règles permutées", "octets permutés a–z", "octets hors a–z"];
    let mut agg = vec![Agg::default(); 4];
    let (mut lost, mut won) = ([0usize; 4], [0usize; 4]);
    let mut mech: HashSet<u64> = HashSet::new();
    let (mut got, mut guard) = (0usize, 0usize);
    while got < n && guard < 10 * n.max(1) {
        guard += 1;
        let s = match draw(&mut rng, pr) {
            Some(s) => s,
            None => break,
        };
        let mut pi: Vec<u8> = (0..s.prog.k() as u8).collect();
        shuffle(&mut pi, &mut rng);
        let progs = vec![
            s.prog.clone(),
            rename_idx(&s.prog, &pi),
            rebyte(&s.prog, &mut rng, &a26),
            rebyte(&s.prog, &mut rng, &a256),
        ];
        let vars: Vec<Sample> = progs.into_iter().filter_map(|p| with_prog(&s, p)).collect();
        if vars.len() < 4 {
            continue;
        }
        mech.insert(canon_hash(&s.prog));
        let mut ok0 = false;
        for (i, x) in vars.iter().enumerate() {
            let st = net.run_sample(x, 1.0, &mut ws, None);
            agg[i].add(x, &st);
            let ok = st.regen && st.trace_all;
            if i == 0 {
                ok0 = ok;
            } else if ok0 && !ok {
                lost[i] += 1;
            } else if !ok0 && ok {
                won[i] += 1;
            }
        }
        got += 1;
    }
    println!("=== EQUIV : D sous réécritures neutres — {} mondes non vus, {} mécanismes distincts ===", got, mech.len());
    println!("{:<22}{:>7}{:>8}{:>8}{:>8}{:>9}{:>10}", "réécriture", "n", "fid%", "régén%", "trace%", "oct/tk%", "perd/gagne");
    for i in 0..4 {
        let a = &agg[i];
        println!(
            "{:<22}{:>7}{:>8.1}{:>8.1}{:>8.1}{:>9.2}{:>7}/{:<3}",
            names[i], a.n, 100.0 * a.fid(), 100.0 * a.regen_rate(), 100.0 * a.trace_rate(), 100.0 * a.byte_acc(), lost[i], won[i]
        );
    }
    println!("perd = juste avant, faux après ; gagne = l'inverse (≈ 0 attendu si D est invariant)");
    for i in 1..4 {
        let r = agg[i].fid() / agg[0].fid().max(1e-9);
        let v = if r >= 0.97 { "INVARIANT" } else if r >= 0.85 { "GRIS" } else { "EMPREINTE" };
        println!("{:<22} fid(variante)/fid(origine) = {:.2}  →  {}", names[i], r, v);
    }
    println!("règle (fixée avant mesure) : INVARIANT si ≥ 0,97 ; EMPREINTE si < 0,85. IC optimistes : tirages corrélés par mécanisme.");
}

// ---- seconde moitié : hazard (risque par tick à ℓ apparié), reads (lecture : τ, oracle, anatomie) ----

#[derive(Clone, Copy, Default)]
struct Rk {
    n: usize,
    e: usize,
}
impl Rk {
    fn add(&mut self, bad: bool) {
        self.n += 1;
        self.e += bad as usize;
    }
    fn h(&self) -> f64 {
        self.e as f64 / self.n.max(1) as f64
    }
}

fn pooled(by: &[Vec<Rk>], cs: &[usize], lo: usize, hi: usize) -> Rk {
    let mut r = Rk::default();
    for &c in cs {
        for l in lo..=hi.min(by[c].len() - 1) {
            r.n += by[c][l].n;
            r.e += by[c][l].e;
        }
    }
    r
}

/// Premier tick fautif de D sur s, lu dans ws.y après run_sample (même critère que diag et sizes).
fn first_error(s: &Sample, ws: &Work) -> Option<usize> {
    for k in 0..s.t {
        let (b, r) = decode(&ws.y[k * NOUT..(k + 1) * NOUT]);
        if r != (s.trace[k] as usize) || (s.flag[k] && b != s.emit[k]) {
            return Some(k);
        }
    }
    None
}

/// Âge de la cellule lue à chaque tick = ticks depuis son écriture (file initiale : écrite au tick −1).
/// Convention de la bande : la tête est en position absolue m·k, les productions s'ajoutent à la queue.
fn head_ages(p: &crate::meta::Program, trace: &[u8]) -> Vec<usize> {
    let mut w: Vec<i64> = vec![-1; p.q0.len()];
    let mut out = Vec::with_capacity(trace.len());
    for (k, &j) in trace.iter().enumerate() {
        out.push((k as i64 - w[p.m * k]) as usize);
        for _ in 0..p.prod[j as usize].len() {
            w.push(k as i64);
        }
    }
    out
}

fn band(l: usize, q: usize) -> usize {
    if l <= q {
        0
    } else if l <= q + q / 2 {
        1
    } else if l <= 2 * q {
        2
    } else {
        3
    }
}

fn band_name(b: usize, q: usize) -> String {
    match b {
        0 => format!("ℓ ≤ {}", q),
        1 => format!("ℓ ∈ ]{}, {}]", q, q + q / 2),
        2 => format!("ℓ ∈ ]{}, {}]", q + q / 2, 2 * q),
        _ => format!("ℓ > {}", 2 * q),
    }
}

fn opt2(x: Option<f64>) -> String {
    match x {
        Some(v) => format!("{:.2}", v),
        None => "n/a".to_string(),
    }
}

/// Valide les conventions de position que `reads` suppose : tête en m·k, ℓ = queue − m·k.
fn check_ages() {
    use crate::meta::{run, Program};
    let rep = Program { m: 1, bytes: vec![b'a', b'b'], prod: vec![vec![0], vec![1]], q0: vec![0, 1] };
    assert_eq!(head_ages(&rep, &run(&rep, 8).unwrap().trace), vec![1, 2, 2, 2, 2, 2, 2, 2]);
    let t2 = Program { m: 2, bytes: vec![b'a', b'b'], prod: vec![vec![0, 1], vec![1]], q0: vec![0, 1] };
    assert_eq!(head_ages(&t2, &run(&t2, 4).unwrap().trace), vec![1, 1, 1, 1]);
    let pr = Prior::demo();
    let mut rng = Rng::new(33);
    let (n, mut amax) = (2000usize, 0usize);
    for _ in 0..n {
        let s = sample_world(&mut rng, &pr, 100_000).expect("monde");
        let ages = head_ages(&s.prog, &s.trace);
        let ql = queue_lens(&s.prog, s.t);
        assert_eq!(ages.len(), s.t);
        assert_eq!(ql.len(), s.t);
        let mut tail = s.prog.q0.len();
        for k in 0..s.t {
            assert!(ages[k] >= 1);
            assert_eq!(ql[k], tail - s.prog.m * k, "ℓ = queue − m·k");
            tail += s.prog.prod[s.trace[k] as usize].len();
            amax = amax.max(ages[k]);
        }
    }
    println!("ok 3 : tête en m·k, ℓ = queue − m·k, âges ≥ 1 sur {} mondes (âge max {})", n, amax);
}

pub fn hazard(net: &Net, pr: &Prior, q: usize, n: usize, seed: u64) {
    let mut rng = Rng::new(seed ^ 0x4A2A);
    let mut ws = Work::new();
    let lm = pr.q_hi.max(q + 1) + 1;
    // classes : 0 peu profond ; 1 profond, avant le premier dépassement de q ; 2 profond, depuis ce dépassement
    let mut by = vec![vec![Rk::default(); lm]; 3];
    // depuis le dépassement, ℓ > q : [surplus jamais lu (utile ≤ q), surplus encore à lire (utile > q)]
    let mut un = vec![[Rk::default(); 2]; lm];
    let (mut nw, mut got, mut guard) = ([0usize; 2], 0usize, 0usize);
    while got < n && guard < 20 * n.max(1) {
        guard += 1;
        let s = match draw(&mut rng, pr) {
            Some(s) => s,
            None => break,
        };
        net.run_sample(&s, 1.0, &mut ws, None);
        let ql = queue_lens(&s.prog, s.t);
        let fd = ql.iter().position(|&x| x > q);
        let fe = first_error(&s, &ws);
        for k in 0..=fe.unwrap_or(s.t - 1) {
            let l = ql[k].min(lm - 1);
            let bad = fe == Some(k);
            let c = match fd {
                None => 0,
                Some(f) if k < f => 1,
                Some(_) => 2,
            };
            by[c][l].add(bad);
            if c == 2 && l > q {
                let u = ql[k].min(s.prog.m * (s.t - k));
                un[l][(u > q) as usize].add(bad);
            }
        }
        nw[fd.is_some() as usize] += 1;
        got += 1;
    }
    println!("=== HAZARD : risque par tick selon la longueur de file ℓ (D entraîné pour ℓ ≤ {}) ===", q);
    println!(
        "{} mondes non vus non triviaux : {} peu profonds (ℓ ≤ {} à tous les ticks), {} profonds | T ∈ [{}, {}], qmax ≤ {}",
        got, nw[0], q, nw[1], pr.t_lo, pr.t_hi, pr.q_hi
    );
    println!("risque = P(première erreur à ce tick | juste jusque-là), en % ; ℓ = file au début du tick, avant la lecture");
    println!(
        "avant / après = avant / depuis le premier tick où ℓ > {} ; non lu / lu = après dépassement, utile = min(ℓ, m·ticks restants) ≤ {} / > {}\n",
        q, q, q
    );
    let cell = |x: Rk| -> String {
        if x.n == 0 {
            format!("{:>6} {:>6}", "·", "·")
        } else {
            format!("{:>6} {:>6.2}", x.n, 100.0 * x.h())
        }
    };
    println!("{:>3} | {:^13} | {:^13} | {:^13} | {:^13} | {:^13}", "ℓ", "peu profonds", "prof. avant", "prof. après", "après non lu", "après lu");
    println!("{:>3} | {:^13} | {:^13} | {:^13} | {:^13} | {:^13}", "", "ticks  risq%", "ticks  risq%", "ticks  risq%", "ticks  risq%", "ticks  risq%");
    for l in 1..lm {
        let row = [by[0][l], by[1][l], by[2][l], un[l][0], un[l][1]];
        if row.iter().all(|x| x.n == 0) {
            continue;
        }
        println!("{:>3} | {} | {} | {} | {} | {}", l, cell(row[0]), cell(row[1]), cell(row[2]), cell(row[3]), cell(row[4]));
    }
    let lo_edge = q.saturating_sub(1).max(1);
    let edge = pooled(&by, &[0, 1, 2], lo_edge, q);
    let smr = |r: Rk| -> Option<f64> {
        let ex = r.n as f64 * edge.h();
        if ex >= 5.0 {
            Some(r.e as f64 / ex)
        } else {
            None
        }
    };
    // (1) classe : profonds avant dépassement contre peu profonds, à ℓ égal
    let (mut obs, mut exp_c, mut cov) = (0usize, 0.0f64, 0usize);
    for l in 1..=q.min(lm - 1) {
        if by[0][l].n >= 30 {
            obs += by[1][l].e;
            exp_c += by[1][l].n as f64 * by[0][l].h();
            cov += by[1][l].n;
        }
    }
    let smr_c = if exp_c >= 5.0 { Some(obs as f64 / exp_c) } else { None };
    // (2) marche : après dépassement, par bande de ℓ, contre le risque au bord de la plage
    let r1 = pooled(&by, &[2], q + 1, q + 2);
    let r2 = pooled(&by, &[2], q + 3, 2 * q);
    let r3 = pooled(&by, &[2], 2 * q + 1, lm);
    // (3) surplus jamais lu / encore à lire
    let (mut nl, mut lu) = (Rk::default(), Rk::default());
    for l in q + 1..lm {
        nl.n += un[l][0].n;
        nl.e += un[l][0].e;
        lu.n += un[l][1].n;
        lu.e += un[l][1].e;
    }
    println!(
        "\n(1) CLASSE, à ℓ égal (ℓ ≤ {}) : profonds avant dépassement, {} erreurs observées / {:.1} attendues au risque des peu profonds ({} ticks) → SMR {}",
        q, obs, exp_c, cov, opt2(smr_c)
    );
    println!(
        "(2) MARCHE, risque en % : bord ℓ ∈ [{}, {}] = {:.2} (n = {}) | après dépassement ℓ ∈ [{}, {}] = {:.2} (SMR {}) | ℓ ∈ [{}, {}] = {:.2} (SMR {}) | ℓ ≥ {} = {:.2} (SMR {})",
        lo_edge, q, 100.0 * edge.h(), edge.n,
        q + 1, q + 2, 100.0 * r1.h(), opt2(smr(r1)),
        q + 3, 2 * q, 100.0 * r2.h(), opt2(smr(r2)),
        2 * q + 1, 100.0 * r3.h(), opt2(smr(r3))
    );
    println!(
        "(3) SURPLUS après dépassement (ℓ > {}) : jamais lu {:.2} % (n = {}, SMR {}) | encore à lire {:.2} % (n = {}, SMR {})   [SMR contre le bord]",
        q, 100.0 * nl.h(), nl.n, opt2(smr(nl)), 100.0 * lu.h(), lu.n, opt2(smr(lu))
    );
    let hi3 = |x: Option<f64>| x.map_or(false, |v| v >= 3.0);
    let verdict = if nw[1] < 100 {
        "ÉCHANTILLON INSUFFISANT (moins de 100 mondes profonds) : augmenter --n"
    } else if hi3(smr_c) {
        "CLASSE : à ℓ égal les profonds ratent déjà avant le dépassement ; le biais est dans la classe de programme, pas dans ℓ"
    } else if hi3(smr(r1)) || hi3(smr(r2)) || hi3(smr(r3)) {
        match smr(nl) {
            Some(v) if v >= 3.0 => "PRÉSENCE : D se dégrade dès que la file dépasse q, même quand le surplus ne sera jamais lu → lecture ou marqueur de queue",
            Some(_) => "UTILE : seul le surplus encore à lire dégrade D → rétention",
            None => "TAILLE, mais « jamais lu » trop rare pour départager PRÉSENCE et UTILE : augmenter --n",
        }
    } else {
        "AUCUN EXCÈS NET à ℓ apparié : ni classe ni taille ; regarder l'indice de tick et le type de production"
    };
    println!("règle (fixée avant mesure) : CLASSE si SMR (1) ≥ 3 ; sinon, si un SMR de (2) ≥ 3 : PRÉSENCE si le SMR « jamais lu » de (3) ≥ 3, sinon UTILE ; sinon AUCUN EXCÈS NET. n ≥ 100 profonds ; SMR ≥ 3 excès net, < 1,5 nul.");
    println!("verdict : {}", verdict);
}

#[derive(Clone, Default)]
struct Res {
    nw: [usize; 2],    // mondes peu profonds / profonds
    ok: [usize; 2],    // …justes de bout en bout
    h: [Rk; 2],        // mondes profonds : risque par tick pour ℓ ≤ q / ℓ > q
    age: [[Rk; 3]; 2], // oracle : [ℓ ≤ q / ℓ > q][âge ≤ q/2 / ≤ q / > q]
}

#[derive(Clone, Copy, Default)]
struct Anat {
    n: usize,
    hit: usize,
    head: f64,
    live: f64,
    old: f64,
    beyond: f64,
    err: usize,
    miss: usize,
}

pub fn reads(net: &mut Net, pr: &Prior, q: usize, taus: &[f64], n: usize, seed: u64) {
    if taus.is_empty() {
        return;
    }
    let mut rng = Rng::new(seed ^ 0x8EAD);
    let mut set: Vec<(Sample, Vec<usize>)> = Vec::new();
    let mut guard = 0usize;
    while set.len() < n && guard < 20 * n.max(1) {
        guard += 1;
        match draw(&mut rng, pr) {
            Some(s) => {
                let ql = queue_lens(&s.prog, s.t);
                set.push((s, ql));
            }
            None => break,
        }
    }
    let nt = taus.len();
    let mut conf: Vec<(f64, bool)> = taus.iter().map(|&t| (t, false)).collect();
    conf.push((1.0, true)); // oracle : τ = 1, attention forcée sur la cellule m·k
    let mut ws = Work::new();
    let mut res: Vec<Res> = Vec::new();
    let mut an = [Anat::default(); 4];
    let mut off: HashMap<i64, usize> = HashMap::new();
    for (ci, &(tau, hard)) in conf.iter().enumerate() {
        net.tau = tau;
        net.hard = hard;
        let mut r = Res::default();
        for (s, ql) in &set {
            net.run_sample(s, 1.0, &mut ws, None);
            let fe = first_error(s, &ws);
            let deep = ql.iter().any(|&x| x > q) as usize;
            r.nw[deep] += 1;
            r.ok[deep] += fe.is_none() as usize;
            let ages = if hard { head_ages(&s.prog, &s.trace) } else { Vec::new() };
            for k in 0..=fe.unwrap_or(s.t - 1) {
                let bad = fe == Some(k);
                let big = (ql[k] > q) as usize;
                if deep == 1 {
                    r.h[big].add(bad);
                }
                if hard {
                    let ab = if ages[k] <= q / 2 { 0 } else if ages[k] <= q { 1 } else { 2 };
                    r.age[big][ab].add(bad);
                }
                if ci == 0 {
                    // anatomie de la lecture au réglage de base
                    let a = ws.att_row(k);
                    let head = s.prog.m * k;
                    let tail = head + ql[k];
                    let (mut am, mut mx) = (0usize, f64::NEG_INFINITY);
                    let (mut live, mut old, mut beyond) = (0.0f64, 0.0f64, 0.0f64);
                    for (i, &v) in a.iter().enumerate() {
                        if v > mx {
                            mx = v;
                            am = i;
                        }
                        if i < head {
                            old += v;
                        } else if i > head && i < tail {
                            live += v;
                        } else if i >= tail {
                            beyond += v;
                        }
                    }
                    let x = &mut an[band(ql[k], q)];
                    x.n += 1;
                    x.hit += (am == head) as usize;
                    x.head += a[head];
                    x.live += live;
                    x.old += old;
                    x.beyond += beyond;
                    if bad {
                        x.err += 1;
                        x.miss += (am != head) as usize;
                        if big == 1 {
                            *off.entry(am as i64 - head as i64).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        res.push(r);
    }
    net.tau = 1.0;
    net.hard = false;

    let nw = res[0].nw;
    let fid = |r: &Res, c: usize| r.ok[c] as f64 / r.nw[c].max(1) as f64;
    println!("=== READS : D (bande) sous sondes de lecture — {} mondes non vus non triviaux, les mêmes pour tous les réglages ===", set.len());
    println!(
        "peu profonds (ℓ ≤ {} à tous les ticks) : {} | profonds : {} | T ∈ [{}, {}], qmax ≤ {}",
        q, nw[0], nw[1], pr.t_lo, pr.t_hi, pr.q_hi
    );
    println!("\n(1) LECTURE : température de la softmax de lecture (logits × τ) et oracle de tête (attention forcée sur la cellule m·k)");
    println!("{:<18}{:>14}{:>14}{:>18}{:>18}", "réglage", "fid peu prof %", "fid prof. %", "risque ℓ≤q %/tick", "risque ℓ>q %/tick");
    for (i, (r, c)) in res.iter().zip(conf.iter()).enumerate() {
        let name = if c.1 {
            "oracle m·k".to_string()
        } else if i == 0 {
            format!("τ = {} (base)", c.0)
        } else {
            format!("τ = {}", c.0)
        };
        println!(
            "{:<18}{:>14.1}{:>14.1}{:>18.2}{:>18.2}",
            name,
            100.0 * fid(r, 0),
            100.0 * fid(r, 1),
            100.0 * r.h[0].h(),
            100.0 * r.h[1].h()
        );
    }

    println!("\n(2) ANATOMIE de la lecture au réglage de base — ticks à risque (jusqu'au premier tick fautif inclus)");
    println!(
        "{:<14}{:>8}{:>12}{:>9}{:>9}{:>10}{:>11}{:>10}{:>12}",
        "file", "ticks", "argmax=m·k", "m. tête", "m. vive", "m. passée", "m. au-delà", "1res err.", "dont ≠ m·k"
    );
    for b in 0..4 {
        let x = an[b];
        if x.n == 0 {
            continue;
        }
        let nn = x.n as f64;
        println!(
            "{:<14}{:>8}{:>11.1}%{:>9.2}{:>9.2}{:>10.2}{:>11.2}{:>10}{:>11.0}%",
            band_name(b, q),
            x.n,
            100.0 * x.hit as f64 / nn,
            x.head / nn,
            x.live / nn,
            x.old / nn,
            x.beyond / nn,
            x.err,
            100.0 * x.miss as f64 / x.err.max(1) as f64
        );
    }
    let mut ov: Vec<(i64, usize)> = off.into_iter().collect();
    ov.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let tot: usize = ov.iter().map(|x| x.1).sum();
    let top: Vec<String> = ov.iter().take(6).map(|&(d, c)| format!("{:+} : {}", d, c)).collect();
    println!(
        "décalage (argmax − m·k) aux premières erreurs avec ℓ > {} — {} erreurs : {}   (+0 = bonne cellule, sortie fausse)",
        q,
        tot,
        if top.is_empty() { "—".to_string() } else { top.join(" | ") }
    );

    let ro = &res[nt];
    println!("\n(3) ORACLE m·k : risque par tick selon l'âge de la cellule lue (ticks depuis son écriture) et ℓ — tous mondes");
    let cellr = |x: Rk| -> String {
        if x.n == 0 {
            "·".to_string()
        } else {
            format!("{} ticks, {:.2} %", x.n, 100.0 * x.h())
        }
    };
    println!(
        "{:<10}{:>26}{:>26}{:>26}",
        "",
        format!("âge ≤ {}", q / 2),
        format!("âge ∈ ]{}, {}]", q / 2, q),
        format!("âge > {}", q)
    );
    for j in 0..2 {
        println!(
            "{:<10}{:>26}{:>26}{:>26}",
            if j == 0 { format!("ℓ ≤ {}", q) } else { format!("ℓ > {}", q) },
            cellr(ro.age[j][0]),
            cellr(ro.age[j][1]),
            cellr(ro.age[j][2])
        );
    }

    let (s0, f0) = (fid(&res[0], 0), fid(&res[0], 1));
    let ft = (1..nt).map(|i| fid(&res[i], 1)).fold(f64::NEG_INFINITY, f64::max);
    let (so, fo) = (fid(ro, 0), fid(ro, 1));
    let hit0 = an[0].hit as f64 / an[0].n.max(1) as f64;
    let prem = an[0].n > 0 && hit0 >= 0.9 && so >= 0.9 * s0;
    let loc = if nw[1] < 100 {
        "INCONCLUSIF (moins de 100 mondes profonds)"
    } else if !prem {
        "ORACLE MAL POSÉ : D ne lit pas en m·k (argmax ou fid peu profonds hors contrôle) ; ignorer l'oracle, lire (2)"
    } else if fo >= 0.8 * s0 {
        "REPÉRAGE : avec la bonne cellule D est juste ; le défaut est la localisation de la tête"
    } else if fo >= f0 + 0.20 {
        "REPÉRAGE PARTIEL : la bonne cellule aide beaucoup sans suffire ; lire aussi (3)"
    } else if fo < f0 + 0.05 {
        "CONTENU : forcer la bonne cellule ne change rien ; la tête est trouvée, son contenu ou le calcul est faux ; lire (3)"
    } else {
        "MIXTE : gain de l'oracle entre 5 et 20 points ; lire (2) et (3)"
    };
    let sharp = nt > 1 && nw[1] >= 100 && ft >= f0 + 0.10;
    let ratio = |a: Rk, b: Rk| -> Option<f64> {
        if a.n >= 200 && b.n >= 200 && b.e > 0 {
            Some(a.h() / b.h())
        } else {
            None
        }
    };
    let pres = ratio(ro.age[1][0], ro.age[0][0]);
    let rete = ratio(ro.age[1][2], ro.age[1][0]);
    let yn = |x: Option<f64>| match x {
        Some(v) if v >= 3.0 => "OUI",
        Some(_) => "non",
        None => "n/a",
    };
    println!(
        "\ncontrôle d'instrument : bande ℓ ≤ {} : argmax = m·k sur {:.1} % des ticks ; fid peu profonds sous oracle {:.1} % (base {:.1} %)",
        q, 100.0 * hit0, 100.0 * so, 100.0 * s0
    );
    println!("règle (fixée avant mesure), profonds ≥ 100 : instrument valide si argmax = m·k ≥ 90 % en bande ℓ ≤ q ET fid(oracle, peu profonds) ≥ 0,9·fid(base, peu profonds) ;");
    println!("  REPÉRAGE si fid(oracle, profonds) ≥ 0,8·fid(base, peu profonds) ; PARTIEL si ≥ fid(base, profonds) + 20 points ; CONTENU si < fid(base, profonds) + 5 points ; sinon MIXTE ;");
    println!("  NETTETÉ si max sur τ > 1 de fid(profonds) ≥ fid(base, profonds) + 10 points ;");
    println!("  (3) PRÉSENCE si risque(ℓ > q, âge ≤ q/2) ≥ 3 × risque(ℓ ≤ q, âge ≤ q/2) ; RÉTENTION si risque(ℓ > q, âge > q) ≥ 3 × risque(ℓ > q, âge ≤ q/2) ; n ≥ 200 ticks par case.");
    println!("oracle : {}", loc);
    println!("netteté (τ > 1) : {}", if sharp { "OUI" } else { "NON" });
    println!("(3) PRÉSENCE : rapport {} → {} | RÉTENTION : rapport {} → {}", opt2(pres), yn(pres), opt2(rete), yn(rete));
}