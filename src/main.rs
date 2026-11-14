mod meta;
mod rng;
mod net;
mod eval;
mod train;
mod diag;
mod sizes;
mod tape;
mod probe;

use meta::*;
use rng::Rng;

fn selftest() {
    let (a, b) = (b'a', b'b');
    // 1. « répété » et « alterné » : deux causes, même effet.
    let rep = Program { m: 1, bytes: vec![a, b], prod: vec![vec![0], vec![1]], q0: vec![0, 1] };
    let alt = Program { m: 1, bytes: vec![a, b], prod: vec![vec![1], vec![0]], q0: vec![0] };
    let er = emit_of(&rep, &run(&rep, 10).unwrap().trace);
    let ea = emit_of(&alt, &run(&alt, 10).unwrap().trace);
    assert_eq!(er, b"ababababab".to_vec());
    assert_eq!(ea, er);
    println!("ok 1 : {}  et  {}  →  {}", rep.show(), alt.show(), text(&er));
    // 2. 2-tag : pop 0, supprime 1, ajoute « 0 1 » → file stable.
    let t2 = Program { m: 2, bytes: vec![a, b], prod: vec![vec![0, 1], vec![1]], q0: vec![0, 1] };
    let r2 = run(&t2, 4).unwrap();
    assert_eq!(r2.trace, vec![0, 0, 0, 0]);
    assert_eq!(r2.qmax, 2);
    assert_eq!(r2.fired, 1);
    println!("ok 2 : 2-tag");
    // 3. non productif.
    let dead = Program { m: 1, bytes: vec![a], prod: vec![vec![]], q0: vec![0] };
    assert!(run(&dead, 1).is_some());
    assert!(run(&dead, 2).is_none());
    println!("ok 3 : file épuisée détectée");
    // 4. mondes : invariants, encodeur, comptabilité.
    let pr = Prior::demo();
    let mut rng = Rng::new(7);
    let mut tot_att = 0u64;
    let n = 2000;
    for _ in 0..n {
        let s = sample_world(&mut rng, &pr, 100_000).expect("monde");
        tot_att += s.attempts as u64;
        assert_eq!(s.trace.len(), s.t);
        assert_eq!(s.emit.len(), s.t);
        assert!(s.flag.iter().any(|&f| f));
        let k = s.prog.k();
        assert_eq!(run(&s.prog, s.t).unwrap().fired, (1u32 << k) - 1);
        let mut z = vec![0.0f32; Z_DIM];
        encode(&s.prog, &s.view, &mut z);
        for j in 0..K_MAX {
            let base = 2 + j * ROW;
            for sl in 0..W_MAX {
                let lo = base + 9 + sl * NSLOT;
                let sum: f32 = z[lo..lo + NSLOT].iter().sum();
                assert_eq!(sum, if j < k { 1.0 } else { 0.0 });
            }
        }
        let qb = 2 + K_MAX * ROW;
        for sl in 0..Q_MAX {
            let sum: f32 = z[qb + sl * NSLOT..qb + (sl + 1) * NSLOT].iter().sum();
            assert_eq!(sum, 1.0);
        }
        assert_eq!(z[H_DIM..H_DIM + 3].iter().sum::<f32>(), 1.0);
        assert_eq!(z[H_DIM + 3..H_DIM + 7].iter().sum::<f32>(), 1.0);
        // 5. mutateur : programme différent, coût > 0.
        let mut bits = 0.0;
        let np = mutate(&mut rng, &s.prog, &mut bits);
        assert!(np != s.prog && bits > 0.0);
    }
    println!("ok 4 : {} mondes, tentatives moyennes par monde accepté = {:.2}", n, tot_att as f64 / n as f64);
    // 6. déterminisme.
    let (mut r1, mut r2) = (Rng::new(42), Rng::new(42));
    let s1 = sample_world(&mut r1, &pr, 100_000).unwrap();
    let s2 = sample_world(&mut r2, &pr, 100_000).unwrap();
    assert_eq!(s1.prog, s2.prog);
    assert_eq!(s1.trace, s2.trace);
    println!("ok 5 : déterminisme\nSELFTEST OK");
}

fn selftest_trunc() {
    let (a, b) = (b'a', b'b');
    // 1. a→aab depuis q0 = a : files lues 1, 3, 5, 7 ; avec cap 6, on garde les ticks 0..2
    let g = Program { m: 1, bytes: vec![a, b], prod: vec![vec![0, 0, 1], vec![]], q0: vec![0] };
    assert_eq!(sizes::queue_lens(&g, 4), vec![1, 3, 5, 7]);
    assert_eq!(run_cap(&g, 10, 6).unwrap().trace, vec![0, 0, 0]);
    // 2. sans dépassement, run_cap = run
    let rep = Program { m: 1, bytes: vec![a, b], prod: vec![vec![0], vec![1]], q0: vec![0, 1] };
    assert_eq!(run_cap(&rep, 9, 6).unwrap().trace, run(&rep, 9).unwrap().trace);
    // 3. file épuisée : None, comme run
    let dead = Program { m: 1, bytes: vec![a], prod: vec![vec![]], q0: vec![0] };
    assert!(run_cap(&dead, 2, 6).is_none());
    println!("ok 1 : run_cap");
    // 4. mondes tronqués et enfants mutés : invariants
    let mut pr = Prior::demo();
    pr.trunc = true;
    pr.q_hi = 3;
    let mut rng = Rng::new(11);
    let mut tot_att = 0u64;
    let n = 3000;
    for _ in 0..n {
        let s = sample_world(&mut rng, &pr, 100_000).expect("monde");
        tot_att += s.attempts as u64;
        assert!(s.t >= 1 && s.t > s.view.o && s.flag.iter().any(|&f| f));
        assert_eq!(run(&s.prog, s.t).unwrap().trace, s.trace);
        assert!(sizes::queue_lens(&s.prog, s.t).iter().all(|&l| l <= pr.q_hi));
        if let Some(c) = mutate_sample(&mut rng, &s, &pr, 200) {
            assert!(c.t >= 1 && c.t > c.view.o && c.t <= s.t);
            assert_eq!(run(&c.prog, c.t).unwrap().trace, c.trace);
            assert!(sizes::queue_lens(&c.prog, c.t).iter().all(|&l| l <= pr.q_hi));
        }
    }
    println!("ok 2 : {} mondes tronqués (q ≤ {}), tentatives moyennes {:.2}", n, pr.q_hi, tot_att as f64 / n as f64);
    println!("TCHECK OK");
}

fn monde(args: &[String]) {
    let seed: u64 = args.get(0).and_then(|s| s.parse().ok()).unwrap_or(1);
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let pr = Prior::demo();
    let mut rng = Rng::new(seed);
    for _ in 0..n {
        let s = sample_world(&mut rng, &pr, 100_000).unwrap();
        let fl: String = s.flag.iter().map(|&f| if f { '#' } else { '.' }).collect();
        println!("{}", s.prog.show());
        println!("  E   = {}", text(&s.emit));
        println!("  vue = {}  (o={}, s={})  T={}  qmax={}", fl, s.view.o, s.view.s, s.t, s.qmax);
        println!("  |p|={:.1} b  |V|={:.1} b  |T|={:.1} b  tentatives={}", s.bits_p, s.bits_v, s.bits_t, s.attempts);
        if let Some(c) = mutate_sample(&mut rng, &s, &pr, 200) {
            println!("  mut = {}\n  E'  = {}  (|p'|={:.1} b)", c.prog.show(), text(&c.emit), c.bits_p);
        }
    }
}

fn arg<T: std::str::FromStr>(a: &[String], i: usize, d: T) -> T {
    a.get(i).and_then(|s| s.parse::<T>().ok()).unwrap_or(d)
}

fn opt<T: std::str::FromStr>(a: &[String], key: &str, d: T) -> T {
    a.windows(2)
        .find(|w| w[0] == key)
        .and_then(|w| w[1].parse::<T>().ok())
        .unwrap_or(d)
}

fn ecfg(a: &[String]) -> eval::ECfg {
    eval::ECfg {
        n: opt(a, "--n", 2000usize),
        secs: opt(a, "--esecs", 120.0f64),
        seed: opt(a, "--eseed", 9001u64),
        gate: opt(a, "--gate", 0.95f64),
    }
}

fn load_or_exit(path: &str) -> tape::Net {
    match tape::Net::load(path) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("{} : {}", path, e);
            std::process::exit(1);
        }
    }
}

/// Plage de référence (1×) : --thi / --qhi / --kmax si donnés, sinon <modèle>.ref, sinon le prior final.
fn ref_prior(a: &[String], path: &str) -> Prior {
    let mut p = Prior::demo();
    if let Some((t, q, k)) = train::load_ref(path) {
        p.t_hi = t;
        p.q_hi = q;
        p.k_max = k;
    }
    p.t_hi = opt(a, "--thi", p.t_hi);
    p.q_hi = opt(a, "--qhi", p.q_hi);
    p.k_max = opt(a, "--kmax", p.k_max).clamp(1, K_MAX);
    p.pool = opt(a, "--pool", p.pool).clamp(K_MAX, 256);
    p
}

fn cmd_train(a: &[String]) {
    let threads_def = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let d = Prior::demo();
    let fixed = if a.iter().any(|x| x.as_str() == "--thi" || x.as_str() == "--qhi") {
        Some((opt(a, "--thi", d.t_hi), opt(a, "--qhi", d.q_hi)))
    } else {
        None
    };
    let cfg = train::Cfg {
        seed: opt(a, "--seed", 1u64),
        steps: opt(a, "--steps", 400_000usize),
        batch: opt(a, "--batch", 64usize),
        threads: opt(a, "--threads", threads_def),
        lr: opt(a, "--lr", 2e-3f64),
        lr_min: opt(a, "--lrmin", 1e-4f64),
        cycle: opt(a, "--cycle", 8000usize),
        max_cycles: opt(a, "--maxcyc", 4usize),
        peak_decay: opt(a, "--peakdecay", 0.7f64),
        clip: opt(a, "--clip", 1.0f64),
        w_trace: opt(a, "--wtrace", 1.0f64),
        mut_frac: opt(a, "--mut", 0.25f64),
        up: opt(a, "--up", 0.90f64),
        gate: opt(a, "--gate", 0.95f64),
        every: opt(a, "--every", 1000usize),
        n_val: opt(a, "--nval", 500usize),
        n_gate: opt(a, "--ngate", 3000usize),
        secs: opt(a, "--secs", 0.0f64),
        from: opt(a, "--from", 1usize).saturating_sub(1),
        kmin: opt(a, "--kmin", 1usize).clamp(1, K_MAX),
        pool: opt(a, "--pool", 26usize).clamp(K_MAX, 256),
        kmax: opt(a, "--kmax", d.k_max).clamp(1, K_MAX),
        fixed,
        out: opt(a, "--out", "m0d.bin".to_string()),
        init: opt(a, "--init", String::new()),
        lr_stage: opt(a, "--lrstage", 5e-4f64),
        warm: opt(a, "--warm", 300usize),
        rho: opt(a, "--rho", 1.2f64),
        trunc: a.iter().any(|x| x.as_str() == "--trunc"),
    };
    let (net, pr) = train::train(&cfg);
    println!("modèle : {} (plage : T ≤ {}, qmax ≤ {}, k ≤ {})\n", cfg.out, pr.t_hi, pr.q_hi, pr.k_max);
    let mut ev = pr.clone();
    ev.trunc = false; // l'évaluation tire toujours des mondes complets
    let v = eval::report(&net, &ecfg(a), &ev);
    std::process::exit(v.code());
}

fn cmd_eval(a: &[String]) {
    let path: String = opt(a, "--model", "m0d.bin".to_string());
    let net = load_or_exit(&path);
    let pr = ref_prior(a, &path);
    let v = eval::report(&net, &ecfg(a), &pr);
    std::process::exit(v.code());
}

fn cmd_diag(a: &[String]) {
    let path: String = opt(a, "--model", "m0d.bin".to_string());
    let net = load_or_exit(&path);
    let pr = ref_prior(a, &path);
    diag::run(
        &net,
        &pr,
        opt(a, "--n", 4000usize),
        opt(a, "--eseed", 9001u64),
        opt(a, "--horizon", 0usize),
    );
}

/// Contrôle de plage d'entraînement : A (plage de référence, lue dans <A>.ref) contre B (entraîné plus large).
/// --axis len (défaut) : B entraîné à T plus long ; --axis depth : B entraîné à qmax plus profond.
fn cmd_cmp(a: &[String]) {
    let pa: String = opt(a, "--a", "out/m1.bin".to_string());
    let pb: String = opt(a, "--b", "out/m1_t12.bin".to_string());
    let depth = opt(a, "--axis", "len".to_string()) == "depth";
    let rgate: f64 = opt(a, "--rgate", 0.30f64);
    let na = load_or_exit(&pa);
    let nb = load_or_exit(&pb);
    let pr = ref_prior(a, &pa);
    println!("A : {} (plage T ≤ {}, qmax ≤ {}, k ≤ {})", pa, pr.t_hi, pr.q_hi, pr.k_max);
    match train::load_ref(&pb) {
        Some((t, q, k)) => {
            println!("B : {} (plage T ≤ {}, qmax ≤ {}, k ≤ {})", pb, t, q, k);
            let wider = if depth { q > pr.q_hi } else { t > pr.t_hi };
            if !wider || k != pr.k_max {
                eprintln!(
                    "attention : B doit être plus large que A sur l'axe testé, à même k_max ; sinon le contrôle ne mesure rien"
                );
            }
        }
        None => eprintln!("attention : {}.ref absent, plage de B inconnue", pb),
    }
    let v = eval::compare(&na, &nb, &pr, &ecfg(a), depth, rgate);
    std::process::exit(v.code());
}

/// Taille de file contre erreurs de D (lecture seule). La file d'entraînement vient de <modèle>.ref
/// (--qtrain pour forcer) ; par défaut T de l'entraînement et file jusqu'à 4× celle de l'entraînement.
fn cmd_sizes(a: &[String]) {
    let path: String = opt(a, "--model", "out/m2.bin".to_string());
    let net = load_or_exit(&path);
    let d = Prior::demo();
    let (t, q, k) = train::load_ref(&path).unwrap_or((d.t_hi, d.q_hi, d.k_max));
    let mut pr = d;
    pr.t_hi = opt(a, "--thi", t);
    pr.k_max = opt(a, "--kmax", k).clamp(1, K_MAX);
    pr.q_hi = opt(a, "--qhi", 4 * q);
    sizes::run(&net, &pr, opt(a, "--qtrain", q), opt(a, "--n", 6000usize), opt(a, "--eseed", 9001u64));
}

fn wide_prior(a: &[String], path: &str) -> (Prior, usize) {
    let d = Prior::demo();
    let (t, q, k) = train::load_ref(path).unwrap_or((d.t_hi, d.q_hi, d.k_max));
    let mut pr = d;
    pr.t_hi = opt(a, "--thi", t.max(24));
    pr.k_max = opt(a, "--kmax", k).clamp(1, K_MAX);
    pr.q_hi = opt(a, "--qhi", 4 * q);
    pr.pool = opt(a, "--pool", pr.pool).clamp(K_MAX, 256);
    (pr, opt(a, "--qtrain", q))
}
fn cmd_equiv(a: &[String]) {
    let path: String = opt(a, "--model", "out/t4.bin".to_string());
    let net = load_or_exit(&path);
    let pr = ref_prior(a, &path);
    probe::equiv(&net, &pr, opt(a, "--n", 3000usize), opt(a, "--eseed", 9001u64));
}
fn cmd_hazard(a: &[String]) {
    let path: String = opt(a, "--model", "out/t4.bin".to_string());
    let net = load_or_exit(&path);
    let (pr, q) = wide_prior(a, &path);
    probe::hazard(&net, &pr, q, opt(a, "--n", 20000usize), opt(a, "--eseed", 9001u64));
}
fn cmd_reads(a: &[String]) {
    let path: String = opt(a, "--model", "out/t4.bin".to_string());
    let mut net = load_or_exit(&path);
    let (pr, q) = wide_prior(a, &path);
    let taus = [1.0, 2.0, 4.0, 8.0, 16.0]; // la base (τ = 1) doit venir en premier
    probe::reads(&mut net, &pr, q, &taus, opt(a, "--n", 6000usize), opt(a, "--eseed", 9001u64));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |k: &str| args.windows(2).find(|w| w[0] == k).map(|w| w[1].clone());
    match flag("--split").as_deref() {
        Some("canon") => eval::set_split(true),
        Some("legacy") => eval::set_split(false),
        _ => {
            if let Some(p) = flag("--model").or(flag("--a")).or(flag("--init")) {
                if let Some(c) = train::load_split(&p) {
                    eval::set_split(c);
                }
            }
        }
    }
    match args.get(1).map(|s| s.as_str()) {
        Some("selftest") => selftest(),
        Some("monde") => monde(&args[2..]),
        Some("gradcheck") => {
            if !net::gradcheck(arg(&args, 2, 1u64)) {
                std::process::exit(1);
            }
        }
        Some("overfit") => net::overfit(arg(&args, 2, 1u64)),
        Some("check") => eval::selfcheck(),
        Some("train") => cmd_train(&args[2..]),
        Some("eval") => cmd_eval(&args[2..]),
        Some("diag") => cmd_diag(&args[2..]),
        Some("cmp") => cmd_cmp(&args[2..]),
        Some("sizes") => cmd_sizes(&args[2..]),
        Some("tgrad") => {
            if !tape::gradcheck(arg(&args, 2, 1u64)) {
                std::process::exit(1);
            }
        },
        Some("tcheck") => selftest_trunc(),
        Some("pcheck") => probe::selfcheck(),
        Some("equiv") => cmd_equiv(&args[2..]),
        Some("hazard") => cmd_hazard(&args[2..]),
        Some("reads") => cmd_reads(&args[2..]),
        _ => eprintln!(
            "usage : selftest | monde [seed] [n] | gradcheck [seed] | overfit [seed] | check | train [--opts] | eval [--opts] | diag [--opts] | cmp [--opts] | sizes [--opts]"
        ),
    }
}