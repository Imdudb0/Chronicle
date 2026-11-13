mod meta;
mod rng;
mod net;
mod eval;
mod train;
mod diag;

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

fn load_or_exit(path: &str) -> net::Net {
    match net::Net::load(path) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("{} : {}", path, e);
            std::process::exit(1);
        }
    }
}

/// Étage de référence (0-based) : --stage N (1-based), sinon <modèle>.stage, sinon étage final.
fn stage_of(a: &[String], path: &str) -> usize {
    let nst = train::stages(&Prior::demo()).len();
    let s: usize = opt(a, "--stage", 0usize);
    if s >= 1 {
        return (s - 1).min(nst - 1);
    }
    train::load_stage(path).unwrap_or(nst - 1).min(nst - 1)
}

fn cmd_train(a: &[String]) {
    let threads_def = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
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
        out: opt(a, "--out", "m0d.bin".to_string()),
        init: opt(a, "--init", String::new()),
    };
    let (net, stage) = train::train(&cfg);
    println!("modèle : {} (étage de référence {})\n", cfg.out, stage + 1);
    let v = eval::report(&net, &ecfg(a), stage);
    std::process::exit(v.code());
}

fn cmd_eval(a: &[String]) {
    let path: String = opt(a, "--model", "m0d.bin".to_string());
    let net = load_or_exit(&path);
    let stage = stage_of(a, &path);
    let v = eval::report(&net, &ecfg(a), stage);
    std::process::exit(v.code());
}

fn cmd_diag(a: &[String]) {
    let path: String = opt(a, "--model", "m0d.bin".to_string());
    let net = load_or_exit(&path);
    let sts = train::stages(&Prior::demo());
    let stage = stage_of(a, &path);
    diag::run(&net, &sts[stage], opt(a, "--n", 4000usize), opt(a, "--eseed", 9001u64), opt(a, "--horizon", 0usize));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
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
        _ => eprintln!(
            "usage : selftest | monde [seed] [n] | gradcheck [seed] | overfit [seed] | check | train [--opts] | eval [--opts] | diag [--opts]"
        ),
    }
}