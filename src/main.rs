mod meta;
mod rng;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("selftest") => selftest(),
        Some("monde") => monde(&args[2..]),
        _ => eprintln!("usage : m0 selftest | m0 monde [seed] [n]"),
    }
}