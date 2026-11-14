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