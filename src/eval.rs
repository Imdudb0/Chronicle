// Protocole d'évaluation de D : split des programmes, fidélité, courbes 1×/2×/4×, verdict.
//
// • Split : un programme C est « non vu » ssi hash(C) mod 8 = 0. L'entraînement ne tire jamais ces
//   programmes (ni en monde propre, ni en voisin muté) ; l'évaluation ne tire qu'eux.
//   « Non vu » = littéralement (C brut), pas modulo renommage : la clôture par quotient est le sujet de M1.
// • fid = régénération exacte sous la vue (octets des ticks observés) ET trace exacte (règle à chaque tick).
// • Axe longueur : mêmes programmes, mêmes vues, T = m·T1. Mondes bornés (qmax ≤ q_hi sur tout
//   l'horizon) : la profondeur reste dans la plage d'entraînement, seul le temps change.
// • Axe profondeur : prior étendu, qmax ∈ [lo, m·q_hi], T dans la plage d'entraînement.
#![allow(dead_code)]

use crate::meta::{emit_of, mutate_sample, run, sample_world, Prior, Program, Sample};
use crate::net::{Net, Stat, Work};
use crate::tape::{Net, Stat, Work};
use crate::rng::Rng;
use std::time::Instant;

// ---------- split des programmes ----------

pub const HELD_MOD: u64 = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Part {
    Train,
    Held,
}

/// FNV-1a sur la représentation du programme, puis mélange final (les bits de poids faible servent au split).
pub fn prog_hash(p: &Program) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut put = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    put(p.m as u64);
    put(0x100 + p.k() as u64);
    for &b in &p.bytes {
        put(b as u64);
    }
    for w in &p.prod {
        put(0x200);
        for &x in w {
            put(x as u64);
        }
    }
    put(0x300);
    for &x in &p.q0 {
        put(x as u64);
    }
    h ^= h >> 32;
    h = h.wrapping_mul(0xd6e8_feb8_6659_fd93);
    h ^ (h >> 32)
}

pub fn in_part(p: &Program, part: Part) -> bool {
    (prog_hash(p) % HELD_MOD == 0) == (part == Part::Held)
}

/// Monde du prior `pr` dont le programme est dans `part`. None si le prior est trop étroit (max_att atteint).
pub fn draw_world(rng: &mut Rng, pr: &Prior, part: Part, max_att: u32) -> Option<Sample> {
    for _ in 0..512 {
        let s = sample_world(rng, pr, max_att)?;
        if in_part(&s.prog, part) {
            return Some(s);
        }
    }
    None
}

/// Même monde (C, V), autre horizon t. None si la file s'épuise avant t.
/// Le crible de vue reproduit celui de meta::finish (positions seulement, jamais de valeurs).
pub fn rehorizon(s: &Sample, t: usize) -> Option<Sample> {
    let r = run(&s.prog, t)?;
    let emit = emit_of(&s.prog, &r.trace);
    let (o, step) = (s.view.o, s.view.s);
    let flag: Vec<bool> = (0..t).map(|i| i >= o && (i - o) % step == 0).collect();
    Some(Sample {
        prog: s.prog.clone(),
        view: s.view,
        t,
        trace: r.trace,
        emit,
        flag,
        qmax: r.qmax,
        bits_p: s.bits_p,
        bits_v: s.bits_v,
        bits_t: s.bits_t,
        attempts: s.attempts,
    })
}

/// Un monde est trivial si E est constant : rien à mémoriser, D peut le réussir à tout horizon.
/// Toutes les mesures (validation, courbes, verdict) l'excluent ; l'entraînement, lui, le garde.
pub fn nontrivial(s: &Sample) -> bool {
    s.emit.iter().any(|&b| b != s.emit[0])
}

// ---------- agrégats ----------

#[derive(Clone, Debug, Default)]
pub struct Agg {
    pub n: usize,
    pub loss: f64,
    pub n_flag: usize,
    pub flag_ok: usize,
    pub n_tick: usize,
    pub trace_ok: usize,
    pub regen: usize,
    pub trace_all: usize,
    pub both: usize,
    pub sum_t: usize,
    pub sum_q: usize,
    pub sum_k: usize,
    pub diverse: usize,
}

fn rate(a: usize, b: usize) -> f64 {
    a as f64 / b.max(1) as f64
}

impl Agg {
    pub fn add(&mut self, s: &Sample, st: &Stat) {
        self.n += 1;
        self.loss += st.loss;
        self.n_flag += st.n_flag;
        self.flag_ok += st.flag_ok;
        self.n_tick += st.t;
        self.trace_ok += st.trace_ok;
        self.regen += st.regen as usize;
        self.trace_all += st.trace_all as usize;
        self.both += (st.regen && st.trace_all) as usize;
        self.sum_t += s.t;
        self.sum_q += s.qmax;
        self.sum_k += s.prog.k();
        self.diverse += s.emit.iter().any(|&b| b != s.emit[0]) as usize;
    }

    pub fn merge(&mut self, o: &Agg) {
        self.n += o.n;
        self.loss += o.loss;
        self.n_flag += o.n_flag;
        self.flag_ok += o.flag_ok;
        self.n_tick += o.n_tick;
        self.trace_ok += o.trace_ok;
        self.regen += o.regen;
        self.trace_all += o.trace_all;
        self.both += o.both;
        self.sum_t += o.sum_t;
        self.sum_q += o.sum_q;
        self.sum_k += o.sum_k;
        self.diverse += o.diverse;
    }

    pub fn fid(&self) -> f64 {
        rate(self.both, self.n)
    }
    pub fn regen_rate(&self) -> f64 {
        rate(self.regen, self.n)
    }
    pub fn trace_rate(&self) -> f64 {
        rate(self.trace_all, self.n)
    }
    pub fn byte_acc(&self) -> f64 {
        rate(self.flag_ok, self.n_flag)
    }
    pub fn rule_acc(&self) -> f64 {
        rate(self.trace_ok, self.n_tick)
    }
    pub fn div_rate(&self) -> f64 {
        rate(self.diverse, self.n)
    }
    pub fn mean_loss(&self) -> f64 {
        self.loss / self.n.max(1) as f64
    }
    pub fn mean_t(&self) -> f64 {
        rate(self.sum_t, self.n)
    }
    pub fn mean_q(&self) -> f64 {
        rate(self.sum_q, self.n)
    }
    pub fn mean_k(&self) -> f64 {
        rate(self.sum_k, self.n)
    }
    /// Demi-largeur de l'intervalle de confiance à 95 % (loi normale) sur fid.
    pub fn ci(&self) -> f64 {
        let p = self.fid();
        1.96 * (p * (1.0 - p) / self.n.max(1) as f64).sqrt()
    }
}

pub fn validate(net: &Net, pr: &Prior, n: usize, rng: &mut Rng, ws: &mut Work) -> Agg {
    let mut a = Agg::default();
    let mut guard = 0usize;
    while a.n < n && guard < 100 * n.max(1) {
        guard += 1;
        if let Some(s) = draw_world(rng, pr, Part::Held, 1_000_000) {
            if !nontrivial(&s) {
                continue;
            }
            let st = net.run_sample(&s, 1.0, ws, None);
            a.add(&s, &st);
        }
    }
    a
}

// ---------- courbes ----------

pub struct ECfg {
    pub n: usize,    // mondes par point
    pub secs: f64,   // plafond de temps par axe (longueur) ou par point (profondeur)
    pub seed: u64,
    pub gate: f64,   // fid(1×) minimale pour que l'extrapolation soit interprétable
}

/// Axe longueur : T = m·T1 (T1 = t_hi d'entraînement), mêmes programmes et vues pour tous les m.
pub fn axis_len(net: &Net, pr: &Prior, mults: &[usize], ec: &ECfg, ws: &mut Work) -> Vec<Agg> {
    let t1 = pr.t_hi;
    let mut p1 = pr.clone();
    p1.t_lo = t1;
    p1.t_hi = t1;
    let mut rng = Rng::new(ec.seed ^ 0x1E17);
    let mut out = vec![Agg::default(); mults.len()];
    let clock = Instant::now();
    let mut guard = 0usize;
    while out[0].n < ec.n && guard < 400 * ec.n.max(1) && clock.elapsed().as_secs_f64() < ec.secs {
        guard += 1;
        let s = match draw_world(&mut rng, &p1, Part::Held, 100_000) {
            Some(s) => s,
            None => continue,
        };
        if !nontrivial(&s) {
            continue; // monde trivial (E constant à 1×) : exclu de toutes les mesures
        }
        // le même monde à tous les horizons ; rejet si la file s'épuise ou si qmax sort de la plage
        let mut at: Vec<Sample> = Vec::with_capacity(mults.len());
        for &m in mults {
            match rehorizon(&s, m * t1) {
                Some(x) if x.qmax <= pr.q_hi => at.push(x),
                _ => break,
            }
        }
        if at.len() < mults.len() {
            continue;
        }
        for (i, x) in at.iter().enumerate() {
            let st = net.run_sample(x, 1.0, ws, None);
            out[i].add(x, &st);
        }
    }
    out
}

/// Axe profondeur : bins de qmax [q_lo, q_hi], (q_hi, 2q_hi], (2q_hi, 4q_hi]. Renvoie (lo, hi, agrégat).
pub fn axis_depth(net: &Net, pr: &Prior, mults: &[usize], ec: &ECfg, ws: &mut Work) -> Vec<(usize, usize, Agg)> {
    let mut out = Vec::new();
    let mut lo = pr.q_lo;
    for (bi, &m) in mults.iter().enumerate() {
        let hi = m * pr.q_hi;
        let mut pb = pr.clone();
        pb.q_lo = lo;
        pb.q_hi = hi;
        let mut rng = Rng::new(ec.seed ^ (0xDE97 + bi as u64));
        let mut a = Agg::default();
        let clock = Instant::now();
        let mut guard = 0usize;
        while a.n < ec.n && guard < 400 * ec.n.max(1) && clock.elapsed().as_secs_f64() < ec.secs {
            guard += 1;
            if let Some(s) = draw_world(&mut rng, &pb, Part::Held, 100_000) {
                if !nontrivial(&s) {
                    continue;
                }
                let st = net.run_sample(&s, 1.0, ws, None);
                a.add(&s, &st);
            }
        }
        out.push((lo, hi, a));
        lo = hi + 1;
    }
    out
}

/// Totalité : voisins mutés de programmes non vus (le voisin est lui aussi non vu).
pub fn totality(net: &Net, pr: &Prior, ec: &ECfg, ws: &mut Work) -> Agg {
    let mut rng = Rng::new(ec.seed ^ 0xC0FFEE);
    let mut a = Agg::default();
    let clock = Instant::now();
    let mut guard = 0usize;
    while a.n < ec.n && guard < 200 * ec.n.max(1) && clock.elapsed().as_secs_f64() < ec.secs {
        guard += 1;
        if let Some(par) = draw_world(&mut rng, pr, Part::Held, 100_000) {
            if let Some(c) = mutate_sample(&mut rng, &par, pr, 200) {
                if in_part(&c.prog, Part::Held) && nontrivial(&c) {
                    let st = net.run_sample(&c, 1.0, ws, None);
                    a.add(&c, &st);
                }
            }
        }
    }
    a
}

// ---------- verdict ----------

pub const MIN_N: usize = 100;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    Pass,
    Gray,
    Kill,
    Inconclusive,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Gray => "GRIS",
            Verdict::Kill => "KILL",
            Verdict::Inconclusive => "INCONCLUSIF",
        }
    }
    pub fn code(self) -> i32 {
        match self {
            Verdict::Pass => 0,
            Verdict::Gray => 1,
            Verdict::Kill => 2,
            Verdict::Inconclusive => 3,
        }
    }
    pub fn advice(self) -> &'static str {
        match self {
            Verdict::Pass => "D tient à 2× : « simulateur appris, extrapolable » survit à M0. Passer à M1 (garder la courbe 4× comme référence).",
            Verdict::Gray => "Dégradation nette sans effondrement : lire le 4× et les colonnes oct/tk et règle/tk (dérive lente ou chute brutale) avant de décider.",
            Verdict::Kill => "D s'effondre à 2× sans correctif architectural : le banc est local et le transfert meurt ici. Corriger le substrat (M0 bis) avant tout M1.",
            Verdict::Inconclusive => "fid(1×) sous le seuil ou échantillon insuffisant : D n'a pas maîtrisé sa propre plage d'entraînement, l'extrapolation ne se juge pas. Prolonger n'aide pas s'il y a plateau : lancer `diag`, puis `train` v2 (cycles) ou changer le substrat.",
        }
    }
}

pub fn judge(a1: &Agg, a2: &Agg, gate: f64) -> Verdict {
    let (f1, f2) = (a1.fid(), a2.fid());
    if a1.n < MIN_N || a2.n < MIN_N || f1 < gate {
        Verdict::Inconclusive
    } else if f2 >= 0.8 * f1 {
        Verdict::Pass
    } else if f2 < 0.5 * f1 {
        Verdict::Kill
    } else {
        Verdict::Gray
    }
}

pub fn combine(a: Verdict, b: Verdict) -> Verdict {
    use Verdict::*;
    if a == Kill || b == Kill {
        Kill
    } else if a == Inconclusive || b == Inconclusive {
        Inconclusive
    } else if a == Gray || b == Gray {
        Gray
    } else {
        Pass
    }
}

// ---------- rapport ----------

fn head() {
    println!(
        "{:<15}{:>6}{:>9}{:>9}{:>9}{:>6}{:>10}{:>10}{:>7}{:>7}{:>6}{:>6}",
        "point", "n", "régén%", "trace%", "fid%", "±", "oct/tk%", "règle/tk%", "T", "qmax", "k", "div"
    );
}

fn row(label: &str, a: &Agg) {
    println!(
        "{:<15}{:>6}{:>9.1}{:>9.1}{:>9.1}{:>6.1}{:>10.2}{:>10.2}{:>7.1}{:>7.1}{:>6.2}{:>6.2}",
        label,
        a.n,
        100.0 * a.regen_rate(),
        100.0 * a.trace_rate(),
        100.0 * a.fid(),
        100.0 * a.ci(),
        100.0 * a.byte_acc(),
        100.0 * a.rule_acc(),
        a.mean_t(),
        a.mean_q(),
        a.mean_k(),
        a.div_rate()
    );
}

pub fn report(net: &Net, ec: &ECfg, pr: &Prior) -> Verdict {
    let mults = [1usize, 2, 4];
    let mut ws = Work::new();
    println!(
        "=== M0 : fidélité de D sur programmes non vus (hash(C) mod {} = 0), mondes non triviaux (E non constant) ===",
        HELD_MOD
    );
    println!(
        "référence (1×) : T ∈ [{}, {}], qmax ≤ {}, k ≤ {} | fid = régénération exacte ∧ trace exacte | ± = IC 95 %",
        pr.t_lo, pr.t_hi, pr.q_hi, pr.k_max
    );

    let al = axis_len(net, pr, &mults, ec, &mut ws);
    println!("\nAXE LONGUEUR : mêmes programmes et mêmes vues, qmax ≤ {} jusqu'à {}×", pr.q_hi, mults[2]);
    head();
    for (i, a) in al.iter().enumerate() {
        row(&format!("{}×  T={}", mults[i], mults[i] * pr.t_hi), a);
    }

    let ad = axis_depth(net, pr, &mults, ec, &mut ws);
    println!("\nAXE PROFONDEUR : même plage de T [{}, {}], bins de qmax ×1 / ×2 / ×4 de {}", pr.t_lo, pr.t_hi, pr.q_hi);
    head();
    for (i, b) in ad.iter().enumerate() {
        row(&format!("{}×  q∈[{},{}]", mults[i], b.0, b.1), &b.2);
    }

    let to = totality(net, pr, ec, &mut ws);
    println!("\nTOTALITÉ : voisins mutés (do local) de programmes non vus, même prior");
    head();
    row("mutés", &to);

    let vl = judge(&al[0], &al[1], ec.gate);
    let vd = judge(&ad[0].2, &ad[1].2, ec.gate);
    let v = combine(vl, vd);
    println!("\n=== VERDICT M0 ===");
    println!(
        "règle : fid(1×) ≥ {:.2} requis ; PASS si fid(2×) ≥ 0,8·fid(1×) ; KILL si fid(2×) < 0,5·fid(1×) ; entre les deux : GRIS ; n ≥ {} par point",
        ec.gate, MIN_N
    );
    let show = |name: &str, f: [f64; 3], vv: Verdict| {
        println!(
            "{:<11} fid 1× {:>5.1}%   2× {:>5.1}%   4× {:>5.1}%   (2×/1× = {:.2}, 4×/1× = {:.2})  →  {}",
            name,
            100.0 * f[0],
            100.0 * f[1],
            100.0 * f[2],
            f[1] / f[0].max(1e-9),
            f[2] / f[0].max(1e-9),
            vv.name()
        );
    };
    show("longueur", [al[0].fid(), al[1].fid(), al[2].fid()], vl);
    show("profondeur", [ad[0].2.fid(), ad[1].2.fid(), ad[2].2.fid()], vd);
    println!("GLOBAL : {}", v.name());
    println!(
        "sous réserve de la porte (fid 1× ignorée ; à lire avec les ± ci-dessus) : longueur {} ; profondeur {}",
        judge(&al[0], &al[1], 0.0).name(),
        judge(&ad[0].2, &ad[1].2, 0.0).name()
    );
    println!("{}", v.advice());
    v
}

/// Contrôle de plage d'entraînement. A = modèle entraîné sur la plage de référence `pr` ; B = modèle entraîné
/// plus large (plus long en T pour l'axe longueur, plus profond en qmax pour l'axe profondeur).
/// Mêmes mondes non vus, mêmes vues (mêmes graines). Rétention au point 2× = fid(A) / fid(B) : ce que A conserve
/// de ce qu'un modèle de même taille, entraîné jusqu'à ce point, sait faire. La difficulté est soustraite.
pub fn compare(a: &Net, b: &Net, pr: &Prior, ec: &ECfg, depth: bool, rgate: f64) -> Verdict {
    let mults = [1usize, 2, 4];
    let mut ws = Work::new();
    let mut rows: Vec<(String, Agg, Agg)> = Vec::new();
    if depth {
        let da = axis_depth(a, pr, &mults, ec, &mut ws);
        let db = axis_depth(b, pr, &mults, ec, &mut ws);
        for (i, (x, y)) in da.into_iter().zip(db.into_iter()).enumerate() {
            rows.push((format!("{}×  q∈[{},{}]", mults[i], x.0, x.1), x.2, y.2));
        }
    } else {
        let la = axis_len(a, pr, &mults, ec, &mut ws);
        let lb = axis_len(b, pr, &mults, ec, &mut ws);
        for (i, (x, y)) in la.into_iter().zip(lb.into_iter()).enumerate() {
            rows.push((format!("{}×  T={}", mults[i], mults[i] * pr.t_hi), x, y));
        }
    }
    println!("=== M0 / contrôle de plage d'entraînement : A (plage de référence) contre B (entraîné plus large) ===");
    println!(
        "référence : T ∈ [{}, {}], qmax ≤ {}, k ≤ {} | axe {} | mêmes mondes non vus pour A et B | A/B = rétention de A",
        pr.t_lo,
        pr.t_hi,
        pr.q_hi,
        pr.k_max,
        if depth { "profondeur" } else { "longueur" }
    );
    println!(
        "{:<15}{:>6}{:>9}{:>6}{:>9}{:>6}{:>9}{:>8}",
        "point", "n", "A fid%", "±", "B fid%", "±", "B−A", "A/B"
    );
    for (label, x, y) in &rows {
        println!(
            "{:<15}{:>6}{:>9.1}{:>6.1}{:>9.1}{:>6.1}{:>9.1}{:>8.2}",
            label,
            x.n.min(y.n),
            100.0 * x.fid(),
            100.0 * x.ci(),
            100.0 * y.fid(),
            100.0 * y.ci(),
            100.0 * (y.fid() - x.fid()),
            x.fid() / y.fid().max(1e-9)
        );
    }
    let (_, x, y) = &rows[1];
    let ratio = x.fid() / y.fid().max(1e-9);
    let v = if x.n < MIN_N || y.n < MIN_N || y.fid() < rgate {
        Verdict::Inconclusive
    } else if ratio >= 0.8 {
        Verdict::Pass
    } else if ratio < 0.5 {
        Verdict::Kill
    } else {
        Verdict::Gray
    };
    println!(
        "\nrègle (fixée avant mesure) : rétention A/B au point 2× ≥ 0,80 → PASS ; < 0,50 → KILL ; entre les deux → GRIS ; INCONCLUSIF si fid(B) au point 2× < {:.2} ou n < {}",
        rgate, MIN_N
    );
    println!(
        "point 2× : A {:.1} %   B {:.1} %   A/B = {:.2}   →   {}",
        100.0 * x.fid(),
        100.0 * y.fid(),
        ratio,
        v.name()
    );
    if depth {
        println!("réserve : B répartit sa capacité sur plus de programmes que A, donc A/B surestime la rétention de A (lire un PASS avec prudence).");
    } else {
        println!("lecture : mêmes programmes, B voit seulement des ticks de plus ; si A/B ≈ 1, la chute de A à 2× est celle qu'un modèle entraîné à 2× subit aussi (capacité, pas plage).");
    }
    v
}

// ---------- smoke test (sans entraînement) ----------

pub fn selfcheck() {
    let pr = Prior::demo();
    let mut rng = Rng::new(5);
    // 1. split : disjoint
    let mut held_mass = 0usize;
    for _ in 0..3000 {
        let a = draw_world(&mut rng, &pr, Part::Train, 100_000).unwrap();
        let b = draw_world(&mut rng, &pr, Part::Held, 100_000).unwrap();
        assert!(in_part(&a.prog, Part::Train) && !in_part(&a.prog, Part::Held));
        assert!(in_part(&b.prog, Part::Held) && !in_part(&b.prog, Part::Train));
        let s = sample_world(&mut rng, &pr, 100_000).unwrap();
        held_mass += (prog_hash(&s.prog) % HELD_MOD == 0) as usize;
    }
    println!(
        "ok 1 : split disjoint ; masse « non vue » sous le prior = {:.3} (1/{} des programmes, pas forcément des tirages)",
        held_mass as f64 / 3000.0,
        HELD_MOD
    );
    // 2. rehorizon : identité à T égal, préfixe à 3T
    let mut ext = 0usize;
    for _ in 0..500 {
        let s = sample_world(&mut rng, &pr, 100_000).unwrap();
        let r = rehorizon(&s, s.t).unwrap();
        assert_eq!(r.trace, s.trace);
        assert_eq!(r.emit, s.emit);
        assert_eq!(r.flag, s.flag);
        assert_eq!(r.qmax, s.qmax);
        if let Some(l) = rehorizon(&s, 3 * s.t) {
            assert_eq!(&l.trace[..s.t], &s.trace[..]);
            assert_eq!(&l.emit[..s.t], &s.emit[..]);
            assert_eq!(&l.flag[..s.t], &s.flag[..]);
            ext += 1;
        }
    }
    println!("ok 2 : rehorizon (identité à T égal ; préfixe à 3T pour {}/500 mondes)", ext);
    // 3. étages
    let sts = crate::train::stages(&pr);
    for w in sts.windows(2) {
        assert!(w[1].t_hi >= w[0].t_hi && w[1].q_hi >= w[0].q_hi);
    }
    let last = &sts[sts.len() - 1];
    assert_eq!((last.t_hi, last.q_hi), (pr.t_hi, pr.q_hi));
    println!(
        "ok 3 : {} étages, de (T≤{}, q≤{}) à (T≤{}, q≤{})",
        sts.len(),
        sts[0].t_hi,
        sts[0].q_hi,
        last.t_hi,
        last.q_hi
    );
    // 4. remplissage des points de courbe sur un réseau non entraîné
    let net = Net::new(&mut Rng::new(3));
    let ec = ECfg { n: 50, secs: 15.0, seed: 11, gate: 0.95 };
    let mut ws = Work::new();
    let t0 = Instant::now();
    let al = axis_len(&net, &pr, &[1, 2, 4], &ec, &mut ws);
    println!("ok 4a : axe longueur, {} mondes obtenus (demandé {}) en {:.1} s", al[0].n, ec.n, t0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let ad = axis_depth(&net, &pr, &[1, 2, 4], &ec, &mut ws);
    for b in &ad {
        println!("ok 4b : profondeur q ∈ [{},{}] : {} mondes obtenus (demandé {}), k moyen {:.2}, div {:.2}", b.0, b.1, b.2.n, ec.n, b.2.mean_k(), b.2.div_rate());
    }
    println!("      ({:.1} s)", t0.elapsed().as_secs_f64());
    println!("CHECK OK");
}