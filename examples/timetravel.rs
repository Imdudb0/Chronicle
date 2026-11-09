//! `cargo run --release --example timetravel`

use std::time::Instant;

use chronicle::kv::{Kv, KvEvent};
use chronicle::{Engine, Options, Result};

fn main() -> Result<()> {
    let _ = std::fs::remove_file("tt.chr");
    let _ = std::fs::remove_dir_all("tt.chr.snapshots");
    let opts = Options { snapshot_every: 20_000, ..Options::default() };

    let (mut db, _) = Engine::<Kv>::open_with("tt.chr", opts.clone())?;
    for i in 0..100_000u32 {
        let key = format!("compteur{}", i % 3).into_bytes();
        db.submit(&KvEvent::Put { key, value: i.to_string().into_bytes() })?;
        if i % 5_000 == 4_999 {
            db.commit()?;
        }
    }
    db.commit()?;
    println!("{} cycles ; snapshots : {:?}", db.last_seq(), db.snapshots());
    drop(db);

    let t = Instant::now();
    let (db, _) = Engine::<Kv>::open_with("tt.chr", opts)?;
    println!("redémarrage depuis le cycle {} en {:?}", db.boot_seq(), t.elapsed());

    // Pas à pas vers le passé.
    let mut cur = db.cursor()?;
    for _ in 0..3 {
        let undone = cur.step_back()?;
        println!("annulé {undone:?} → cycle {}", cur.seq());
    }

    // Saut direct dans le passé.
    cur.goto(1_234)?;
    let v = cur.state().get(b"compteur0").map(String::from_utf8_lossy);
    println!("au cycle 1234 : compteur0 = {v:?}");

    // Point d'arrêt depuis le début.
    cur.goto(0)?;
    let hit = cur.run_until(|s, _| s.get(b"compteur0") == Some(&b"30000"[..]))?;
    println!("compteur0 = 30000 pour la première fois au cycle {hit:?}");
    Ok(())
}