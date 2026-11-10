//! Group commit : un msync par événement, ou un msync par lot ?
//!
//! `cargo run --release --example group_commit [DOSSIER]`
//!
//! À lancer sur un vrai disque : sur tmpfs, msync ne coûte presque rien et les
//! trois mesures se ressemblent. Les journaux `gc-*.chr` sont créés dans DOSSIER
//! (défaut : le dossier courant).

use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use chronicle::kv::{Kv, KvEvent};
use chronicle::{Engine, Options, Result, Server};

const BASELINE: u64 = 1_000;
const CLIENTS: u64 = 16;
const PER_CLIENT: u64 = 1_000;

fn event(i: u64) -> KvEvent {
    KvEvent::Put { key: format!("k{}", i % 1_000).into_bytes(), value: i.to_string().into_bytes() }
}

/// Chemin d'un journal vierge (et sans snapshots) dans `dir`.
fn fresh(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(dir.join(format!("{name}.snapshots")));
    path
}

fn report(label: &str, count: u64, took: Duration, syncs: u64) {
    println!(
        "{label:<36}{:>10.0} événements/s   ({count} en {took:.2?}, {syncs} msync, {:.1} événements par msync)",
        count as f64 / took.as_secs_f64(),
        count as f64 / syncs.max(1) as f64,
    );
}

fn main() -> Result<()> {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".into()));
    // Pas de snapshot : on mesure le journal, pas la sérialisation de l'état.
    let opts = Options { snapshot_every: 0, ..Options::default() };
    let total = CLIENTS * PER_CLIENT;

    // 1. Référence : un thread, `commit` (donc msync) après chaque événement.
    let (mut db, _) = Engine::<Kv>::open_with(fresh(&dir, "gc-seq.chr"), opts.clone())?;
    let t = Instant::now();
    for i in 0..BASELINE {
        db.submit(&event(i))?;
        db.commit()?;
    }
    report("1 thread, msync par événement", BASELINE, t.elapsed(), BASELINE);
    drop(db);

    // 2. Serveur, CLIENTS threads : chacun attend la durabilité de chaque écriture.
    let (server, _) = Server::<Kv>::start(fresh(&dir, "gc-loop.chr"), opts.clone())?;
    let h = server.handle();
    let t = Instant::now();
    let workers: Vec<_> = (0..CLIENTS)
        .map(|c| {
            let h = h.clone();
            thread::spawn(move || -> Result<()> {
                for i in 0..PER_CLIENT {
                    h.submit(&event(c * PER_CLIENT + i)).wait()?;
                }
                Ok(())
            })
        })
        .collect();
    for worker in workers {
        worker.join().expect("un client a paniqué")?;
    }
    report(&format!("{CLIENTS} clients, group commit"), total, t.elapsed(), h.stats().syncs);
    server.shutdown()?;

    // 3. Serveur, un seul thread mais pipeliné : tout part d'abord, on attend ensuite.
    let (server, _) = Server::<Kv>::start(fresh(&dir, "gc-pipe.chr"), opts)?;
    let h = server.handle();
    let t = Instant::now();
    let acks: Vec<_> = (0..total).map(|i| h.submit(&event(i))).collect();
    for ack in acks {
        ack.wait()?;
    }
    report("1 thread, pipeliné", total, t.elapsed(), h.stats().syncs);
    server.shutdown()
}