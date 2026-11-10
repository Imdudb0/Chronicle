//! Tests d'intégration de la boucle d'événements (Linux uniquement).
//!
//! Aucun `sleep`, aucune mesure de temps : le comportement est prouvé en comptant
//! (msync, cycles, réponses). Chaque test a un chien de garde : une boucle qui
//! pend (réveil perdu) fait échouer le test au lieu de bloquer `cargo test`.
#![cfg(target_os = "linux")]

use std::io::Write;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use chronicle::kv::{Kv, KvEvent};
use chronicle::{block_on, Engine, Error, Handle, Options, Server, StateMachine};
use serde::{Deserialize, Serialize};

fn put(k: &str, v: &str) -> KvEvent {
    KvEvent::Put { key: k.as_bytes().to_vec(), value: v.as_bytes().to_vec() }
}

/// Sans snapshot périodique : ces tests comptent les msync, pas les snapshots.
fn quiet() -> Options {
    Options { snapshot_every: 0, ..Options::default() }
}

fn start(path: &Path, opts: Options) -> Server<Kv> {
    Server::<Kv>::start(path, opts).unwrap().0
}

/// À garder vivant jusqu'à la fin du test : `let _wd = watchdog();`
struct Watchdog {
    _cancel: mpsc::Sender<()>,
}

fn watchdog() -> Watchdog {
    const LIMIT: Duration = Duration::from_secs(30);
    let (tx, rx) = mpsc::channel::<()>();
    thread::spawn(move || {
        // `Disconnected` : le test est terminé ; `Timeout` : il pend.
        if rx.recv_timeout(LIMIT) == Err(mpsc::RecvTimeoutError::Timeout) {
            // Sur le vrai stderr : `eprintln!` serait capturé, puis perdu par `exit`.
            let _ = writeln!(std::io::stderr(), "\ntest bloqué depuis plus de {LIMIT:?} (réveil perdu ?) : abandon");
            std::process::exit(101);
        }
    });
    Watchdog { _cancel: tx }
}

/// État de test : `!Send` (il ne doit jamais quitter le thread de la boucle) et
/// dont `apply` viole exprès le contrat « infaillible » sur l'événement 13.
#[derive(Default, Serialize, Deserialize)]
struct Counter {
    n: u64,
    _pin: PhantomData<*const ()>,
}

impl StateMachine for Counter {
    type Event = u64;

    fn genesis() -> Self {
        Self::default()
    }

    fn apply(&mut self, by: &u64) {
        assert_ne!(*by, 13, "événement empoisonné");
        self.n += by;
    }
}

fn assert_send_sync<T: Send + Sync + 'static>() {}

/// Preuve du group commit sans chronomètre : on bloque la boucle dans une lecture,
/// 200 écritures s'empilent dans la file, on la libère : un seul lot, un seul msync.
#[test]
fn queued_writes_share_a_single_msync() {
    const N: u64 = 200;
    let _wd = watchdog();
    let dir = tempfile::tempdir().unwrap();
    let server = start(&dir.path().join("kv.log"), quiet());
    let h = server.handle();

    let (entered_tx, entered_rx) = mpsc::channel();
    let (gate_tx, gate_rx) = mpsc::channel::<()>();
    let gate = h.read(move |_| {
        let _ = entered_tx.send(());
        let _ = gate_rx.recv(); // la boucle dort ici jusqu'à la levée de la barrière
    });
    entered_rx.recv().unwrap(); // la boucle est maintenant dans la lecture

    let writes: Vec<_> = (0..N).map(|i| h.submit(&put(&format!("k{i}"), "v"))).collect();
    assert_eq!(h.stats().syncs, 0); // la boucle dort : rien n'a pu être validé

    gate_tx.send(()).unwrap();
    gate.wait().unwrap();
    for (i, write) in writes.into_iter().enumerate() {
        assert_eq!(write.wait().unwrap(), i as u64 + 1);
    }
    let stats = h.stats();
    assert_eq!(stats.syncs, 1, "{N} écritures, un seul msync");
    assert_eq!(stats.durable_seq, N);
    server.shutdown().unwrap();
}

#[test]
fn reads_follow_writes_and_are_answered_only_once_durable() {
    let _wd = watchdog();
    let dir = tempfile::tempdir().unwrap();
    let server = start(&dir.path().join("kv.log"), quiet());
    let h = server.handle();

    // Déposées dans cet ordre par un même thread : la lecture voit l'écriture.
    let write = h.submit(&put("a", "1"));
    let read = h.read(|kv| kv.get(b"a").map(<[u8]>::to_vec));
    assert_eq!(read.wait().unwrap(), Some(b"1".to_vec()));
    // Réponse rendue ⇒ le cycle qu'elle reflète est déjà durable.
    assert_eq!(h.stats().durable_seq, 1);
    assert_eq!(write.wait().unwrap(), 1);

    // Une lecture seule n'écrit rien : pas de msync de plus.
    let syncs = h.stats().syncs;
    assert_eq!(h.read(|kv| kv.len()).wait().unwrap(), 1);
    assert_eq!(h.stats().syncs, syncs);
    server.shutdown().unwrap();
}

#[test]
fn concurrent_clients_get_unique_consecutive_cycles() {
    const CLIENTS: u64 = 8;
    const EACH: u64 = 50;
    let _wd = watchdog();
    let dir = tempfile::tempdir().unwrap();
    let server = start(&dir.path().join("kv.log"), quiet());
    let h = server.handle();

    let workers: Vec<_> = (0..CLIENTS)
        .map(|c| {
            let h = h.clone();
            thread::spawn(move || {
                (0..EACH)
                    .map(|i| h.submit(&put(&format!("c{c}-{i}"), "v")).wait().unwrap())
                    .collect::<Vec<u64>>()
            })
        })
        .collect();

    let mut all = Vec::new();
    for worker in workers {
        let seqs = worker.join().unwrap();
        assert!(seqs.windows(2).all(|w| w[0] < w[1]), "cycles croissants pour un même client");
        all.extend(seqs);
    }
    all.sort_unstable();
    assert_eq!(all, (1..=CLIENTS * EACH).collect::<Vec<u64>>()); // ni trou, ni doublon
    assert_eq!(h.read(|kv| kv.len()).wait().unwrap() as u64, CLIENTS * EACH);
    assert_eq!(h.stats().durable_seq, CLIENTS * EACH);
    server.shutdown().unwrap();
}

#[test]
fn tasks_run_on_the_loop_and_await_durability() {
    let _wd = watchdog();
    let dir = tempfile::tempdir().unwrap();
    let server = start(&dir.path().join("kv.log"), quiet());
    let h = server.handle();

    let (done_tx, done_rx) = mpsc::channel();
    h.spawn(move |ctx| async move {
        let mut acks = Vec::new();
        for i in 0..5 {
            let seq = ctx.submit(&put(&format!("t{i}"), "x")).await.unwrap();
            // `await` ne rend la main qu'une fois le cycle durable.
            acks.push((seq, ctx.handle().stats().durable_seq));
        }
        let len = ctx.read(|kv| kv.len());
        let _ = done_tx.send((acks, len));
    })
    .unwrap();

    let (acks, len) = done_rx.recv().unwrap();
    assert_eq!(len, 5);
    for (i, (seq, durable)) in acks.into_iter().enumerate() {
        assert_eq!(seq, i as u64 + 1);
        assert!(durable >= seq, "cycle {seq} acquitté avant d'être durable");
    }
    // La tâche et les clients externes partagent la même boucle et la même séquence.
    assert_eq!(h.submit(&put("outside", "y")).wait().unwrap(), 6);
    server.shutdown().unwrap();
}

#[test]
fn read_at_replays_the_past_without_disturbing_the_present() {
    let _wd = watchdog();
    let dir = tempfile::tempdir().unwrap();
    let server = start(&dir.path().join("kv.log"), quiet());
    let h = server.handle();

    h.submit(&put("k", "v1")).wait().unwrap();
    h.submit(&put("k", "v2")).wait().unwrap();
    h.submit(&KvEvent::Delete { key: b"k".to_vec() }).wait().unwrap();

    let at = |seq| h.read_at(seq, |kv| kv.get(b"k").map(<[u8]>::to_vec)).wait();
    assert_eq!(at(0).unwrap(), None);
    assert_eq!(at(1).unwrap(), Some(b"v1".to_vec()));
    assert_eq!(at(2).unwrap(), Some(b"v2".to_vec()));
    assert_eq!(at(3).unwrap(), None);
    assert!(matches!(at(4), Err(Error::SeqOutOfRange { requested: 4, last: 3 })));

    // Le présent n'a pas bougé : le prochain cycle est bien le 4.
    assert_eq!(h.submit(&put("k", "v4")).wait().unwrap(), 4);
    assert_eq!(at(4).unwrap(), Some(b"v4".to_vec()));
    server.shutdown().unwrap();
}

#[test]
fn shutdown_answers_queued_commands_then_closes_the_handle() {
    let _wd = watchdog();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.log");
    let server = start(&path, quiet());
    let h = server.handle();

    let queued: Vec<_> = (0..10).map(|i| h.submit(&put(&format!("k{i}"), "v"))).collect();
    server.shutdown().unwrap(); // traite d'abord ce qui est déjà déposé
    for (i, pending) in queued.into_iter().enumerate() {
        assert_eq!(pending.wait().unwrap(), i as u64 + 1);
    }

    // La poignée survit au serveur, mais plus rien ne s'exécute.
    assert!(matches!(h.submit(&put("late", "x")).wait(), Err(Error::Closed)));
    assert!(matches!(h.read(|kv| kv.len()).wait(), Err(Error::Closed)));
    assert!(matches!(h.spawn(|_| async {}), Err(Error::Closed)));

    // Journal libéré (flock) et snapshot final écrit : redémarrage instantané.
    let (db, rec) = Engine::<Kv>::open(&path).unwrap();
    assert_eq!((rec.frames, db.last_seq(), db.boot_seq()), (10, 10, 10));
    assert_eq!(db.state().len(), 10);
}

#[test]
fn restart_recovers_acknowledged_writes_and_snapshots_run_on_the_loop() {
    let _wd = watchdog();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.log");
    let opts = Options { snapshot_every: 20, ..Options::default() };

    let (server, rec) = Server::<Kv>::start(&path, opts.clone()).unwrap();
    assert_eq!(rec.frames, 0);
    let h = server.handle();
    for i in 0..50 {
        h.submit(&put(&format!("k{}", i % 10), &i.to_string())).wait().unwrap();
    }
    server.shutdown().unwrap();

    // Snapshots écrits par la boucle : aux cycles 20 et 40, puis à l'arrêt.
    {
        let (db, _) = Engine::<Kv>::open(&path).unwrap();
        assert_eq!(db.snapshots().to_vec(), vec![20, 40, 50]);
    }

    let (server, rec) = Server::<Kv>::start(&path, opts).unwrap();
    assert_eq!(rec.frames, 50);
    let h = server.handle();
    let last = h.read(|kv| kv.get(b"k9").map(<[u8]>::to_vec)).wait().unwrap();
    assert_eq!(last, Some(b"49".to_vec()));
    assert_eq!(h.submit(&put("next", "1")).wait().unwrap(), 51);
    server.shutdown().unwrap();
}

#[test]
fn a_second_server_on_the_same_journal_is_refused_until_shutdown() {
    let _wd = watchdog();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.log");

    let (first, _) = Server::<Kv>::start(&path, quiet()).unwrap();
    assert!(matches!(Server::<Kv>::start(&path, quiet()), Err(Error::Locked)));
    first.shutdown().unwrap();
    assert!(Server::<Kv>::start(&path, quiet()).is_ok());
}

#[test]
fn block_on_composes_pending_futures() {
    let _wd = watchdog();
    let dir = tempfile::tempdir().unwrap();
    let server = start(&dir.path().join("kv.log"), quiet());
    let h = server.handle();

    let (a, b, len) = block_on(async {
        let a = h.submit(&put("a", "1")).await?;
        let b = h.submit(&put("b", "2")).await?;
        let len = h.read(|kv| kv.len()).await?;
        Ok::<_, Error>((a, b, len))
    })
    .unwrap();
    assert_eq!((a, b, len), (1, 2, 2));

    // Deux commandes en vol à la fois, attendues ensuite.
    let (c, d) = block_on(async {
        let (c, d) = (h.submit(&put("c", "3")), h.submit(&put("d", "4")));
        Ok::<_, Error>((c.await?, d.await?))
    })
    .unwrap();
    assert_eq!((c, d), (3, 4));
    server.shutdown().unwrap();
}

#[test]
fn the_state_never_has_to_be_send() {
    assert_send_sync::<Handle<Counter>>(); // vérifié à la compilation
    let _wd = watchdog();
    let dir = tempfile::tempdir().unwrap();
    let (server, _) = Server::<Counter>::start(dir.path().join("c.log"), quiet()).unwrap();
    let h = server.handle();
    assert_eq!(h.submit(&5).wait().unwrap(), 1);
    assert_eq!(h.submit(&7).wait().unwrap(), 2);
    assert_eq!(h.read(|c| c.n).wait().unwrap(), 12);
    server.shutdown().unwrap();
}

#[test]
fn a_panicking_apply_closes_the_loop_instead_of_hanging_clients() {
    let _wd = watchdog();
    let dir = tempfile::tempdir().unwrap();
    let (server, _) = Server::<Counter>::start(dir.path().join("c.log"), quiet()).unwrap();
    let h = server.handle();
    assert_eq!(h.submit(&1).wait().unwrap(), 1);

    // L'événement 13 fait paniquer `apply` sur la boucle : le client en attente est
    // libéré (Closed) et plus aucune commande n'est acceptée.
    assert!(matches!(h.submit(&13).wait(), Err(Error::Closed)));
    assert!(matches!(h.submit(&2).wait(), Err(Error::Closed)));

    // `shutdown` relaie la panique de la boucle.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| server.shutdown()));
    assert!(outcome.is_err());
}

/// `Engine::submit_encoded` (partie 3a) : un message mal formé n'atteint jamais le
/// journal, sinon il rendrait chaque rejeu impossible.
#[test]
fn malformed_events_never_reach_the_journal() {
    let dir = tempfile::tempdir().unwrap();
    let (mut db, _) = Engine::<Kv>::open(dir.path().join("kv.log")).unwrap();
    let good = postcard::to_allocvec(&put("a", "1")).unwrap();

    let mut padded = good.clone();
    padded.push(0);
    assert!(matches!(db.submit_encoded(&padded), Err(Error::TrailingBytes)));
    assert!(db.submit_encoded(&[0xFF, 0xFF, 0xFF]).is_err());
    assert_eq!(db.last_seq(), 0); // rien n'a été écrit

    assert_eq!(db.submit_encoded(&good).unwrap(), 1);
    assert_eq!(db.state().get(b"a"), Some(&b"1"[..]));
}