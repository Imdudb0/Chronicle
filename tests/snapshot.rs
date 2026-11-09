use std::ops::Range;
use std::path::Path;

use chronicle::kv::{Kv, KvEvent};
use chronicle::{Engine, Options, StateMachine};

/// Événement déterministe n° `i` (mélange de Put et de Delete).
fn event(i: u32) -> KvEvent {
    let key = format!("k{}", i % 13).into_bytes();
    if i % 5 == 4 {
        KvEvent::Delete { key }
    } else {
        KvEvent::Put { key, value: format!("v{i}").into_bytes() }
    }
}

/// Snapshot automatique tous les `every` cycles, sans élagage.
fn snap(every: u64) -> Options {
    Options { snapshot_every: every, keep_recent: 1000, milestone_every: 0, snapshot_dir: None }
}

/// Soumet les événements de `range`, `commit` tous les `batch`, et renvoie
/// l'historique exact : `history[c]` = état au cycle `c`.
fn build(path: &Path, options: Options, range: Range<u32>, batch: u32) -> Vec<Kv> {
    let (mut db, _) = Engine::<Kv>::open_with(path, options).unwrap();
    let mut history = vec![Kv::genesis()];
    for i in range {
        db.submit(&event(i)).unwrap();
        history.push(db.state().clone());
        if (i + 1) % batch == 0 {
            db.commit().unwrap();
        }
    }
    db.commit().unwrap();
    history
}

#[test]
fn restart_uses_latest_snapshot_and_snapshots_are_only_a_cache() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.log");
    let history = build(&path, snap(100), 0..250, 50);

    let (db, _) = Engine::<Kv>::open_with(&path, snap(100)).unwrap();
    assert_eq!(db.snapshots().to_vec(), vec![100, 200]);
    assert_eq!(db.boot_seq(), 200);
    assert_eq!(db.state(), &history[250]);
    drop(db);

    // Sans aucun snapshot, le même état ressort du seul journal.
    std::fs::remove_dir_all(dir.path().join("kv.log.snapshots")).unwrap();
    let (db, _) = Engine::<Kv>::open_with(&path, snap(100)).unwrap();
    assert_eq!(db.boot_seq(), 0);
    assert_eq!(db.state(), &history[250]);
}

#[test]
fn corrupt_snapshot_falls_back_to_an_older_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.log");
    let history = build(&path, snap(100), 0..250, 50);

    let newest = dir.path().join("kv.log.snapshots").join(format!("{:020}.snap", 200u64));
    let mut bytes = std::fs::read(&newest).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&newest, bytes).unwrap();

    let (db, _) = Engine::<Kv>::open_with(&path, snap(100)).unwrap();
    assert_eq!(db.boot_seq(), 100);
    assert_eq!(db.state(), &history[250]);
    assert_eq!(db.state_at(150).unwrap(), history[150]);
}

#[test]
fn snapshot_from_another_history_is_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = (dir.path().join("a.log"), dir.path().join("b.log"));
    build(&a, snap(100), 0..150, 50); // snapshot au cycle 100
    let history_b = build(&b, snap(0), 1000..1150, 50); // même longueur, autres événements

    let foreign = Options { snapshot_dir: Some(dir.path().join("a.log.snapshots")), ..snap(0) };
    let (db, _) = Engine::<Kv>::open_with(&b, foreign).unwrap();
    assert_eq!(db.boot_seq(), 0); // l'ancre ne correspond pas : snapshot écarté
    assert_eq!(db.state(), &history_b[150]);
}

#[test]
fn snapshot_newer_than_journal_is_quarantined() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.log");
    let history = build(&path, snap(100), 0..250, 50); // snapshots aux cycles 100 et 200

    // Le journal perd sa fin (panne machine) : tout ce qui suit 1/4 est remis à zéro.
    let tail = {
        let (db, _) = Engine::<Kv>::open_with(&path, snap(0)).unwrap();
        db.journal().tail() as usize
    };
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[tail / 4..].fill(0);
    std::fs::write(&path, &bytes).unwrap();

    let (db, rec) = Engine::<Kv>::open_with(&path, snap(100)).unwrap();
    assert!(rec.frames > 0 && rec.frames < 100);
    assert_eq!((db.boot_seq(), db.snapshots().len()), (0, 0));
    assert_eq!(db.state(), &history[rec.frames as usize]);

    let orphans = std::fs::read_dir(dir.path().join("kv.log.snapshots"))
        .unwrap()
        .filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "orphan"))
        .count();
    assert_eq!(orphans, 2);
}

#[test]
fn time_travel_and_cursor_match_history() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.log");
    let history = build(&path, snap(40), 0..300, 20);
    let (db, _) = Engine::<Kv>::open_with(&path, snap(40)).unwrap();
    assert!(db.boot_seq() > 0);

    for (cycle, expected) in history.iter().enumerate() {
        assert_eq!(&db.state_at(cycle as u64).unwrap(), expected, "state_at({cycle})");
    }

    let mut cur = db.cursor_at(0).unwrap();
    assert_eq!(cur.peek().unwrap(), Some(event(0)));
    while cur.step_forward().unwrap().is_some() {
        assert_eq!(cur.state(), &history[cur.seq() as usize]);
    }
    assert!(cur.at_end());
    while let Some(undone) = cur.step_back().unwrap() {
        assert_eq!(cur.state(), &history[cur.seq() as usize]);
        assert_eq!(undone, event(cur.seq() as u32)); // le cycle c+1 porte event(c)
    }
    assert_eq!(cur.seq(), 0);

    cur.goto(250).unwrap();
    assert_eq!(cur.state(), &history[250]);
    cur.goto(123).unwrap();
    assert_eq!(cur.state(), &history[123]);
    assert!(cur.goto(301).is_err());

    // Point d'arrêt : premier cycle où l'état compte au moins 8 clés.
    cur.goto(0).unwrap();
    let first = history.iter().position(|s| s.len() >= 8).map(|c| c as u64);
    assert!(first.is_some());
    assert_eq!(cur.run_until(|s, _| s.len() >= 8).unwrap(), first);
}

#[test]
fn retention_keeps_recent_and_milestones() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.log");
    let policy = Options { snapshot_every: 10, keep_recent: 2, milestone_every: 100, snapshot_dir: None };
    build(&path, policy.clone(), 0..400, 10);

    let (db, _) = Engine::<Kv>::open_with(&path, policy).unwrap();
    // Jalons : premier snapshot de chaque centaine ; récents : les deux derniers.
    assert_eq!(db.snapshots().to_vec(), vec![10, 100, 200, 300, 390, 400]);
}