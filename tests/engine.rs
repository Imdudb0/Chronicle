use std::io::{Read, Seek, SeekFrom, Write};

use chronicle::kv::{Kv, KvEvent};
use chronicle::{Engine, Journal};

fn put(k: &str, v: &str) -> KvEvent {
    KvEvent::Put { key: k.as_bytes().to_vec(), value: v.as_bytes().to_vec() }
}

fn del(k: &str) -> KvEvent {
    KvEvent::Delete { key: k.as_bytes().to_vec() }
}

#[test]
fn state_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.log");
    {
        let (mut db, rec) = Engine::<Kv>::open(&path).unwrap();
        assert_eq!(rec.frames, 0);
        assert_eq!(db.submit(&put("a", "1")).unwrap(), 1);
        assert_eq!(db.submit(&put("b", "2")).unwrap(), 2);
        db.submit(&del("a")).unwrap();
        db.commit().unwrap();
    }
    let (db, rec) = Engine::<Kv>::open(&path).unwrap();
    assert_eq!(rec.frames, 3);
    assert_eq!(db.last_seq(), 3);
    assert_eq!(db.state().get(b"a"), None);
    assert_eq!(db.state().get(b"b"), Some(&b"2"[..]));
}

#[test]
fn time_travel_replays_any_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let (mut db, _) = Engine::<Kv>::open(dir.path().join("kv.log")).unwrap();
    db.submit(&put("k", "v1")).unwrap();
    db.submit(&put("k", "v2")).unwrap();
    db.submit(&del("k")).unwrap();

    assert!(db.state_at(0).unwrap().is_empty());
    assert_eq!(db.state_at(1).unwrap().get(b"k"), Some(&b"v1"[..]));
    assert_eq!(db.state_at(2).unwrap().get(b"k"), Some(&b"v2"[..]));
    assert_eq!(db.state_at(3).unwrap().get(b"k"), None);
    assert!(db.state_at(4).is_err());
}

#[test]
fn corrupted_tail_is_discarded_and_log_stays_appendable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.log");
    let tail = {
        let (mut db, _) = Engine::<Kv>::open(&path).unwrap();
        for i in 0..5 {
            db.submit(&put(&format!("k{i}"), "v")).unwrap();
        }
        db.commit().unwrap();
        db.journal().tail()
    };

    // Altère le dernier octet de la dernière trame.
    let mut f = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
    let mut byte = [0u8; 1];
    f.seek(SeekFrom::Start(tail - 1)).unwrap();
    f.read_exact(&mut byte).unwrap();
    f.seek(SeekFrom::Start(tail - 1)).unwrap();
    f.write_all(&[byte[0] ^ 0xFF]).unwrap();
    drop(f);

    let (mut db, rec) = Engine::<Kv>::open(&path).unwrap();
    assert_eq!(rec.frames, 4);
    assert!(rec.discarded_bytes > 0);
    assert_eq!(db.state().len(), 4);
    assert_eq!(db.submit(&put("after", "crash")).unwrap(), 5);
    db.commit().unwrap();
    drop(db);

    let (db, rec) = Engine::<Kv>::open(&path).unwrap();
    assert_eq!((rec.frames, rec.discarded_bytes), (5, 0));
    assert_eq!(db.state().get(b"after"), Some(&b"crash"[..]));
}

/// Simule une coupure à *chaque* octet : tout ce qui suit `cut` n'a jamais
/// atteint le disque. L'état récupéré doit toujours être un préfixe exact de
/// l'historique, et croître avec `cut`.
#[test]
fn recovery_from_every_cut_point() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.log");

    let (mut reference, _) = Engine::<Kv>::open(&path).unwrap();
    for i in 0..20 {
        reference.submit(&put(&format!("k{}", i % 7), &format!("v{i}"))).unwrap();
    }
    reference.commit().unwrap();

    let tail = reference.journal().tail() as usize;
    let image_len = tail.next_multiple_of(4096);
    let image = std::fs::read(&path).unwrap()[..image_len].to_vec();

    let mut previous = 0;
    for cut in 16..=tail {
        let mut bytes = image.clone();
        bytes[cut..].fill(0);
        let copy = dir.path().join("cut.log");
        std::fs::write(&copy, &bytes).unwrap();

        let (db, rec) = Engine::<Kv>::open(&copy).unwrap();
        assert!(rec.frames >= previous, "cut {cut}");
        previous = rec.frames;
        assert_eq!(db.state(), &reference.state_at(rec.frames).unwrap(), "cut {cut}");
        drop(db);
        std::fs::remove_file(&copy).unwrap();
    }
    assert_eq!(previous, 20);
}

#[test]
fn journal_grows_and_seeks_by_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("raw.log");
    {
        let (mut j, _) = Journal::open_with_capacity(&path, 4096).unwrap();
        for i in 0..3000u32 {
            assert_eq!(j.append(&i.to_le_bytes()).unwrap(), i as u64 + 1);
        }
        j.sync().unwrap();
    }
    let (j, rec) = Journal::open(&path).unwrap();
    assert_eq!(rec.frames, 3000);

    let f = j.frames_from(2049).next().unwrap();
    assert_eq!(f.seq, 2049);
    assert_eq!(f.payload, &2048u32.to_le_bytes()[..]);
    assert_eq!(j.frames_from(2990).count(), 11);
    assert_eq!(j.frames_from(3001).count(), 0);
}

#[cfg(unix)]
#[test]
fn second_open_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kv.log");
    let first = Journal::open(&path).unwrap();
    assert!(matches!(Journal::open(&path), Err(chronicle::Error::Locked)));
    drop(first);
    assert!(Journal::open(&path).is_ok());
}