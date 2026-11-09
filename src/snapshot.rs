//! Snapshots : copies sérialisées de l'état à un cycle donné.
//!
//! Un snapshot est un **cache** : le journal reste l'unique source de vérité.
//! Illisible, périmé ou étranger au journal, un snapshot est simplement ignoré ;
//! on peut même supprimer tout le dossier sans perdre une donnée.
//!
//! ```text
//! 0   .. 8     "CHRNSNP\0"
//! 8   .. 12    version du format u32 LE
//! 12  .. 16    StateMachine::STATE_VERSION u32 LE
//! 16  .. 24    cycle u64 LE (état *après* application de ce cycle)
//! 24  .. 28    ancre u32 LE : empreinte de la trame `cycle` du journal
//! 28  .. n-4   état sérialisé (postcard)
//! n-4 .. n     crc32 de tout ce qui précède
//! ```
//!
//! Fichier `<cycle sur 20 chiffres>.snap`, écrit sous `.tmp`, fsync, renommé,
//! puis dossier fsync : un snapshot apparaît entier ou pas du tout.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::journal::Journal;
use crate::StateMachine;

const MAGIC: [u8; 8] = *b"CHRNSNP\0";
const FORMAT: u32 = 1;
const HEADER: usize = 28;

pub(crate) struct SnapshotStore {
    dir: PathBuf,
    /// Cycles des snapshots présents, par ordre croissant.
    seqs: Vec<u64>,
}

impl SnapshotStore {
    /// Liste le dossier (absent = vide) et supprime les `.tmp` d'écritures interrompues.
    pub(crate) fn open(dir: PathBuf) -> Result<Self> {
        let mut seqs: Vec<u64> = Vec::new();
        match fs::read_dir(&dir) {
            Ok(entries) => {
                for entry in entries {
                    let path = entry?.path();
                    match path.extension().and_then(|e| e.to_str()) {
                        Some("tmp") => {
                            let _ = fs::remove_file(&path);
                        }
                        Some("snap") => {
                            let stem = path.file_stem().and_then(|s| s.to_str());
                            if let Some(seq) = stem.and_then(|s| s.parse::<u64>().ok()) {
                                seqs.push(seq);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        seqs.sort_unstable();
        Ok(Self { dir, seqs })
    }

    pub(crate) fn seqs(&self) -> &[u64] {
        &self.seqs
    }

    pub(crate) fn latest(&self) -> Option<u64> {
        self.seqs.last().copied()
    }

    /// Écrit le snapshot de `state`, qui doit être l'état au dernier cycle de
    /// `journal`. L'appelant garantit que le journal est durable jusque-là.
    pub(crate) fn save<S: StateMachine>(&mut self, journal: &Journal, state: &S) -> Result<u64> {
        let seq = journal.last_seq();
        if seq == 0 {
            return Ok(0);
        }
        let payload = postcard::to_allocvec(state)?;
        let mut head = [0u8; HEADER];
        head[0..8].copy_from_slice(&MAGIC);
        head[8..12].copy_from_slice(&FORMAT.to_le_bytes());
        head[12..16].copy_from_slice(&S::STATE_VERSION.to_le_bytes());
        head[16..24].copy_from_slice(&seq.to_le_bytes());
        head[24..28].copy_from_slice(&anchor_of(journal, seq).unwrap_or_default().to_le_bytes());

        let mut crc = crc32fast::Hasher::new();
        crc.update(&head);
        crc.update(&payload);
        let trailer = crc.finalize().to_le_bytes();

        fs::create_dir_all(&self.dir)?;
        let parts: [&[u8]; 3] = [&head, &payload, &trailer];
        write_atomic(&self.dir, &snap_name(seq), &parts)?;
        if let Err(pos) = self.seqs.binary_search(&seq) {
            self.seqs.insert(pos, seq);
        }
        Ok(seq)
    }

    /// Snapshot valide le plus récent de cycle ≤ `upto`, sinon l'état initial.
    /// Renvoie l'état et le cycle qu'il couvre (0 = genesis).
    pub(crate) fn load_best<S: StateMachine>(&self, journal: &Journal, upto: u64) -> (S, u64) {
        for &seq in self.seqs.iter().rev() {
            if seq > upto {
                continue;
            }
            if let Some(state) = self.load::<S>(journal, seq) {
                return (state, seq);
            }
        }
        (S::genesis(), 0)
    }

    fn load<S: StateMachine>(&self, journal: &Journal, seq: u64) -> Option<S> {
        let bytes = fs::read(snap_path(&self.dir, seq)).ok()?;
        let (body, trailer) = bytes.split_at(bytes.len().checked_sub(4)?);
        if body.len() < HEADER || crc32fast::hash(body) != u32::from_le_bytes(trailer.try_into().ok()?) {
            return None;
        }
        let u32_at = |o: usize| u32::from_le_bytes(body[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(body[o..o + 8].try_into().unwrap());
        let valid = body[0..8] == MAGIC[..]
            && u32_at(8) == FORMAT
            && u32_at(12) == S::STATE_VERSION
            && u64_at(16) == seq
            && Some(u32_at(24)) == anchor_of(journal, seq);
        if !valid {
            return None;
        }
        // Tout le payload doit être consommé : détecte un état dont le schéma a changé.
        let (state, rest) = postcard::take_from_bytes::<S>(&body[HEADER..]).ok()?;
        rest.is_empty().then_some(state)
    }

    /// Met de côté (`.orphan`, jamais supprimés) les snapshots plus récents que le
    /// journal : ils décrivent un futur que ce journal n'a pas (ou plus).
    pub(crate) fn quarantine_after(&mut self, seq: u64) {
        let keep = self.seqs.partition_point(|&s| s <= seq);
        for s in self.seqs.drain(keep..) {
            let path = snap_path(&self.dir, s);
            let _ = fs::rename(&path, path.with_extension("orphan"));
        }
    }

    /// Garde les `keep_recent` plus récents, plus le premier snapshot de chaque
    /// tranche de `milestone_every` cycles (0 = aucun jalon). Supprime le reste.
    pub(crate) fn prune(&mut self, keep_recent: usize, milestone_every: u64) {
        let recent_from = self.seqs.len().saturating_sub(keep_recent);
        let mut previous = None;
        let mut kept = Vec::with_capacity(self.seqs.len());
        for (i, &seq) in self.seqs.iter().enumerate() {
            let bucket = (milestone_every != 0).then(|| seq / milestone_every);
            let milestone = bucket.is_some() && bucket != previous;
            previous = bucket;
            if i >= recent_from || milestone {
                kept.push(seq);
            } else {
                let _ = fs::remove_file(snap_path(&self.dir, seq));
            }
        }
        self.seqs = kept;
    }
}

fn snap_name(seq: u64) -> String {
    format!("{seq:020}.snap")
}

fn snap_path(dir: &Path, seq: u64) -> PathBuf {
    dir.join(snap_name(seq))
}

/// Empreinte (CRC de cycle ‖ payload) de la trame `seq` : lie un snapshot à *ce*
/// journal. `None` si le journal n'a pas ce cycle.
fn anchor_of(journal: &Journal, seq: u64) -> Option<u32> {
    let frame = journal.frames_from(seq).next().filter(|f| f.seq == seq)?;
    let mut h = crc32fast::Hasher::new();
    h.update(&frame.seq.to_le_bytes());
    h.update(frame.payload);
    Some(h.finalize())
}

/// Écrit `parts` dans `dir/name` de façon atomique : tmp, fsync, rename, fsync du dossier.
fn write_atomic(dir: &Path, name: &str, parts: &[&[u8]]) -> io::Result<()> {
    let tmp = dir.join(format!("{name}.tmp"));
    let written = (|| -> io::Result<()> {
        let mut file = File::create(&tmp)?;
        for part in parts {
            file.write_all(part)?;
        }
        file.sync_all()?;
        fs::rename(&tmp, dir.join(name))
    })();
    if written.is_err() {
        let _ = fs::remove_file(&tmp);
        return written;
    }
    sync_dir(dir)
}

#[cfg(unix)]
fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_: &Path) -> io::Result<()> {
    Ok(())
}