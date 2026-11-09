//! Journal append-only mappé en mémoire (memmap2).
//!
//! ```text
//! 0    .. 16    en-tête : "CHRNLOG\0" | version u32 LE | réservé u32
//! 16   .. tail  trames  : len u32 LE | crc32 u32 LE | cycle u64 LE | payload[len]
//! tail .. cap   zéros (capacité préallouée)
//! ```
//!
//! Le CRC couvre `len ‖ cycle ‖ payload`. Les cycles commencent à 1 : une trame
//! de cycle 0 n'existe jamais, donc la zone vierge marque la fin du journal sans
//! marqueur explicite. À l'ouverture, les trames sont validées une à une (CRC +
//! continuité des cycles) ; tout ce qui suit la dernière trame valide est une
//! écriture déchirée et est remis à zéro.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use memmap2::MmapMut;

use crate::error::{Error, Result};

const MAGIC: [u8; 8] = *b"CHRNLOG\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 16;
const FRAME_HEADER: usize = 16;
const MAX_PAYLOAD: usize = 16 << 20;
const MIN_CAPACITY: usize = 4096;
const DEFAULT_CAPACITY: u64 = 1 << 20;
/// Une entrée d'index creuse toutes les `INDEX_STRIDE` trames (accès par cycle).
const INDEX_STRIDE: u64 = 1024;

/// Trame validée, empruntée telle quelle au mapping (zéro copie).
#[derive(Debug, Clone, Copy)]
pub struct Frame<'a> {
    pub seq: u64,
    pub payload: &'a [u8],
}

/// Bilan de la vérification faite à l'ouverture.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Recovery {
    /// Trames valides retrouvées.
    pub frames: u64,
    /// Octets écartés après la dernière trame valide (écriture déchirée).
    pub discarded_bytes: u64,
}

pub struct Journal {
    file: File,
    map: MmapMut,
    /// Offset de la prochaine trame.
    tail: usize,
    /// Tout ce qui précède a été msync.
    synced: usize,
    next_seq: u64,
    /// `index[i]` = offset de la trame de cycle `i * INDEX_STRIDE + 1`.
    index: Vec<usize>,
}

impl Journal {
    pub fn open(path: impl AsRef<Path>) -> Result<(Self, Recovery)> {
        Self::open_with_capacity(path, DEFAULT_CAPACITY)
    }

    pub fn open_with_capacity(path: impl AsRef<Path>, capacity: u64) -> Result<(Self, Recovery)> {
        let path = path.as_ref();
        let file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
        lock_exclusive(&file)?;

        let existing = file.metadata()?.len();
        if existing == 0 {
            file.set_len(capacity.max(MIN_CAPACITY as u64))?;
        } else if existing < HEADER_LEN as u64 {
            return Err(Error::BadHeader("file shorter than header"));
        }

        // SAFETY: le fichier est verrouillé en exclusif (flock) et jamais tronqué
        // pendant la vie du mapping. Un tiers non coopératif qui le tronquerait
        // pourrait provoquer un SIGBUS : c'est le contrat habituel de mmap.
        let mut map = unsafe { MmapMut::map_mut(&file)? };

        if map[..HEADER_LEN].iter().all(|&b| b == 0) {
            // Fichier neuf (ou créé puis interrompu avant l'écriture de l'en-tête).
            map[0..8].copy_from_slice(&MAGIC);
            map[8..12].copy_from_slice(&VERSION.to_le_bytes());
            map.flush_range(0, HEADER_LEN)?;
            file.sync_all()?;
            sync_parent_dir(path)?;
        } else {
            if map[0..8] != MAGIC[..] {
                return Err(Error::BadHeader("bad magic"));
            }
            let version = u32::from_le_bytes(map[8..12].try_into().unwrap());
            if version != VERSION {
                return Err(Error::BadHeader("unsupported version"));
            }
        }

        let scan = scan_frames(&map);
        let discarded = scrub_tail(&mut map, scan.tail);
        if discarded > 0 {
            map.flush_range(scan.tail, discarded)?;
        }

        let recovery = Recovery { frames: scan.next_seq - 1, discarded_bytes: discarded as u64 };
        let journal = Journal {
            file,
            map,
            tail: scan.tail,
            synced: scan.tail,
            next_seq: scan.next_seq,
            index: scan.index,
        };
        Ok((journal, recovery))
    }

    /// Ajoute une trame et renvoie son cycle. Aucune garantie de durabilité
    /// tant que [`Journal::sync`] n'a pas été appelé.
    pub fn append(&mut self, payload: &[u8]) -> Result<u64> {
        if payload.len() > MAX_PAYLOAD {
            return Err(Error::PayloadTooLarge(payload.len()));
        }
        let need = FRAME_HEADER + payload.len();
        if self.tail + need > self.map.len() {
            self.grow(need)?;
        }

        let (off, seq) = (self.tail, self.next_seq);
        let body = off + FRAME_HEADER;
        let len = payload.len() as u32;

        let mut header = [0u8; FRAME_HEADER];
        header[0..4].copy_from_slice(&len.to_le_bytes());
        header[4..8].copy_from_slice(&frame_crc(len, seq, payload).to_le_bytes());
        header[8..16].copy_from_slice(&seq.to_le_bytes());

        self.map[body..body + payload.len()].copy_from_slice(payload);
        self.map[off..body].copy_from_slice(&header);

        if (seq - 1) % INDEX_STRIDE == 0 {
            self.index.push(off);
        }
        self.tail = body + payload.len();
        self.next_seq += 1;
        Ok(seq)
    }

    /// Barrière de durabilité (msync). Un crash du *processus* ne perd rien
    /// (pages partagées du noyau) ; une panne machine peut perdre ce qui
    /// n'a pas été synchronisé.
    pub fn sync(&mut self) -> Result<()> {
        if self.tail > self.synced {
            self.map.flush_range(self.synced, self.tail - self.synced)?;
            self.synced = self.tail;
        }
        Ok(())
    }

    /// Dernier cycle journalisé (0 si le journal est vide).
    pub fn last_seq(&self) -> u64 {
        self.next_seq - 1
    }

    /// Offset (en octets) de la fin du journal.
    pub fn tail(&self) -> u64 {
        self.tail as u64
    }

    /// Toutes les trames, depuis le cycle 1.
    pub fn frames(&self) -> Frames<'_> {
        Frames { buf: &self.map[..self.tail], off: HEADER_LEN }
    }

    /// Les trames à partir du cycle `seq` (accès via l'index creux).
    pub fn frames_from(&self, seq: u64) -> Frames<'_> {
        let seq = seq.max(1);
        let buf = &self.map[..self.tail];
        if seq >= self.next_seq {
            return Frames { buf, off: buf.len() };
        }
        let mut it = Frames { buf, off: self.index[((seq - 1) / INDEX_STRIDE) as usize] };
        for _ in 0..(seq - 1) % INDEX_STRIDE {
            it.next();
        }
        it
    }

    fn grow(&mut self, need: usize) -> Result<()> {
        let wanted = self.tail + need;
        let mut cap = self.map.len().max(MIN_CAPACITY);
        while cap < wanted {
            cap *= 2;
        }
        self.file.set_len(cap as u64)?;
        self.file.sync_data()?; // persiste la nouvelle taille
        // SAFETY: voir `open_with_capacity`.
        self.map = unsafe { MmapMut::map_mut(&self.file)? };
        Ok(())
    }
}

/// Itérateur zéro-copie sur des trames déjà validées.
pub struct Frames<'a> {
    buf: &'a [u8],
    off: usize,
}

impl<'a> Iterator for Frames<'a> {
    type Item = Frame<'a>;

    fn next(&mut self) -> Option<Frame<'a>> {
        let (len, _crc, seq) = read_header(self.buf, self.off)?;
        let body = self.off + FRAME_HEADER;
        let payload = self.buf.get(body..body.checked_add(len)?)?;
        self.off = body + len;
        Some(Frame { seq, payload })
    }
}

fn read_header(buf: &[u8], off: usize) -> Option<(usize, u32, u64)> {
    let h = buf.get(off..off.checked_add(FRAME_HEADER)?)?;
    let len = u32::from_le_bytes(h[0..4].try_into().ok()?) as usize;
    let crc = u32::from_le_bytes(h[4..8].try_into().ok()?);
    let seq = u64::from_le_bytes(h[8..16].try_into().ok()?);
    Some((len, crc, seq))
}

fn frame_crc(len: u32, seq: u64, payload: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(&len.to_le_bytes());
    h.update(&seq.to_le_bytes());
    h.update(payload);
    h.finalize()
}

struct Scan {
    tail: usize,
    next_seq: u64,
    index: Vec<usize>,
}

/// Avance tant que les trames sont valides (CRC + cycles consécutifs).
fn scan_frames(buf: &[u8]) -> Scan {
    let (mut off, mut next_seq, mut index) = (HEADER_LEN, 1u64, Vec::new());
    while let Some((len, crc, seq)) = read_header(buf, off) {
        if seq != next_seq || len > MAX_PAYLOAD {
            break;
        }
        let body = off + FRAME_HEADER;
        let Some(payload) = buf.get(body..body + len) else { break };
        if frame_crc(len as u32, seq, payload) != crc {
            break;
        }
        if (seq - 1) % INDEX_STRIDE == 0 {
            index.push(off);
        }
        off = body + len;
        next_seq += 1;
    }
    Scan { tail: off, next_seq, index }
}

/// Remet à zéro les octets non nuls situés après `from`, sans salir les pages
/// déjà vierges. Renvoie l'étendue écartée.
fn scrub_tail(map: &mut [u8], from: usize) -> usize {
    const CHUNK: usize = 4096;
    let (mut off, mut dirty_end) = (from, from);
    while off < map.len() {
        let end = (off + CHUNK).min(map.len());
        if let Some(last) = map[off..end].iter().rposition(|&b| b != 0) {
            dirty_end = off + last + 1;
            map[off..end].fill(0);
        }
        off = end;
    }
    dirty_end - from
}

#[cfg(unix)]
fn lock_exclusive(file: &File) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    // SAFETY: le descripteur est valide tant que `file` est vivant.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc == 0 {
        return Ok(());
    }
    let err = io::Error::last_os_error();
    Err(if err.kind() == io::ErrorKind::WouldBlock { Error::Locked } else { Error::Io(err) })
}

#[cfg(not(unix))]
fn lock_exclusive(_: &File) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn sync_parent_dir(path: &Path) -> Result<()> {
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    File::open(dir)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent_dir(_: &Path) -> Result<()> {
    Ok(())
}