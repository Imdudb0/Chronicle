use std::path::{Path, PathBuf};

use serde::{de::DeserializeOwned, Serialize};

use crate::cursor::Cursor;
use crate::error::{Error, Result};
use crate::journal::{Frame, Journal, Recovery};
use crate::snapshot::SnapshotStore;

/// Transition d'état pure.
///
/// `apply` doit être **déterministe et infaillible** : pas d'horloge, de RNG,
/// d'I/O ni d'itération sur un `HashMap`. Toute donnée non déterministe (heure,
/// id aléatoire) est lue *avant* et embarquée dans l'événement.
///
/// L'état doit aussi être sérialisable : c'est ce qu'on écrit dans les snapshots.
pub trait StateMachine: Sized + Serialize + DeserializeOwned {
    type Event: Serialize + DeserializeOwned;

    /// À incrémenter dès que la représentation sérialisée de l'état change
    /// (champ ajouté, réordonné, type modifié) : les snapshots existants sont
    /// alors ignorés et l'état est reconstruit depuis le journal.
    const STATE_VERSION: u32 = 1;

    /// État au cycle 0.
    fn genesis() -> Self;

    fn apply(&mut self, event: &Self::Event);
}

/// Réglages du moteur (voir [`Engine::open_with`]).
#[derive(Debug, Clone)]
pub struct Options {
    /// `commit` écrit un snapshot dès que ce nombre de cycles s'est écoulé depuis
    /// le précédent. `0` = pas de snapshot automatique (les existants restent lus).
    pub snapshot_every: u64,
    /// Nombre de snapshots récents conservés (au moins 1).
    pub keep_recent: usize,
    /// Conserve en plus le premier snapshot de chaque tranche de `milestone_every`
    /// cycles : des jalons espacés qui accélèrent le time-travail dans le passé
    /// lointain. `0` = aucun jalon.
    pub milestone_every: u64,
    /// Dossier des snapshots. Défaut : `<journal>.snapshots`.
    pub snapshot_dir: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self { snapshot_every: 10_000, keep_recent: 3, milestone_every: 100_000, snapshot_dir: None }
    }
}

pub struct Engine<S: StateMachine> {
    journal: Journal,
    snapshots: SnapshotStore,
    opts: Options,
    boot_seq: u64,
    state: S,
}

impl<S: StateMachine> Engine<S> {
    /// Équivaut à `open_with(path, Options::default())`.
    pub fn open(path: impl AsRef<Path>) -> Result<(Self, Recovery)> {
        Self::open_with(path, Options::default())
    }

    /// Ouvre (ou crée) le journal, charge le snapshot valide le plus récent, puis
    /// rejoue seulement les trames qui le suivent.
    pub fn open_with(path: impl AsRef<Path>, opts: Options) -> Result<(Self, Recovery)> {
        let path = path.as_ref();
        let (journal, recovery) = Journal::open(path)?;
        let dir = opts.snapshot_dir.clone().unwrap_or_else(|| default_dir(path));
        let mut snapshots = SnapshotStore::open(dir)?;
        snapshots.quarantine_after(journal.last_seq());
        let (mut state, boot_seq) = snapshots.load_best::<S>(&journal, journal.last_seq());
        fold(&mut state, journal.frames_from(boot_seq + 1), u64::MAX)?;
        Ok((Self { journal, snapshots, opts, boot_seq, state }, recovery))
    }

    /// Journalise l'événement, puis l'applique. Renvoie son cycle.
    ///
    /// On applique l'événement *décodé depuis les octets journalisés*, jamais
    /// l'objet d'origine : chemin « live » et chemin de rejeu sont ainsi
    /// identiques par construction.
    pub fn submit(&mut self, event: &S::Event) -> Result<u64> {
        let bytes = postcard::to_allocvec(event)?;
        let decoded: S::Event = postcard::from_bytes(&bytes)?;
        let seq = self.journal.append(&bytes)?;
        self.state.apply(&decoded);
        Ok(seq)
    }

    /// Barrière de durabilité : tout ce qui a été soumis est sur disque au retour.
    /// Écrit aussi un snapshot si `snapshot_every` cycles se sont écoulés depuis le
    /// dernier.
    ///
    /// Idempotent : si le snapshot échoue (disque plein…), les événements sont
    /// déjà durables et rappeler `commit` retente simplement le snapshot.
    pub fn commit(&mut self) -> Result<()> {
        self.journal.sync()?;
        let every = self.opts.snapshot_every;
        let since = self.last_seq().saturating_sub(self.snapshots.latest().unwrap_or(0));
        if every != 0 && since >= every {
            self.write_snapshot()?;
        }
        Ok(())
    }

    /// Force un snapshot maintenant (ex. avant un arrêt propre). Renvoie son cycle
    /// (0 = journal vide, rien d'écrit).
    pub fn snapshot(&mut self) -> Result<u64> {
        self.journal.sync()?; // un snapshot ne dépasse jamais le journal durable
        self.write_snapshot()
    }

    fn write_snapshot(&mut self) -> Result<u64> {
        let seq = self.snapshots.save(&self.journal, &self.state)?;
        self.snapshots.prune(self.opts.keep_recent.max(1), self.opts.milestone_every);
        Ok(seq)
    }

    pub fn state(&self) -> &S {
        &self.state
    }

    pub fn last_seq(&self) -> u64 {
        self.journal.last_seq()
    }

    pub fn journal(&self) -> &Journal {
        &self.journal
    }

    /// Cycle du snapshot dont on est parti à l'ouverture (0 = rejeu complet).
    pub fn boot_seq(&self) -> u64 {
        self.boot_seq
    }

    /// Cycles des snapshots disponibles, par ordre croissant.
    pub fn snapshots(&self) -> &[u64] {
        self.snapshots.seqs()
    }

    /// Time-travel : reconstruit l'état exact à la fin du cycle `seq` (0 = état
    /// initial). Coût : lecture du snapshot valide le plus proche en amont, puis
    /// rejeu des cycles qui les séparent.
    pub fn state_at(&self, seq: u64) -> Result<S> {
        let last = self.journal.last_seq();
        if seq > last {
            return Err(Error::SeqOutOfRange { requested: seq, last });
        }
        let (mut state, base) = self.snapshots.load_best::<S>(&self.journal, seq);
        fold(&mut state, self.journal.frames_from(base + 1), seq)?;
        Ok(state)
    }

    /// Curseur de debug positionné sur le dernier cycle.
    pub fn cursor(&self) -> Result<Cursor<'_, S>> {
        self.cursor_at(self.last_seq())
    }

    /// Curseur de debug positionné sur le cycle `seq` (0 = état initial).
    pub fn cursor_at(&self, seq: u64) -> Result<Cursor<'_, S>> {
        Ok(Cursor::new(self, seq, self.state_at(seq)?))
    }
}

/// `<journal>.snapshots` : on *ajoute* un suffixe (sans remplacer l'extension),
/// pour que `a.log` et `a.txt` ne partagent jamais le même dossier.
fn default_dir(journal: &Path) -> PathBuf {
    let mut name = journal.as_os_str().to_owned();
    name.push(".snapshots");
    PathBuf::from(name)
}

fn fold<'a, S: StateMachine>(
    state: &mut S,
    frames: impl Iterator<Item = Frame<'a>>,
    upto: u64,
) -> Result<()> {
    for frame in frames.take_while(|f| f.seq <= upto) {
        let event: S::Event = postcard::from_bytes(frame.payload)?;
        state.apply(&event);
    }
    Ok(())
}