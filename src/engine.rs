use std::path::Path;

use serde::{de::DeserializeOwned, Serialize};

use crate::error::{Error, Result};
use crate::journal::{Frame, Journal, Recovery};

/// Transition d'état pure.
///
/// `apply` doit être **déterministe et infaillible** : pas d'horloge, de RNG,
/// d'I/O ni d'itération sur un `HashMap`. Toute donnée non déterministe (heure,
/// id aléatoire) est lue *avant* et embarquée dans l'événement.
pub trait StateMachine: Sized {
    type Event: Serialize + DeserializeOwned;

    /// État au cycle 0.
    fn genesis() -> Self;

    fn apply(&mut self, event: &Self::Event);
}

pub struct Engine<S: StateMachine> {
    journal: Journal,
    state: S,
}

impl<S: StateMachine> Engine<S> {
    /// Ouvre (ou crée) le journal et reconstruit l'état en RAM par rejeu.
    pub fn open(path: impl AsRef<Path>) -> Result<(Self, Recovery)> {
        let (journal, recovery) = Journal::open(path)?;
        let mut state = S::genesis();
        fold(&mut state, journal.frames(), u64::MAX)?;
        Ok((Self { journal, state }, recovery))
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
    pub fn commit(&mut self) -> Result<()> {
        self.journal.sync()
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

    /// Time-travel : reconstruit l'état exact à la fin du cycle `seq`
    /// (0 = état initial). Version naïve en O(n) ; accélérée par snapshots en partie 2.
    pub fn state_at(&self, seq: u64) -> Result<S> {
        let last = self.journal.last_seq();
        if seq > last {
            return Err(Error::SeqOutOfRange { requested: seq, last });
        }
        let mut state = S::genesis();
        fold(&mut state, self.journal.frames(), seq)?;
        Ok(state)
    }
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