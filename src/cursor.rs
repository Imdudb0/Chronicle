use crate::engine::{Engine, StateMachine};
use crate::error::{Error, Result};
use crate::journal::Frames;

/// Curseur de time-travel debugging : vue **en lecture seule** de l'état à un
/// cycle donné, qu'on déplace dans l'historique.
///
/// * en avant, `step_forward` applique un seul événement : O(1) ;
/// * en arrière, on ne sait pas « dé-appliquer » : on repart du snapshot le plus
///   proche en amont et on rejoue jusqu'à la cible.
///
/// Le curseur emprunte l'[`Engine`] : impossible de `submit` tant qu'il existe
/// (ce qui protège aussi le mapping mémoire, remappé quand le journal grandit).
pub struct Cursor<'e, S: StateMachine> {
    engine: &'e Engine<S>,
    seq: u64,
    state: S,
    /// Rend la trame `seq + 1` au prochain `next()`.
    frames: Frames<'e>,
}

impl<'e, S: StateMachine> Cursor<'e, S> {
    pub(crate) fn new(engine: &'e Engine<S>, seq: u64, state: S) -> Self {
        let frames = engine.journal().frames_from(seq + 1);
        Self { engine, seq, state, frames }
    }

    /// Cycle courant (0 = état initial).
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// État exact à la fin du cycle courant.
    pub fn state(&self) -> &S {
        &self.state
    }

    pub fn into_state(self) -> S {
        self.state
    }

    pub fn at_end(&self) -> bool {
        self.seq == self.engine.last_seq()
    }

    /// Événement que `step_forward` appliquerait (cycle `seq + 1`).
    pub fn peek(&self) -> Result<Option<S::Event>> {
        self.event_at(self.seq + 1)
    }

    /// Applique le cycle suivant et renvoie son événement ; `None` en fin de journal.
    pub fn step_forward(&mut self) -> Result<Option<S::Event>> {
        let Some(frame) = self.frames.next() else { return Ok(None) };
        debug_assert_eq!(frame.seq, self.seq + 1);
        let event: S::Event = postcard::from_bytes(frame.payload)?;
        self.state.apply(&event);
        self.seq = frame.seq;
        Ok(Some(event))
    }

    /// Revient au cycle précédent et renvoie l'événement annulé ; `None` au cycle 0.
    pub fn step_back(&mut self) -> Result<Option<S::Event>> {
        if self.seq == 0 {
            return Ok(None);
        }
        let undone = self.seq;
        self.goto(undone - 1)?;
        self.event_at(undone)
    }

    /// Saute au cycle `target` : rejeu incrémental si on avance, snapshot le plus
    /// proche si on recule.
    pub fn goto(&mut self, target: u64) -> Result<()> {
        let engine = self.engine;
        let last = engine.last_seq();
        if target > last {
            return Err(Error::SeqOutOfRange { requested: target, last });
        }
        if target < self.seq {
            self.state = engine.state_at(target)?;
            self.seq = target;
            self.frames = engine.journal().frames_from(target + 1);
        } else {
            while self.seq < target {
                if self.step_forward()?.is_none() {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Point d'arrêt : avance jusqu'au premier cycle suivant où `stop(état, cycle)`
    /// est vrai et renvoie ce cycle. `None` si la fin du journal est atteinte (le
    /// curseur y reste).
    pub fn run_until(&mut self, mut stop: impl FnMut(&S, u64) -> bool) -> Result<Option<u64>> {
        while self.step_forward()?.is_some() {
            if stop(&self.state, self.seq) {
                return Ok(Some(self.seq));
            }
        }
        Ok(None)
    }

    fn event_at(&self, seq: u64) -> Result<Option<S::Event>> {
        match self.engine.journal().frames_from(seq).next() {
            Some(frame) if frame.seq == seq => Ok(Some(postcard::from_bytes(frame.payload)?)),
            _ => Ok(None),
        }
    }
}