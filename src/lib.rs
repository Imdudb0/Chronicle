//! # Chronicle
//!
//! Base de données in-memory déterministe, entièrement dérivée d'un journal
//! d'événements append-only mappé en mémoire.
//!
//! * [`Journal`] : trames `len | crc32 | cycle | payload` dans un fichier mmap.
//! * [`StateMachine`] : transition d'état pure et déterministe (`apply`).
//! * [`Engine`] : journalise puis applique ; redémarre depuis le dernier
//!   snapshot valide ; time-travel via [`Engine::state_at`] et [`Cursor`].

mod cursor;
mod engine;
mod error;
mod journal;
mod snapshot;
pub mod kv;

pub use cursor::Cursor;
pub use engine::{Engine, Options, StateMachine};
pub use error::{Error, Result};
pub use journal::{Frame, Frames, Journal, Recovery};