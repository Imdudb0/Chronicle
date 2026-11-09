//! # Chronicle
//!
//! Base de données in-memory déterministe, entièrement dérivée d'un journal
//! d'événements append-only mappé en mémoire.
//!
//! * [`Journal`] : trames `len | crc32 | cycle | payload` dans un fichier mmap.
//! * [`StateMachine`] : transition d'état pure et déterministe (`apply`).
//! * [`Engine`] : journalise puis applique ; rejoue le journal au démarrage.

mod engine;
mod error;
mod journal;
pub mod kv;

pub use engine::{Engine, StateMachine};
pub use error::{Error, Result};
pub use journal::{Frame, Frames, Journal, Recovery};