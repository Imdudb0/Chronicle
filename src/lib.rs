//! # Chronicle
//!
//! Base de données in-memory déterministe, entièrement dérivée d'un journal
//! d'événements append-only mappé en mémoire.
//!
//! * [`Journal`] : trames `len | crc32 | cycle | payload` dans un fichier mmap.
//! * [`StateMachine`] : transition d'état pure et déterministe (`apply`).
//! * [`Engine`] : journalise puis applique ; redémarre depuis le dernier
//!   snapshot valide ; time-travel via [`Engine::state_at`] et [`Cursor`].
//! * `Server` (Linux) : boucle d'événements mono-thread sans Tokio, group
//!   commit, `Handle` clonable pour les clients.

mod cursor;
mod engine;
mod error;
mod journal;
mod snapshot;
pub mod kv;

#[cfg(target_os = "linux")]
mod oneshot;
#[cfg(target_os = "linux")]
mod runtime;
#[cfg(target_os = "linux")]
mod sys;

pub use cursor::Cursor;
pub use engine::{Engine, Options, StateMachine};
pub use error::{Error, Result};
pub use journal::{Frame, Frames, Journal, Recovery};
#[cfg(target_os = "linux")]
pub use runtime::{block_on, Ctx, Handle, Pending, Server, Stats};