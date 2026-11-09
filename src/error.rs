use std::{fmt, io};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Codec(postcard::Error),
    BadHeader(&'static str),
    PayloadTooLarge(usize),
    /// Un autre processus (ou handle) détient déjà le journal.
    Locked,
    SeqOutOfRange { requested: u64, last: u64 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "i/o error: {e}"),
            Error::Codec(e) => write!(f, "codec error: {e}"),
            Error::BadHeader(why) => write!(f, "invalid journal header: {why}"),
            Error::PayloadTooLarge(n) => write!(f, "payload too large: {n} bytes"),
            Error::Locked => write!(f, "journal is locked by another handle"),
            Error::SeqOutOfRange { requested, last } => {
                write!(f, "cycle {requested} out of range (last = {last})")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Codec(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<postcard::Error> for Error {
    fn from(e: postcard::Error) -> Self {
        Error::Codec(e)
    }
}