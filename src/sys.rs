//! Appels système bruts (Linux) : `eventfd` et `epoll`, directement via `libc`.

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::time::Duration;

pub(crate) const EPOLLIN: u32 = libc::EPOLLIN as u32;

fn cvt(rc: libc::c_int) -> io::Result<libc::c_int> {
    if rc < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(rc)
    }
}

/// Compteur noyau : y écrire rend le descripteur lisible, donc réveille un
/// `epoll_wait`, depuis n'importe quel thread.
pub(crate) struct EventFd(OwnedFd);

impl EventFd {
    pub(crate) fn new() -> io::Result<Self> {
        // SAFETY: appel système sans pointeur.
        let raw = cvt(unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) })?;
        // SAFETY: descripteur neuf, dont nous sommes l'unique propriétaire.
        Ok(Self(unsafe { OwnedFd::from_raw_fd(raw) }))
    }

    /// Rend le descripteur lisible. Si le compteur est saturé (EAGAIN), il
    /// l'est déjà : rien à faire.
    pub(crate) fn notify(&self) {
        let one = 1u64.to_ne_bytes();
        loop {
            // SAFETY: `one` est valide en lecture pour 8 octets.
            let n = unsafe { libc::write(self.0.as_raw_fd(), one.as_ptr().cast(), one.len()) };
            if n >= 0 || io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                return;
            }
        }
    }

    /// Remet le compteur à zéro (sans effet s'il l'est déjà).
    pub(crate) fn drain(&self) {
        let mut buf = [0u8; 8];
        // SAFETY: `buf` est valide en écriture pour 8 octets.
        unsafe { libc::read(self.0.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) };
    }
}

impl AsRawFd for EventFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

/// Instance epoll. Déclenchement par niveau : un descripteur reste signalé
/// tant qu'on ne l'a pas vidé.
pub(crate) struct Epoll(OwnedFd);

impl Epoll {
    pub(crate) fn new() -> io::Result<Self> {
        // SAFETY: appel système sans pointeur.
        let raw = cvt(unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) })?;
        // SAFETY: descripteur neuf, dont nous sommes l'unique propriétaire.
        Ok(Self(unsafe { OwnedFd::from_raw_fd(raw) }))
    }

    /// Surveille `fd` ; `token` sera rendu par [`Epoll::wait`] quand il est signalé.
    pub(crate) fn add(&self, fd: RawFd, token: u64, events: u32) -> io::Result<()> {
        let mut event = libc::epoll_event { events, u64: token };
        // SAFETY: `event` est valide pour la durée de l'appel.
        cvt(unsafe { libc::epoll_ctl(self.0.as_raw_fd(), libc::EPOLL_CTL_ADD, fd, &mut event) })?;
        Ok(())
    }

    /// Bloque jusqu'à un événement (indéfiniment si `timeout` est `None`) et
    /// ajoute à `out` les jetons signalés. Une interruption par signal n'est
    /// pas une erreur : retour sans jeton.
    pub(crate) fn wait(&self, out: &mut Vec<u64>, timeout: Option<Duration>) -> io::Result<()> {
        const MAX: usize = 64;
        let millis = match timeout {
            None => -1,
            Some(d) => d.as_millis().min(i32::MAX as u128) as libc::c_int,
        };
        let mut events: Vec<libc::epoll_event> = Vec::with_capacity(MAX);
        // SAFETY: le tampon a la place de MAX événements ; le noyau en écrit `n`.
        let n = unsafe {
            libc::epoll_wait(self.0.as_raw_fd(), events.as_mut_ptr(), MAX as libc::c_int, millis)
        };
        if n < 0 {
            let err = io::Error::last_os_error();
            return if err.kind() == io::ErrorKind::Interrupted { Ok(()) } else { Err(err) };
        }
        // SAFETY: les `n` premiers événements viennent d'être initialisés par le noyau.
        unsafe { events.set_len(n as usize) };
        out.extend(events.iter().map(|e| e.u64));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eventfd_wakes_epoll_until_drained() {
        let (ep, ev) = (Epoll::new().unwrap(), EventFd::new().unwrap());
        ep.add(ev.as_raw_fd(), 7, EPOLLIN).unwrap();

        let mut tokens = Vec::new();
        ep.wait(&mut tokens, Some(Duration::ZERO)).unwrap();
        assert!(tokens.is_empty());

        ev.notify();
        ev.notify(); // les écritures s'additionnent : un seul drain suffit
        ep.wait(&mut tokens, Some(Duration::from_secs(1))).unwrap();
        assert_eq!(tokens, [7]);

        ev.drain();
        tokens.clear();
        ep.wait(&mut tokens, Some(Duration::ZERO)).unwrap();
        assert!(tokens.is_empty());
    }
}