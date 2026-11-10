//! Canal « une valeur, une fois », entre threads. Côté réception, c'est une
//! `Future` (dans une tâche) ou un appel bloquant `wait` (dans un thread).
//!
//! Le `Mutex` n'est tenu que quelques instructions, et jamais pendant
//! `Waker::wake` (qui peut exécuter du code arbitraire).

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};

struct State<T> {
    value: Option<T>,
    waker: Option<Waker>,
    tx_dropped: bool,
    rx_dropped: bool,
}

struct Inner<T> {
    state: Mutex<State<T>>,
    ready: Condvar,
}

fn lock<T>(m: &Mutex<State<T>>) -> MutexGuard<'_, State<T>> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) struct Sender<T> {
    inner: Arc<Inner<T>>,
}

pub(crate) struct Receiver<T> {
    inner: Arc<Inner<T>>,
}

/// L'émetteur a disparu sans rien envoyer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Canceled;

pub(crate) fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let inner = Arc::new(Inner {
        state: Mutex::new(State { value: None, waker: None, tx_dropped: false, rx_dropped: false }),
        ready: Condvar::new(),
    });
    (Sender { inner: inner.clone() }, Receiver { inner })
}

impl<T> Sender<T> {
    /// Dépose la valeur ; la rend si le récepteur n'existe plus. Le réveil du
    /// récepteur est fait par `Drop`, une fois le verrou relâché.
    pub(crate) fn send(self, value: T) -> Result<(), T> {
        let mut st = lock(&self.inner.state);
        if st.rx_dropped {
            return Err(value);
        }
        st.value = Some(value);
        Ok(())
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let waker = {
            let mut st = lock(&self.inner.state);
            st.tx_dropped = true;
            st.waker.take()
        };
        self.inner.ready.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

impl<T> Receiver<T> {
    /// Attend la valeur en bloquant le thread courant. Jamais depuis une tâche
    /// de la boucle d'événements : utilise `.await`.
    pub(crate) fn wait(self) -> Result<T, Canceled> {
        let mut st = lock(&self.inner.state);
        loop {
            if let Some(value) = st.value.take() {
                return Ok(value);
            }
            if st.tx_dropped {
                return Err(Canceled);
            }
            st = self.inner.ready.wait(st).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        lock(&self.inner.state).rx_dropped = true;
    }
}

impl<T> Future for Receiver<T> {
    type Output = Result<T, Canceled>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut st = lock(&self.inner.state);
        if let Some(value) = st.value.take() {
            return Poll::Ready(Ok(value));
        }
        if st.tx_dropped {
            return Poll::Ready(Err(Canceled));
        }
        if !st.waker.as_ref().is_some_and(|w| w.will_wake(cx.waker())) {
            st.waker = Some(cx.waker().clone());
        }
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Wake;

    struct Count(AtomicUsize);

    impl Wake for Count {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn future_is_woken_when_the_value_arrives() {
        let count = Arc::new(Count(AtomicUsize::new(0)));
        let waker = Waker::from(count.clone());
        let mut cx = Context::from_waker(&waker);
        let (tx, mut rx) = channel::<u32>();

        assert!(Pin::new(&mut rx).poll(&mut cx).is_pending());
        tx.send(5).unwrap();
        assert_eq!(count.0.load(Ordering::SeqCst), 1);
        assert_eq!(Pin::new(&mut rx).poll(&mut cx), Poll::Ready(Ok(5)));
    }

    #[test]
    fn dropped_sender_cancels_and_dropped_receiver_returns_the_value() {
        let (tx, rx) = channel::<u8>();
        drop(tx);
        assert_eq!(rx.wait(), Err(Canceled));

        let (tx, rx) = channel::<u8>();
        drop(rx);
        assert_eq!(tx.send(9), Err(9));
    }

    #[test]
    fn wait_blocks_until_another_thread_sends() {
        let (tx, rx) = channel::<u8>();
        let sender = std::thread::spawn(move || tx.send(42).unwrap());
        assert_eq!(rx.wait(), Ok(42));
        sender.join().unwrap();
    }
}