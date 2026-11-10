//! Boucle d'événements mono-thread, sans Tokio.
//!
//! Un thread unique possède l'`Engine` : aucun verrou sur l'état, aucune course
//! possible, et `S` n'a même pas besoin d'être `Send` (il naît et reste sur ce
//! thread). Les autres threads déposent des commandes dans une file (un `Mutex`
//! tenu quelques instructions, jamais pendant un I/O ni du code utilisateur) et
//! réveillent la boucle par un `eventfd` surveillé par `epoll`.
//!
//! ```text
//!  clients (n threads)               thread « chronicle-loop »
//!  Handle::submit ─► file + eventfd ─► epoll_wait
//!                                      1. exécute les commandes   (apply en RAM)
//!                                      2. interroge les tâches réveillées
//!                                      3. UN msync pour tout le lot  (group commit)
//!   Pending (Future / wait) ◄───────── 4. répond, réveille les tâches
//!                                      5. snapshot périodique
//! ```
//!
//! * **Group commit** : pendant que la boucle synchronise, les clients empilent
//!   leurs commandes ; le lot suivant les valide toutes d'un seul `msync`.
//! * **Pas de lecture sale** : aucune réponse (écriture *ou* lecture) ne part
//!   avant que le journal soit durable jusqu'au cycle qu'elle reflète.
//! * **Panne d'écriture** : si `msync` échoue, l'état en RAM a pris de l'avance
//!   sur le journal, et un `fsync` réessayé après échec peut mentir. La boucle
//!   répond l'erreur au lot, s'arrête, et `Server::shutdown` la renvoie. On
//!   repart alors du journal.

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::io;
use std::mem;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, JoinHandle, Thread};

use crate::engine::{Engine, Options, StateMachine};
use crate::error::{Error, Result};
use crate::journal::Recovery;
use crate::oneshot::{self, Sender};
use crate::sys::{Epoll, EventFd, EPOLLIN};

/// Jeton epoll de l'eventfd de réveil (les sockets de la partie 5 auront les leurs).
const WAKE: u64 = 0;

type LocalFuture = Pin<Box<dyn Future<Output = ()>>>;
/// Réponse retenue jusqu'au msync du lot ; reçoit l'erreur si celui-ci a échoué.
type Deferred = Box<dyn FnOnce(Option<&Error>)>;

// ───────────────────────────── côté multi-thread ─────────────────────────────

struct Queue<S: StateMachine> {
    jobs: Vec<Box<dyn FnOnce(&Ctx<S>) + Send>>,
    woken: Vec<usize>,
    closed: bool,
}

impl<S: StateMachine> Queue<S> {
    fn new(closed: bool) -> Self {
        Queue { jobs: Vec::new(), woken: Vec::new(), closed }
    }
}

struct Shared<S: StateMachine> {
    queue: Mutex<Queue<S>>,
    wake: EventFd,
    syncs: AtomicU64,
    durable: AtomicU64,
    snapshot_errors: AtomicU64,
}

fn lock<S: StateMachine>(queue: &Mutex<Queue<S>>) -> MutexGuard<'_, Queue<S>> {
    queue.lock().unwrap_or_else(PoisonError::into_inner)
}

impl<S: StateMachine> Shared<S> {
    fn new() -> io::Result<Self> {
        Ok(Shared {
            queue: Mutex::new(Queue::new(false)),
            wake: EventFd::new()?,
            syncs: AtomicU64::new(0),
            durable: AtomicU64::new(0),
            snapshot_errors: AtomicU64::new(0),
        })
    }

    /// Dépose une commande, puis réveille la boucle (dans cet ordre : un réveil
    /// vidé par la boucle correspond toujours à une commande déjà visible).
    /// `false` si la boucle est fermée : la commande est alors détruite, ce qui
    /// annule sa réponse (`Error::Closed`).
    fn post<F>(&self, job: F) -> bool
    where
        F: FnOnce(&Ctx<S>) + Send + 'static,
    {
        let rejected = {
            let mut queue = lock(&self.queue);
            if queue.closed {
                Some(job)
            } else {
                queue.jobs.push(Box::new(job));
                None
            }
        };
        // `rejected` est détruit à la sortie, verrou déjà relâché.
        if rejected.is_some() {
            return false;
        }
        self.wake.notify();
        true
    }

    /// Rend tout ce qui a été déposé (commandes, tâches réveillées) et l'état de fermeture.
    fn take(&self) -> Queue<S> {
        let mut queue = lock(&self.queue);
        let closed = queue.closed;
        mem::replace(&mut *queue, Queue::new(closed))
    }

    /// Fermeture demandée par `Server` : les commandes déjà déposées seront traitées.
    fn request_close(&self) {
        lock(&self.queue).closed = true;
        self.wake.notify();
    }

    /// Fermeture définitive, à la sortie de la boucle. Rend les commandes jamais
    /// traitées ; l'appelant les détruit *hors du verrou* (leur destruction réveille
    /// des wakers, qui prennent ce même verrou).
    fn close(&self) -> Queue<S> {
        let mut queue = lock(&self.queue);
        mem::replace(&mut *queue, Queue::new(true))
    }
}

/// Réveil d'une tâche : un drapeau (évite les doublons dans la file) + son numéro.
struct TaskWaker<S: StateMachine> {
    id: usize,
    queued: AtomicBool,
    shared: Arc<Shared<S>>,
}

impl<S: StateMachine + 'static> Wake for TaskWaker<S> {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        if !self.queued.swap(true, Ordering::AcqRel) {
            lock(&self.shared.queue).woken.push(self.id);
            self.shared.wake.notify();
        }
    }
}

/// Garde de sortie : quoi qu'il arrive (arrêt, erreur fatale, panique dans
/// `apply`), la file est fermée et les commandes restantes sont annulées. Aucun
/// client ne reste bloqué.
struct CloseOnExit<S: StateMachine>(Arc<Shared<S>>);

impl<S: StateMachine> Drop for CloseOnExit<S> {
    fn drop(&mut self) {
        drop(self.0.close());
    }
}

// ─────────────────────── côté boucle (un seul thread) ───────────────────────

struct Core<S: StateMachine> {
    engine: RefCell<Engine<S>>,
    shared: Arc<Shared<S>>,
    /// Dernier cycle durable (msync fait).
    durable: Cell<u64>,
    /// Tâches qui attendent la prochaine barrière de durabilité.
    waiters: RefCell<Vec<Waker>>,
    /// Réponses retenues jusqu'au prochain msync.
    deferred: RefCell<Vec<Deferred>>,
    /// Tâches spawnées pendant ce tour, à adopter par l'exécuteur.
    spawned: RefCell<Vec<LocalFuture>>,
}

impl<S: StateMachine> Core<S> {
    /// Barrière du lot : un msync pour tout ce qui a été appliqué pendant ce tour
    /// (commandes et tâches), puis les réponses retenues, puis le réveil des
    /// tâches qui attendaient. Une erreur ici est fatale (voir l'en-tête).
    fn commit_batch(&self) -> Result<()> {
        let last = self.engine.borrow().last_seq();
        let mut outcome: Result<()> = Ok(());
        if last > self.durable.get() {
            outcome = self.engine.borrow_mut().sync();
            if outcome.is_ok() {
                self.durable.set(last);
                self.shared.durable.store(last, Ordering::Release);
                self.shared.syncs.fetch_add(1, Ordering::Relaxed);
            }
        }
        let deferred = mem::take(&mut *self.deferred.borrow_mut());
        let failure = outcome.as_ref().err();
        for done in deferred {
            done(failure);
        }
        if outcome.is_ok() {
            let waiters = mem::take(&mut *self.waiters.borrow_mut());
            for waker in waiters {
                waker.wake();
            }
        }
        outcome
    }

    /// Snapshot périodique, après les réponses (les clients n'en paient pas la
    /// latence). Un échec n'est pas fatal, le journal fait foi : on le compte et
    /// on réessaie `backoff` cycles plus tard.
    fn snapshot_if_due(&self, retry_after: &mut u64, backoff: u64) {
        let mut engine = self.engine.borrow_mut();
        let last = engine.last_seq();
        if last >= *retry_after && engine.maybe_snapshot().is_err() {
            self.shared.snapshot_errors.fetch_add(1, Ordering::Relaxed);
            *retry_after = last + backoff;
        }
    }
}

/// Accès à la boucle depuis une tâche qui s'y exécute. Jamais `Send` : la boucle
/// et ses tâches vivent sur un seul thread. Une tâche ne doit jamais bloquer
/// (`Pending::wait`, `sleep`, I/O bloquant) : elle bloquerait toute la base.
pub struct Ctx<S: StateMachine> {
    core: Rc<Core<S>>,
}

impl<S: StateMachine> Clone for Ctx<S> {
    fn clone(&self) -> Self {
        Ctx { core: self.core.clone() }
    }
}

impl<S: StateMachine + 'static> Ctx<S> {
    /// Journalise et applique l'événement, puis attend qu'il soit durable (le
    /// msync du lot en cours, partagé avec tous les autres) avant de rendre son cycle.
    pub async fn submit(&self, event: &S::Event) -> Result<u64> {
        let seq = self.core.engine.borrow_mut().submit(event)?;
        Durable { core: self.core.clone(), seq }.await
    }

    /// Lit l'état *courant*. Il peut contenir des cycles pas encore durables ;
    /// `submit(..).await` n'a pas ce défaut.
    pub fn read<R>(&self, f: impl FnOnce(&S) -> R) -> R {
        f(self.core.engine.borrow().state())
    }

    /// Lance une tâche sur la boucle ; elle est interrogée au tour suivant.
    pub fn spawn(&self, task: impl Future<Output = ()> + 'static) {
        self.core.spawned.borrow_mut().push(Box::pin(task));
    }

    /// Un `Handle` (clonable, `Send`) vers cette même boucle.
    pub fn handle(&self) -> Handle<S> {
        Handle { shared: self.core.shared.clone() }
    }

    fn defer(&self, f: impl FnOnce(Option<&Error>) + 'static) {
        self.core.deferred.borrow_mut().push(Box::new(f));
    }
}

/// Se résout quand le cycle `seq` est durable.
struct Durable<S: StateMachine> {
    core: Rc<Core<S>>,
    seq: u64,
}

impl<S: StateMachine> Future for Durable<S> {
    type Output = Result<u64>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<u64>> {
        if self.core.durable.get() >= self.seq {
            Poll::Ready(Ok(self.seq))
        } else {
            self.core.waiters.borrow_mut().push(cx.waker().clone());
            Poll::Pending
        }
    }
}

// ───────────────────────────── exécuteur maison ─────────────────────────────

struct Task<S: StateMachine> {
    fut: LocalFuture,
    flag: Arc<TaskWaker<S>>,
    waker: Waker,
}

/// Dalle de tâches `!Send`. Les réveils passent par la file partagée : un tour
/// de boucle de plus pour un réveil local, mais un seul chemin pour tous les
/// réveils (les sockets de la partie 5 s'y brancheront).
struct Executor<S: StateMachine> {
    slots: Vec<Option<Task<S>>>,
    free: Vec<usize>,
}

impl<S: StateMachine + 'static> Executor<S> {
    fn new() -> Self {
        Executor { slots: Vec::new(), free: Vec::new() }
    }

    /// Adopte les tâches spawnées pendant le tour ; leur première interrogation
    /// passe par la file de réveil.
    fn admit(&mut self, ctx: &Ctx<S>) {
        let incoming = mem::take(&mut *ctx.core.spawned.borrow_mut());
        for fut in incoming {
            let id = self.free.pop().unwrap_or(self.slots.len());
            let flag = Arc::new(TaskWaker {
                id,
                queued: AtomicBool::new(false),
                shared: ctx.core.shared.clone(),
            });
            let waker = Waker::from(flag.clone());
            waker.wake_by_ref();
            let task = Some(Task { fut, flag, waker });
            if id == self.slots.len() {
                self.slots.push(task);
            } else {
                self.slots[id] = task;
            }
        }
    }

    fn poll_task(&mut self, id: usize) {
        // Numéro périmé (tâche terminée) : ignoré. Réveil en trop : sans effet.
        let Some(Some(task)) = self.slots.get_mut(id) else { return };
        task.flag.queued.store(false, Ordering::Release);
        let mut cx = Context::from_waker(&task.waker);
        if task.fut.as_mut().poll(&mut cx).is_ready() {
            self.slots[id] = None;
            self.free.push(id);
        }
    }
}

// ───────────────────────────── API pour les clients ─────────────────────────────

/// Compteurs de la boucle, lisibles sans verrou depuis n'importe quel thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Group commits (msync) effectués.
    pub syncs: u64,
    /// Dernier cycle durable.
    pub durable_seq: u64,
    /// Snapshots périodiques échoués (le journal reste intact).
    pub snapshot_errors: u64,
}

/// Poignée vers la boucle : `Clone + Send + Sync`, quel que soit `S`.
pub struct Handle<S: StateMachine> {
    shared: Arc<Shared<S>>,
}

impl<S: StateMachine> Clone for Handle<S> {
    fn clone(&self) -> Self {
        Handle { shared: self.shared.clone() }
    }
}

impl<S: StateMachine + 'static> Handle<S> {
    /// Journalise et applique l'événement. La réponse (son cycle) n'arrive
    /// qu'une fois le journal durable. L'encodage postcard se fait sur le thread
    /// appelant : la boucle ne fait que décoder et appliquer.
    pub fn submit(&self, event: &S::Event) -> Pending<u64> {
        let bytes = match postcard::to_allocvec(event) {
            Ok(bytes) => bytes,
            Err(e) => return Pending::failed(e.into()),
        };
        self.call(move |ctx, tx| {
            let applied = ctx.core.engine.borrow_mut().submit_encoded(&bytes);
            match applied {
                Ok(seq) => ctx.defer(reply_when_durable(tx, seq)),
                // Refusé avant journalisation : rien à rendre durable.
                Err(e) => {
                    let _ = tx.send(Err(e));
                }
            }
        })
    }

    /// Exécute `f` sur la boucle, contre l'état courant. Les commandes sont
    /// traitées dans l'ordre de dépôt : une lecture voit les écritures déposées
    /// avant elle. `f` doit être brève (elle bloque tout le monde).
    pub fn read<R: Send + 'static>(
        &self,
        f: impl FnOnce(&S) -> R + Send + 'static,
    ) -> Pending<R> {
        self.call(move |ctx, tx| {
            let value = f(ctx.core.engine.borrow().state());
            ctx.defer(reply_when_durable(tx, value));
        })
    }

    /// Time-travel : exécute `f` sur l'état reconstruit à la fin du cycle `seq`.
    /// Le rejeu a lieu sur la boucle, qu'il bloque pendant sa durée (outil de debug).
    pub fn read_at<R: Send + 'static>(
        &self,
        seq: u64,
        f: impl FnOnce(&S) -> R + Send + 'static,
    ) -> Pending<R> {
        self.call(move |ctx, tx| {
            let replayed = ctx.core.engine.borrow().state_at(seq);
            match replayed {
                Ok(state) => ctx.defer(reply_when_durable(tx, f(&state))),
                Err(e) => {
                    let _ = tx.send(Err(e));
                }
            }
        })
    }

    /// Lance une tâche sur la boucle. `make` s'exécute *sur le thread de la boucle*
    /// et peut produire une future non-`Send`. `Err(Closed)` si la boucle est arrêtée.
    pub fn spawn<F, Fut>(&self, make: F) -> Result<()>
    where
        F: FnOnce(Ctx<S>) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + 'static,
    {
        let posted = self.shared.post(move |ctx: &Ctx<S>| ctx.spawn(make(ctx.clone())));
        if posted {
            Ok(())
        } else {
            Err(Error::Closed)
        }
    }

    pub fn stats(&self) -> Stats {
        Stats {
            syncs: self.shared.syncs.load(Ordering::Relaxed),
            durable_seq: self.shared.durable.load(Ordering::Acquire),
            snapshot_errors: self.shared.snapshot_errors.load(Ordering::Relaxed),
        }
    }

    fn call<T: Send + 'static>(
        &self,
        job: impl FnOnce(&Ctx<S>, Sender<Result<T>>) + Send + 'static,
    ) -> Pending<T> {
        let (tx, rx) = oneshot::channel();
        // Boucle fermée : la commande est détruite, `tx` avec elle, et `Pending`
        // se résout en `Err(Closed)`.
        self.shared.post(move |ctx: &Ctx<S>| job(ctx, tx));
        Pending { rx }
    }
}

/// Réponse retenue jusqu'au msync du lot (jamais de lecture sale).
fn reply_when_durable<T: 'static>(
    tx: Sender<Result<T>>,
    value: T,
) -> impl FnOnce(Option<&Error>) + 'static {
    move |failure: Option<&Error>| {
        let _ = tx.send(match failure {
            None => Ok(value),
            Some(e) => Err(replicate(e)),
        });
    }
}

/// `Error` n'est pas `Clone` (`io::Error`) : copie pour répondre à tout un lot.
fn replicate(e: &Error) -> Error {
    match e {
        Error::Io(e) => Error::Io(io::Error::new(e.kind(), e.to_string())),
        other => Error::Io(io::Error::new(io::ErrorKind::Other, other.to_string())),
    }
}

/// Réponse à venir d'une commande. C'est une `Future` (à `.await` dans n'importe
/// quel exécuteur, y compris [`block_on`]) ; `wait` bloque le thread courant.
#[must_use = "la commande part même si la réponse est ignorée, mais son échec le serait aussi"]
pub struct Pending<T> {
    rx: oneshot::Receiver<Result<T>>,
}

impl<T> Pending<T> {
    fn failed(e: Error) -> Self {
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(Err(e));
        Pending { rx }
    }

    /// Bloque le thread courant jusqu'à la réponse. Jamais depuis la boucle ni une
    /// de ses tâches (elle s'attendrait elle-même) : utilise `.await`.
    pub fn wait(self) -> Result<T> {
        self.rx.wait().unwrap_or(Err(Error::Closed))
    }
}

impl<T> Future for Pending<T> {
    type Output = Result<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<T>> {
        match Pin::new(&mut self.rx).poll(cx) {
            Poll::Ready(reply) => Poll::Ready(reply.unwrap_or(Err(Error::Closed))),
            Poll::Pending => Poll::Pending,
        }
    }
}

struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// Exécuteur minimal : pilote `fut` sur le thread courant, qui dort entre deux
/// réveils. Suffit à composer plusieurs `.await` de [`Pending`] sans runtime externe.
pub fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = std::pin::pin!(fut);
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(out) = fut.as_mut().poll(&mut cx) {
            return out;
        }
        thread::park();
    }
}

// ───────────────────────────── serveur ─────────────────────────────

/// Possède le thread de la boucle. Lâché sans `shutdown`, il s'arrête proprement
/// (en ignorant les erreurs).
pub struct Server<S: StateMachine> {
    handle: Handle<S>,
    thread: Option<JoinHandle<Result<()>>>,
}

impl<S: StateMachine + 'static> Server<S> {
    /// Démarre la boucle. Le moteur est ouvert *sur le thread de la boucle* (rejeu
    /// du journal compris) ; les erreurs d'ouverture remontent ici.
    pub fn start(path: impl Into<PathBuf>, opts: Options) -> Result<(Self, Recovery)> {
        let path = path.into();
        let shared = Arc::new(Shared::<S>::new()?);
        let epoll = Epoll::new()?;
        epoll.add(shared.wake.as_raw_fd(), WAKE, EPOLLIN)?;

        let (ready, started) = oneshot::channel::<Result<Recovery>>();
        let thread = {
            let shared = shared.clone();
            thread::Builder::new()
                .name("chronicle-loop".into())
                .spawn(move || run::<S>(path, opts, shared, epoll, ready))?
        };
        match started.wait() {
            Ok(Ok(recovery)) => {
                Ok((Server { handle: Handle { shared }, thread: Some(thread) }, recovery))
            }
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            // Le thread est mort avant de répondre.
            Err(_) => match thread.join() {
                Err(payload) => std::panic::resume_unwind(payload),
                Ok(Err(e)) => Err(e),
                Ok(Ok(())) => Err(Error::Closed),
            },
        }
    }

    pub fn handle(&self) -> Handle<S> {
        self.handle.clone()
    }

    /// Arrêt propre : traite les commandes déjà déposées, rend leurs réponses
    /// durables, écrit un snapshot si utile, libère le journal (verrou compris),
    /// puis rend l'erreur fatale qui aurait arrêté la boucle plus tôt. Les tâches
    /// encore en vol sont abandonnées ; tout ce qui a été acquitté est durable.
    pub fn shutdown(mut self) -> Result<()> {
        self.stop()
    }

    fn stop(&mut self) -> Result<()> {
        let Some(thread) = self.thread.take() else { return Ok(()) };
        self.handle.shared.request_close();
        match thread.join() {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
}

impl<S: StateMachine> Drop for Server<S> {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            self.handle.shared.request_close();
            let _ = thread.join(); // une panique de la boucle ne se propage pas depuis Drop
        }
    }
}

fn run<S: StateMachine + 'static>(
    path: PathBuf,
    opts: Options,
    shared: Arc<Shared<S>>,
    epoll: Epoll,
    ready: Sender<Result<Recovery>>,
) -> Result<()> {
    let backoff = opts.snapshot_every.max(1);
    let (engine, recovery) = match Engine::<S>::open_with(&path, opts) {
        Ok(opened) => opened,
        Err(e) => {
            let _ = ready.send(Err(e));
            return Ok(());
        }
    };
    // Déclaré avant `ctx` : détruit après lui (moteur libéré, verrou relâché).
    let _closer = CloseOnExit(shared.clone());
    let last = engine.last_seq();
    shared.durable.store(last, Ordering::Release);
    let ctx = Ctx {
        core: Rc::new(Core {
            engine: RefCell::new(engine),
            shared: shared.clone(),
            durable: Cell::new(last),
            waiters: RefCell::new(Vec::new()),
            deferred: RefCell::new(Vec::new()),
            spawned: RefCell::new(Vec::new()),
        }),
    };
    let _ = ready.send(Ok(recovery));

    let mut tasks = Executor::<S>::new();
    let mut tokens = Vec::new();
    let mut retry_after = 0;
    loop {
        tokens.clear();
        epoll.wait(&mut tokens, None)?;
        // Vider le réveil *avant* de prendre la file : une commande déposée
        // entre les deux est traitée ce tour-ci, le réveil en trop est inoffensif.
        shared.wake.drain();
        let batch = shared.take();

        for job in batch.jobs {
            job(&ctx);
        }
        tasks.admit(&ctx);
        for id in batch.woken {
            tasks.poll_task(id);
        }
        tasks.admit(&ctx);

        ctx.core.commit_batch()?;
        ctx.core.snapshot_if_due(&mut retry_after, backoff);

        if batch.closed {
            // Arrêt propre : snapshot final pour un redémarrage instantané
            // (échec ignoré : un snapshot n'est qu'un cache).
            let mut engine = ctx.core.engine.borrow_mut();
            if engine.snapshots().last() != Some(&engine.last_seq()) {
                let _ = engine.snapshot();
            }
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::{Kv, KvEvent};

    #[test]
    fn submit_read_and_shutdown_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kv.log");
        let (server, recovery) = Server::<Kv>::start(&path, Options::default()).unwrap();
        assert_eq!(recovery.frames, 0);
        let h = server.handle();

        let put = KvEvent::Put { key: b"a".to_vec(), value: b"1".to_vec() };
        assert_eq!(h.submit(&put).wait().unwrap(), 1);
        let got = h.read(|kv| kv.get(b"a").map(<[u8]>::to_vec)).wait().unwrap();
        assert_eq!(got, Some(b"1".to_vec()));
        assert_eq!(h.stats().durable_seq, 1);

        server.shutdown().unwrap();
        assert!(matches!(h.submit(&put).wait(), Err(Error::Closed)));
    }
}