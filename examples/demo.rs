use chronicle::kv::{Kv, KvEvent};
use chronicle::{Engine, Result};

fn main() -> Result<()> {
    let (mut db, recovery) = Engine::<Kv>::open("demo.chr")?;
    println!("démarrage : {recovery:?}, dernier cycle = {}", db.last_seq());

    let n = db.last_seq() + 1;
    db.submit(&KvEvent::Put { key: b"cycle".to_vec(), value: n.to_string().into_bytes() })?;
    db.commit()?;

    let before = db.state_at(n - 1)?; // état exact au cycle précédent
    println!("maintenant : {:?}", db.state().get(b"cycle").map(String::from_utf8_lossy));
    println!("cycle {} : {:?}", n - 1, before.get(b"cycle").map(String::from_utf8_lossy));
    Ok(())
}