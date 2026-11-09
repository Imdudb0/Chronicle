//! Machine à états d'exemple : magasin clé-valeur ordonné.
//! `BTreeMap` (et non `HashMap`) : ordre d'itération et sérialisation déterministes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::StateMachine;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvEvent {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Kv {
    map: BTreeMap<Vec<u8>, Vec<u8>>,
}

impl Kv {
    pub fn get(&self, key: &[u8]) -> Option<&[u8]> {
        self.map.get(key).map(Vec::as_slice)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &[u8])> {
        self.map.iter().map(|(k, v)| (k.as_slice(), v.as_slice()))
    }
}

impl StateMachine for Kv {
    type Event = KvEvent;

    fn genesis() -> Self {
        Self::default()
    }

    fn apply(&mut self, event: &KvEvent) {
        match event {
            KvEvent::Put { key, value } => {
                self.map.insert(key.clone(), value.clone());
            }
            KvEvent::Delete { key } => {
                self.map.remove(key);
            }
        }
    }
}