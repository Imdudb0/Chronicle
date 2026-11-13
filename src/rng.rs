// xoshiro256** + splitmix64, zéro dépendance.
#[derive(Clone)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        let s = [
            splitmix64(&mut x),
            splitmix64(&mut x),
            splitmix64(&mut x),
            splitmix64(&mut x),
        ];
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Entier uniforme dans [0, n) (n > 0).
    pub fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * (n as u64)) >> 32) as usize
    }

    /// Entier uniforme dans [lo, hi] (bornes incluses).
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    /// Gaussienne standard (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.f64().max(1e-12);
        let u2 = self.f64();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    /// Tirage catégoriel selon des poids positifs (non normalisés).
    pub fn categorical(&mut self, w: &[f64]) -> usize {
        let tot: f64 = w.iter().sum();
        let mut x = self.f64() * tot;
        for (i, &wi) in w.iter().enumerate() {
            if x < wi {
                return i;
            }
            x -= wi;
        }
        let mut last = 0;
        for (i, &wi) in w.iter().enumerate() {
            if wi > 0.0 {
                last = i;
            }
        }
        last
    }

    pub fn fork(&mut self) -> Rng {
        Rng::new(self.next_u64())
    }
}