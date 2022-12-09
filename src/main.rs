#![feature(portable_simd)]

use std::simd::{f32x16, SimdFloat};
use std::time::Instant;

const VEC_LEN: usize = 10000000;
const TEST_ROUND: usize = 1000;

fn main() {
    let vec_store = vec![1.0f32; VEC_LEN];
    let vec = vec_store.as_slice();
    let mut t_start = Instant::now();

    // iter sum
    for _ in 0..TEST_ROUND {
        assert_eq!(iter_sum(vec) as usize, VEC_LEN);
    }
    let t_end = Instant::now();

    println!("iter sum:    {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // for loop sum
    for _ in 0..TEST_ROUND {
        assert_eq!(for_loop_sum(vec) as usize, VEC_LEN);
    }
    let t_end = Instant::now();

    println!("loop sum:    {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // simd sum
    for _ in 0..TEST_ROUND {
        assert_eq!(simd_sum(vec) as usize, VEC_LEN);
    }
    let t_end = Instant::now();

    println!("simd sum:    {}ms", (t_end - t_start).as_millis());
}

/// iter sum
fn iter_sum(vec: &[f32]) -> f32 {
    vec.iter().sum::<f32>()
}

/// for loop sum
fn for_loop_sum(vec: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in vec {
        s += i;
    }
    s
}

/// simd sum
fn simd_sum(vec: &[f32]) -> f32 {
    let mut s = 0.0f32;
    let (slow0, simd, slow1) = vec.as_simd::<16>();
    for i in slow0 {
        s += i;
    }
    let mut simd0 = f32x16::splat(0.0);
    for ix16 in simd {
        simd0 += ix16;
    }
    s += simd0.reduce_sum();
    for i in slow1 {
        s += i;
    }
    s
}
