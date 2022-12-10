#![feature(portable_simd)]

use std::simd::{f32x16, SimdFloat};
use std::time::Instant;

const VEC_LEN: usize = 10000000;
const TEST_ROUND: usize = 1000;

fn main() {
    let vec_store = vec![1.0f32; VEC_LEN];
    let mut vec_store_2 = vec![1.0f32; VEC_LEN];
    let vec = vec_store.as_slice();
    let vec2 = vec_store_2.as_mut_slice();
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

    // iter copy
    for _ in 0..TEST_ROUND {
        iter_copy(vec, vec2);
    }
    let t_end = Instant::now();

    println!("iter copy:   {}ms", (t_end - t_start).as_millis());

    // slice copy
    for _ in 0..TEST_ROUND {
        slice_copy(vec, vec2);
    }
    let t_end = Instant::now();

    println!("slice copy:  {}ms", (t_end - t_start).as_millis());

    // simd copy
    for _ in 0..TEST_ROUND {
        simd_copy(vec, vec2);
    }
    let t_end = Instant::now();

    println!("simd copy:   {}ms", (t_end - t_start).as_millis());
}

/// iter sum
fn iter_sum(vec: &[f32]) -> f32 {
    vec.iter().sum::<f32>()
}

fn iter_copy(vec: &[f32], vec2: &mut [f32]) {
    vec.iter().zip(vec2.iter_mut()).for_each(|(v, v2)| *v2 = *v);
}

/// for loop sum
fn for_loop_sum(vec: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in vec {
        s += i;
    }
    s
}

/// memcpy
fn slice_copy(vec: &[f32], vec2: &mut [f32]) {
    vec2.copy_from_slice(vec);
}

/// for loop sum
fn simd_copy(vec: &[f32], vec2: &mut [f32]) {
    if vec.len() != vec2.len() {
        panic!("err, unequal length");
    }
    let mut ind = 0usize;
    let (slow0, simd, slow1) = vec2.as_simd_mut::<16>();
    for i in slow0 {
        *i = unsafe { *vec.get_unchecked(ind) };
        ind += 1;
    }
    for ix16 in simd {
        *ix16 = f32x16::from_slice(&vec[ind..]);
        ind += 16;
    }
    for i in slow1 {
        *i = unsafe { *vec.get_unchecked(ind) };
        ind += 1;
    }
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
