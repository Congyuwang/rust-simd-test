#![feature(portable_simd)]
#![feature(iter_array_chunks)]
#![feature(array_chunks)]

use std::simd::{Simd, SimdFloat, StdFloat};
use std::time::Instant;

const VEC_LEN: usize = 10000000;
const TEST_ROUND: usize = 1000;
const LANE: usize = 16;

pub fn main() {
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
    t_start = t_end;

    // iter copy
    for _ in 0..TEST_ROUND {
        iter_copy(vec, vec2);
    }
    let t_end = Instant::now();

    println!("iter copy:   {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // slice copy
    for _ in 0..TEST_ROUND {
        slice_copy(vec, vec2);
    }
    let t_end = Instant::now();

    println!("slice copy:  {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // simd copy
    for _ in 0..TEST_ROUND {
        simd_copy(vec, vec2);
    }
    let t_end = Instant::now();

    println!("Bad idea, leave this to _memcopy intrinsic");
    println!("simd copy:   {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // iter neg
    for _ in 0..TEST_ROUND {
        iter_neg(vec2);
    }
    let t_end = Instant::now();

    println!("the compiler already uses SIMD for this:");
    println!("iter neg:    {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // simd neg
    for _ in 0..TEST_ROUND {
        simd_neg(vec2);
    }
    let t_end = Instant::now();

    println!("simd neg:    {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // simd sub
    for _ in 0..TEST_ROUND {
        simd_sub(vec2, vec);
    }
    let t_end = Instant::now();

    println!("simd sub:    {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // iter neg
    for _ in 0..TEST_ROUND {
        iter_sub(vec, vec2);
    }
    let t_end = Instant::now();

    println!("the compiler already uses SIMD for this:");
    println!("iter sub:    {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // simd sub
    for _ in 0..TEST_ROUND {
        axpby_simd(2.0, vec, -2.0, vec2);
    }
    let t_end = Instant::now();

    println!("simd axpby:  {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // simd sub
    for _ in 0..TEST_ROUND {
        axpby_simd_2(2.0, vec, -2.0, vec2);
    }
    let t_end = Instant::now();

    println!("simd axpby2: {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // iter neg
    for _ in 0..TEST_ROUND {
        axpby_iter(2.0, vec, -2.0, vec2);
    }
    let t_end = Instant::now();

    println!("this compiler is already doing SIMD for this");
    println!("iter axpby:  {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // iter norm
    let mut o = 0.0;
    for _ in 0..TEST_ROUND {
        o = iter_norm_squared(vec);
    }
    let t_end = Instant::now();

    println!("iter norm:   {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // simd norm
    for _ in 0..TEST_ROUND {
        o = simd_norm_squared(vec);
    }
    let t_end = Instant::now();

    println!("simd norm:   {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // iter dot
    for _ in 0..TEST_ROUND {
        o = iter_dot(vec, vec2);
    }
    let t_end = Instant::now();

    println!("iter dot:    {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // simd dot
    for _ in 0..TEST_ROUND {
        o = simd_dot(vec, vec2);
    }
    let t_end = Instant::now();

    println!("simd dot:    {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // iter metric
    for _ in 0..TEST_ROUND {
        o = iter_metric(vec, vec2);
    }
    let t_end = Instant::now();

    println!("iter metric: {}ms", (t_end - t_start).as_millis());
    t_start = t_end;

    // simd metric
    for _ in 0..TEST_ROUND {
        o = simd_metric(vec, vec2);
    }
    let t_end = Instant::now();

    println!("simd metric: {}ms", (t_end - t_start).as_millis());
    println!("{o}");
}

/// iter sum
fn iter_sum(vec: &[f32]) -> f32 {
    vec.iter().sum::<f32>()
}

fn iter_copy(vec: &[f32], vec2: &mut [f32]) {
    vec2.iter_mut().zip(vec).for_each(|(v, v2)| *v = *v2);
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
/// Bad idea: just use memcpy!
fn simd_copy(vec: &[f32], vec2: &mut [f32]) {
    if vec.len() != vec2.len() {
        panic!("err, unequal length");
    }
    let (slow0, simd, slow1) = vec2.as_simd_mut::<LANE>();
    let len0 = slow0.len();
    let len1 = len0 + simd.len() * 16;
    slow0.iter_mut().zip(vec.iter()).for_each(|(i, j)| *i = *j);
    simd.iter_mut()
        .zip(vec[len0..].array_chunks::<LANE>())
        .for_each(|(i, j)| *i = Simd::from_array(*j));
    slow1
        .iter_mut()
        .zip(vec[len1..].iter())
        .for_each(|(i, j)| *i = *j);
}

/// simd sum
fn simd_sum(vec: &[f32]) -> f32 {
    let mut s = 0.0f32;
    let (slow0, simd, slow1) = vec.as_simd::<LANE>();
    for i in slow0 {
        s += i;
    }
    let mut simd0 = Simd::<f32, LANE>::splat(0.0);
    for ix16 in simd {
        simd0 += ix16;
    }
    s += simd0.reduce_sum();
    for i in slow1 {
        s += i;
    }
    s
}

#[inline(always)]
pub fn iter_norm_squared(v: &[f32]) -> f32 {
    v.iter().map(|v| v * v).sum()
}

#[inline(always)]
pub fn simd_norm_squared(v: &[f32]) -> f32 {
    let mut norm = 0.0f32;
    let mut norm_sim = Simd::<f32, 16>::splat(0.0f32);
    let (v0, v_sim, v1) = v.as_simd::<LANE>();
    for i in v0 {
        norm = i.mul_add(*i, norm);
    }
    for i in v1 {
        norm = i.mul_add(*i, norm);
    }
    for i_sim in v_sim {
        norm_sim = i_sim.mul_add(*i_sim, norm_sim);
    }
    norm += norm_sim.reduce_sum();
    norm
}

pub fn iter_neg(x: &mut [f32]) {
    x.iter_mut().for_each(|x| *x = -*x);
}

/// memory bound
pub fn simd_neg(x: &mut [f32]) {
    let (x0, x_sim, x1) = x.as_simd_mut::<LANE>();
    for i in x0 {
        *i = -*i;
    }
    for i_sim in x_sim {
        // note that 16 is hard coded here
        *i_sim = -*i_sim;
    }
    for i in x1 {
        *i = -*i;
    }
}

#[inline(always)]
fn iter_sub(vec: &[f32], vec2: &mut [f32]) {
    // vec.iter()
    //     .zip(vec2.iter_mut())
    //     .for_each(|(v, v2)| *v2 -= *v);
    for i in 0..vec2.len() {
        vec2[i] -= vec[i];
    }
}

pub fn simd_sub(x: &mut [f32], c: &[f32]) {
    let (slow0, simd, slow1) = x.as_simd_mut::<LANE>();
    let len0 = slow0.len();
    let len1 = len0 + simd.len() * 16;
    let (c0, c_sim, c1) = (
        &c[..len0],
        c[len0..len1]
            .array_chunks()
            .map(|&b| Simd::<f32, LANE>::from_array(b)),
        &c[len1..],
    );
    slow0.iter_mut().zip(c0).for_each(|(i, j)| *i -= j);
    simd.iter_mut().zip(c_sim).for_each(|(i, j)| *i -= j);
    slow1.iter_mut().zip(c1).for_each(|(i, j)| *i -= j);
}

pub fn iter_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

pub fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let (a0, a_sim, a1) = a.as_simd::<LANE>();
    dot += a_sim
        .iter()
        .zip(
            b[a0.len()..]
                .array_chunks::<LANE>()
                .map(|&b| Simd::<f32, LANE>::from_array(b)),
        )
        .fold(Simd::<f32, LANE>::splat(0.0f32), |acc, (a, b)| {
            a.mul_add(b, acc)
        })
        .reduce_sum();
    dot += iter_dot(a0, &b);
    dot += iter_dot(a1, &b[a0.len() + a_sim.len() * LANE..]);
    dot
}

#[inline(always)]
pub fn iter_metric(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter().copied()).fold(0.0, |acc, (a, b)| {
        let d = a - b;
        d.mul_add(d, acc)
    })
}

pub fn simd_metric(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let (a0, a_sim, a1) = a.as_simd::<LANE>();
    let p0 = a0.len();
    let p1 = p0 + a_sim.len() * LANE;
    let (b0, b_sim, b1) = (
        &b[..p0],
        b[p0..p1]
            .array_chunks()
            .map(|&b| Simd::<f32, LANE>::from_array(b)),
        &b[p1..],
    );
    dot += a_sim
        .iter()
        .zip(b_sim)
        .fold(Simd::<f32, LANE>::splat(0.0f32), |acc, (a, b)| {
            let d = a - b;
            d.mul_add(d, acc)
        })
        .reduce_sum();
    dot += iter_metric(a0, b0);
    dot += iter_metric(a1, b1);
    dot
}

/// y <- a * x + b * y
#[inline(always)]
pub fn axpby_simd(a: f32, x: &[f32], b: f32, y: &mut [f32]) {
    let mut ind = 0usize;
    let (y0, y_sim, y1) = y.as_simd_mut::<LANE>();
    let a_sim = Simd::<f32, 16>::splat(a);
    let b_sim = Simd::<f32, 16>::splat(b);
    unsafe {
        for i in y0 {
            *i = i.mul_add(b, a * x.get_unchecked(ind));
            ind += 1;
        }
        for i_sim in y_sim {
            // note that 16 is hard coded here
            *i_sim = i_sim.mul_add(b_sim, a_sim * Simd::<f32, 16>::from_slice(&x[ind..]));
            ind += LANE;
        }
        for i in y1 {
            *i = i.mul_add(b, a * x.get_unchecked(ind));
            ind += 1;
        }
    }
}

/// y <- a * x + b * y
#[inline(always)]
pub fn axpby_simd_2(a: f32, x: &[f32], b: f32, y: &mut [f32]) {
    let (y0, y_sim, y1) = y.as_simd_mut::<LANE>();
    let len0 = y0.len();
    let len1 = len0 + y_sim.len() * LANE;
    let (x0, x_sim, x1) = (
        &x[..len0],
        x[len0..len1]
            .array_chunks()
            .map(|&b| Simd::<f32, LANE>::from_array(b)),
        &x[len1..],
    );
    let a_sim = Simd::<f32, 16>::splat(a);
    let b_sim = Simd::<f32, 16>::splat(b);
    y_sim
        .iter_mut()
        .zip(x_sim)
        .for_each(|(y, x)| *y = y.mul_add(b_sim, a_sim * x));
    y0.iter_mut()
        .zip(x0)
        .for_each(|(y, x)| *y = y.mul_add(b, a * x));
    y1.iter_mut()
        .zip(x1)
        .for_each(|(y, x)| *y = y.mul_add(b, a * x));
}

#[inline(always)]
pub fn axpby_iter(a: f32, x: &[f32], b: f32, y: &mut [f32]) {
    y.iter_mut()
        .zip(x.iter())
        .for_each(|(y, x)| *y = y.mul_add(b, a * x))
}
