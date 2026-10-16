use crate::f16::F16;
use glam::{Vec2, Vec4};

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn swiglu(gate: f32, up: f32) -> f32 {
    silu(gate) * up
}

pub fn dot4_strided_f16(
    weights_t: &[F16],
    ptr: usize,
    stride: usize,
    input: &[f32],
    c: usize,
) -> f32 {
    let w = Vec4::new(
        weights_t[ptr].to_f32(),
        weights_t[ptr + stride].to_f32(),
        weights_t[ptr + stride * 2].to_f32(),
        weights_t[ptr + stride * 3].to_f32(),
    );
    let inp = Vec4::new(input[c], input[c + 1], input[c + 2], input[c + 3]);
    w.dot(inp)
}

pub fn dot4_contiguous_f32(a: &[f32], a_idx: usize, b: &[f32], b_idx: usize) -> f32 {
    let va = Vec4::new(a[a_idx], a[a_idx + 1], a[a_idx + 2], a[a_idx + 3]);
    let vb = Vec4::new(b[b_idx], b[b_idx + 1], b[b_idx + 2], b[b_idx + 3]);
    va.dot(vb)
}

pub fn rope_rotate_2d(val0: f32, val1: f32, theta: f32) -> (f32, f32) {
    let cos = theta.cos();
    let sin = theta.sin();
    let v = Vec2::new(val0, val1);
    (v.x * cos - v.y * sin, v.x * sin + v.y * cos)
}
