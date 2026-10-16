use bytemuck::{Pod, Zeroable};
use core::ops::Mul;

#[derive(Copy, Clone, Default, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct F16(pub u16);

unsafe impl Zeroable for F16 {}
unsafe impl Pod for F16 {}

impl F16 {
    pub const ZERO: Self = Self(0);

    pub const fn to_f32(self) -> f32 {
        let u = self.0;
        let sign = ((u >> 15) as u32) << 31;
        let exp = ((u >> 10) & 0x1F) as u32;
        let mant = (u & 0x3FF) as u32;

        if exp == 0 {
            f32::from_bits(sign)
        } else if exp == 0x1F {
            f32::from_bits(sign | (0xFF << 23) | (mant << 13))
        } else {
            f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
        }
    }

    pub const fn from_f32(val: f32) -> Self {
        let u = val.to_bits();
        let sign = ((u >> 31) & 1) as u16;
        let exp = ((u >> 23) & 0xFF) as i32;
        let mant = (u & 0x7FFFFF) as u32;

        let new_exp = exp - 127 + 15;

        if exp == 0 {
            Self(sign << 15)
        } else if exp == 0xFF {
            Self((sign << 15) | (0x1F << 10) | ((mant >> 13) as u16))
        } else if new_exp >= 31 {
            Self((sign << 15) | (0x1F << 10))
        } else if new_exp <= 0 {
            Self(sign << 15)
        } else {
            let e16 = (new_exp as u16) << 10;
            let m16 = (mant >> 13) as u16;
            Self((sign << 15) | e16 | m16)
        }
    }
}

impl Mul<f32> for F16 {
    type Output = f32;
    fn mul(self, rhs: f32) -> f32 {
        self.to_f32() * rhs
    }
}
