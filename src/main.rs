#![allow(unused)]
#![feature(inline_const)]
#![feature(const_trait_impl)]

use std::arch::aarch64::*;

#[rustfmt::skip]
fn main() { unsafe { main_ (); } }
pub trait ToArray<T: Default, const N: usize> {
    fn to_array(self) -> [T; N];
}
impl<T: Default, const N: usize, I: Iterator<Item = T>> ToArray<T, N> for I {
    fn to_array(self) -> [T; N] {
        let mut x: [T; N] = std::array::from_fn(|_| Default::default());
        for (idx, elem) in self.enumerate() {
            x[idx] = elem;
        }
        x
    }
}
/// Transposes an array into a table for vectorized lookups
pub trait TransposeTable<T, const N: usize> {
    fn transpose_table(&self, register_count: usize) -> Self;
}
impl<T: Clone, const N: usize> TransposeTable<T, N> for [T; N] {
    fn transpose_table(&self, register_count: usize) -> Self {
        std::array::from_fn(|i| {
            let offset = i / register_count;
            let idx = ((i % register_count) * 16 + offset);
            self[idx].clone()
        })
    }
}
unsafe fn printx<T>(t: T) {
    let x: u128 = unsafe { *std::mem::transmute::<&T, &u128>(&t) };
    let x: [u8; 16] = unsafe { *std::mem::transmute::<&T, &[u8; 16]>(&t) };
    for i in x {
        print!("{:0>2x} ", i);
    }
    println!("");
}
unsafe fn printb<T>(t: T) {
    let x: [u8; 16] = unsafe { *std::mem::transmute::<&T, &[u8; 16]>(&t) };
    for i in x {
        print!("{:0>8b} ", i);
    }
    println!("");
}
unsafe fn proc1() {
    let x = (0xffu64, 0xffffu64);
    println!("{:0>16x}\n{:0>16x}", x.0, x.1);
    let v1 = vld1q_u64(&x.0 as *const _);
    printx(v1);
}
// detect non-ASCII
unsafe fn proc2() {
    let ascii = "ln foo(x: u64, h^: * -> *) { /*  def */ }".as_bytes();
    let utf8 = "ln λ(x ∈ u64, h^: *→*) { /* φ */ }".as_bytes();

    let mask = &0b10000000;
    let bytes = ascii;

    if bytes.len() >= 16 {
        let v = vld1q_u8(bytes.as_ptr());
        let m = vld1q_dup_u8(mask);
        let res = vandq_u8(v, m);
        printb(v);
        printb(m);
        println!("result:");
        printb(res);
        // horizontal max to see if there are any non-zero bytes
        let x = vmaxvq_u8(res);
        if x != 0 {
            panic!("non_ascii detected! ");
        }
        println!("max: {:0>8b}", x);
    } else {
        panic!("nope");
    }
}
// table lookup
unsafe fn proc3() {
    let index_bytes: [u8; 16] = (0..16).map(|x| x / 4).to_array();
    let dst_bytes: [u8; 16] = (0..16).rev().to_array();

    let v1 = vld1q_u8(&index_bytes[0]);
    let dst = vld1q_u8(&dst_bytes[0]);
    let v2 = vqtbl1q_u8(dst, v1);
    printx(v1);
    printx(dst);
    println!("indexing:");
    printx(v2);
}
unsafe fn proc4() {
    //assume ascii (minimum 16 characters)
    let bytes = "ln xy = 13 + 37;".as_bytes();

    // this is a lookup table for ASCII conversions (WIP)
    // Note that the maximum 4 registers can hold is 64 bytes
    let mut conv_table1: [u8; 64] = (0..64).to_array().transpose_table(4);
    let mut conv_table2: [u8; 64] = (64..128).to_array().transpose_table(4);

    let v192 = vld1q_dup_u8(&0xc0);
    // read first 16 characters into register
    let v1 = vld1q_u8(&bytes[0]);
    /*
    00000000 11111111 11111111 00000000 00000000 11111111
    >>
    00000000 00000001 00000001 00000000 00000000 00000001

    alternative:
    00000000 11111111 11111111 00000000 00000000 11111111
    or
    00000001 00000000 00000000 00000001 00000001 00000000
    =
    00000001 11111111 11111111 00000001 00000001 11111111
    */
    // This shifts indices down by 64. This allows us to index exclusively
    // the upper 64 ASCII characters.
    let v1_upper = vaddq_u8(v1, v192); // v1[i] = v[i] - 64;

    // map lower ascii characters (indices >=64 get ignored)
    let conv1 = vld4q_u8(&conv_table1[0]);
    // map upper ascii characters (indices < 64 get ignored)
    let conv2 = vld4q_u8(&conv_table2[0]);

    let index_1 = vqtbl4q_u8(conv1, v1);
    let index_2 = vqtbl4q_u8(conv2, v1_upper);

    printx(v1);
    printx(index_1);
    printx(index_2);
}
unsafe fn proc5() {
    //assume ascii (minimum 16 characters)
    let bytes = "ln foo = 13 + 37".as_bytes();

    // look up tables
    let mut conv_table1: [u8; 64] = (0..64).to_array().transpose_table(4);
    let mut conv_table2: [u8; 64] = (64..128).to_array().transpose_table(4);
    conv_table1
        .iter_mut()
        .chain(conv_table2.iter_mut())
        .for_each(|c| {
            if c.is_ascii_alphanumeric() {
                *c = 0xff;
            }
        });
    let conv1 = vld4q_u8(&conv_table1[0]);
    let conv2 = vld4q_u8(&conv_table2[0]);
    /*
    Approach: overflow identifier recognition

    First shuffle:
    - 0 1 2 3 4 5 6 - 7 8 9 a b c d
    |               |
    +-------+-------+
            |
       carry holder
    */
    let vinput = vld1q_u8(&bytes[0]);
    let carry_holder_mask = vld1q_dup_u64(&0xffffffffffffff00u64);
    let carry_holder_mask = vreinterpretq_u8_u64(carry_holder_mask);

    let carry_shuffle: &[u8] = &[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
    // load shuffle mask
    let shuf1 = vld1q_u8(&carry_shuffle[0]);

    // perform lookup
    let v192 = vld1q_dup_u8(&0xc0);
    let v1_upper = vaddq_u8(vinput, v192); // v1[i] = v[i] - 64;
    let index_1 = vqtbl4q_u8(conv1, vinput);
    let index_2 = vqtbl4q_u8(conv2, v1_upper);
    let mapped = vorrq_u8(index_1, index_2);

    // apply shuffle
    let mapped_shuf = vqtbl1q_u8(mapped, shuf1);
    // zero out carry 
    let mapped_shuf = vandq_u8(mapped_shuf, carry_holder_mask);

    // check for equality
    let equality_map = vceqq_u8(mapped, mapped_shuf);
    let masked = veorq_u8(mapped, equality_map);
    /*
    s 1 1 s 1 1 1 s
      s 1 1 s 1 1 1 s
    s 1 0 s 1 0 0 s
    s 1 0 s 1 0 0 s
    */
    // look up in table
    printx(vinput);
    printx(mapped);
    printx(mapped_shuf);
    printx(equality_map);
    printx(masked);
}
unsafe fn main_() {
    proc5();
}
