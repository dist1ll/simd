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
    for i in x.iter().rev() {
        print!("{:0>2x} ", i);
    }
    println!("");
}
unsafe fn printb<T>(t: T) {
    let x: [u8; 16] = unsafe { *std::mem::transmute::<&T, &[u8; 16]>(&t) };
    for i in x.iter().rev() {
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
    let bytes1 = "let x=0;\nfoo(1);".as_bytes();
    proc6(&bytes1[0]);
}

unsafe fn proc6(input: *const u8) {
    // look up tables
    let mut conv_table1: [u8; 64] = (0..64).to_array().transpose_table(4);
    let mut conv_table2: [u8; 64] = (64..128).to_array().transpose_table(4);
    conv_table1
        .iter_mut()
        .chain(conv_table2.iter_mut())
        .for_each(|c| {
            if c.is_ascii_alphabetic() {
                *c = 0xff;
            }
            if c.is_ascii_whitespace() {
                *c = 0xfd;
            }
            if c.is_ascii_digit() {
                *c = 0xfe;
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
    let vinput = vld1q_u8(input);
    let c = 0xffffffffffffffffffffffffffffff00u128;
    let carry_holder_mask = vld1q_u8(&c as *const _ as *const _);

    let carry_shuffle: &[u8] = &[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
    // load shuffle mask
    let shuf1 = vld1q_u8(&carry_shuffle[0]);

    // perform lookup
    let v192 = vld1q_dup_u8(&0xc0);
    let v1_upper = vaddq_u8(vinput, v192); // v1[i] = v[i] - 64;
    let index_1 = vqtbl4q_u8(conv1, vinput);
    let index_2 = vqtbl4q_u8(conv2, v1_upper);
    let mapped = vorrq_u8(index_1, index_2);

    printx(vinput);
    printx(v1_upper);
    printx(index_1);
    printx(index_2);
    printx(mapped);
    // apply shuffle
    let mapped_shuf = vqtbl1q_u8(mapped, shuf1);
    // zero out carry
    let mapped_shuf = vandq_u8(mapped_shuf, carry_holder_mask);

    // check for equality
    let equality_map = vceqq_u8(mapped, mapped_shuf);
    // ignore equality check for multi-char tokens (i.e. char map >= 128)
    let v128 = vld1q_dup_u8(&0x80);
    let greater_than = vcgtq_u8(mapped, v128);
    let equality_map = vmvnq_u8(vandq_u8(greater_than, equality_map));

    // xor the result with your mask
    let masked = vandq_u8(mapped, equality_map);
    /*
    s 1 1 s 1 1 1 s
      s 1 1 s 1 1 1 s
    s 1 0 s 1 0 0 s
    s 1 0 s 1 0 0 s
    */
    // look up in table
}
unsafe fn proc7() {
    println!("hello world");
    let c = 0xff00ff00ff00ff00ff00ff00ff00ff00u128;
    let zero = vld1q_u8(&0u128 as *const _ as *const _);
    let c = vld1q_u8(&c as *const _ as *const _);
    let shift = vsraq_n_u8::<7>(zero, c);
    let shift = vreinterpretq_u16_u8(shift);
    let acc16 = vreinterpretq_u32_u16(vsraq_n_u16::<7>(shift, shift));
    let acc32 = vreinterpretq_u64_u32(vsraq_n_u32::<14>(acc16, acc16));
    let acc64 = vreinterpretq_u8_u64(vsraq_n_u64::<28>(acc32, acc32));
    let result_low = vgetq_lane_u8::<0>(acc64);
    let result_high = vgetq_lane_u8::<8>(acc64);
    println!("{:b}", result_low);
    println!("{:b}", result_high);
    printb(c);
    printb(shift);
    printb(acc16);
    printb(acc32);
    printb(acc64);
    println!(
        "Result: {:b}",
        (result_low as u16 + ((result_high as u16) << 8))
    );
}
fn proc8_transform(idx: u8) -> u64 {
    // let x = vec![];
    let mut i = 0;
    while i < 8 {
        if idx & (1 << i) != 0 {
            // current i is a starting point
            let start = i;
            let mut count = 1;
            i += 1;
            while i < 8 && ((idx & (1 << i)) != 0) {
                count += 1;
                i += 1;
            }
            println!("start: {start}, count: {count}");
        }
        i += 1;
    }
    9
}
unsafe fn proc8() {
    println!("hello world");
    let shift = 0x00fffefdfcfbfaf900fffefdfcfbfaf9u128;
    let c = 0xff00ff00ff00ff00ff00ff00ff00ffffu128;
    let shift = vld1q_u8(&shift as *const _ as *const _);
    let c = vld1q_u8(&c as *const _ as *const _);
    printb(c);
    printb(shift);
    let vmask = vandq_u8(c, vdupq_n_u8(0x80));
    printb(vmask);
    let vmask = vshlq_u8(vmask, vreinterpretq_s8_u8(shift));
    printb(vmask);
    let low = vaddv_u8(vget_low_u8(vmask));
    let high = vaddv_u8(vget_high_u8(vmask));
    println!("{:b} | transformed: {:b}", low, proc8_transform(low));
    println!("{:b}", high);
}
unsafe fn main_() {
    proc8();
}
