#![allow(unused)]

use std::arch::aarch64::*;

#[rustfmt::skip]
fn main() { unsafe { main_ (); } }
unsafe fn printx<T>(t: T) {
    let x: u128 = unsafe { *std::mem::transmute::<&T, &u128>(&t) };
    println!("{:0>32x}", x);
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
unsafe fn main_() {
    proc2();
}
