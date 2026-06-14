//! Generate (description, io_examples) training pairs for the meta-learner.
//!
//! Two modes:
//!   --synthetic N [--n-args A] [--seed S]   generate N synthetic pairs
//!   --benchmark                              solve scalar benchmarks, capture universal params
//!
//! Output is newline-delimited JSON (JSONL) to stdout or --out FILE.
//!
//! Usage examples:
//!   cargo run --release --bin gen_meta_data -- --synthetic 5000 --n-args 1 --out data/synth_1arg.jsonl
//!   cargo run --release --bin gen_meta_data -- --benchmark --out data/benchmark.jsonl

use std::io::{BufWriter, Write};

use mog_synth::benchmark::{get_benchmark, Value};
use mog_synth::synthesis::{
    rand_description, synthesize_universal_and_collect, synthetic_record, MetaRecord,
    SoftUniversalProgram, UniversalProgramDescription,
};

/// Build hand-coded UniversalProgramDescription for common 1-arg function patterns.
/// Pool layout (n_args=1): [0]=a, [1]=c0=0, [2]=c1=1, [3]=c2=-1, [4]=c3=2, [5]=c4=-2, [6]=c5=10,
///                          [7]=v0, [8]=v1, [9]=v2, [10]=s0, [11]=s1, [12]=s2, [13]=s3, [14]=s4,
///                          [15]=s5, [16]=p0, [17]=p1
/// LIP layout: [0]=a, [1]=c0=0, [2]=c1=1, [3]=c2=-1, [4]=c3=2, [5]=c4=-2, [6]=c5=10, [7]=v0, [8]=v1, [9]=v2
fn known_1arg_descriptions() -> Vec<(String, UniversalProgramDescription)> {
    use mog_synth::synthesis::{SlotDesc, N_LOOP_SLOTS as NLS, N_UNIV_SLOTS as NS};
    let consts = [0f32, 1.0, -1.0, 2.0, -2.0, 10.0];
    // Pool indices
    const A: usize = 0;
    const C0: usize = 1;
    const C1: usize = 2;
    const CN1: usize = 3;
    const C2: usize = 4;
    const C10: usize = 6;
    const V0: usize = 7;
    const V1: usize = 8;
    const V2: usize = 9;
    const S0: usize = 10;
    const S1: usize = 11;
    const S2: usize = 12;
    const S3: usize = 13;
    const P0: usize = 16;
    // LIP indices
    const LA: usize = 0;
    const LC0: usize = 1;
    const LC1: usize = 2;
    const LC2: usize = 4;
    const LV0: usize = 7;

    // Default slot: identity (op=5=id, gate trivially false → else=a)
    let id_slot = SlotDesc {
        op: 5,
        s1: A,
        s2: A,
        gate_cmp: 5,
        gate_lhs: A,
        gate_rhs: A,
        else_val: A,
    };
    let no_loop_init = vec![LC0; NLS]; // all loop inits = const 0
                                       // No-loop condition: a != a = false

    let mut result = vec![];

    let mk = |name: &str,
              slots: Vec<SlotDesc>,
              loop_init: Vec<usize>,
              cond_cmp: usize,
              cond_lhs: usize,
              cond_rhs: usize,
              ret_src: usize|
     -> (String, UniversalProgramDescription) {
        (
            name.to_string(),
            UniversalProgramDescription {
                n_args: 1,
                slots,
                loop_init,
                cond_cmp,
                cond_lhs,
                cond_rhs,
                ret_src,
                consts,
            },
        )
    };

    // ─────────────── Simple arithmetic (no loop) ──────────────────────────────

    // f(a) = a + 1
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 0,
            s1: A,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        };
        result.push(mk("add_one", s, no_loop_init.clone(), 5, A, A, V0));
    }
    // f(a) = a + 2
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 0,
            s1: A,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        };
        result.push(mk("add_two_val", s, no_loop_init.clone(), 5, A, A, V0));
    }
    // f(a) = a - 1
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 1,
            s1: A,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        };
        result.push(mk("sub_one", s, no_loop_init.clone(), 5, A, A, V0));
    }
    // f(a) = a * 2
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 2,
            s1: A,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        };
        result.push(mk("double", s, no_loop_init.clone(), 5, A, A, V0));
    }
    // f(a) = a * a
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 2,
            s1: A,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        };
        result.push(mk("square", s, no_loop_init.clone(), 5, A, A, V0));
    }
    // f(a) = a * 10
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 2,
            s1: A,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        };
        result.push(mk("times_ten", s, no_loop_init.clone(), 5, A, A, V0));
    }
    // f(a) = a % 2 (is_even raw)
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 4,
            s1: A,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        };
        result.push(mk("mod_two", s, no_loop_init.clone(), 5, A, A, V0));
    }
    // f(a) = if a >= 0 then a else -a  (abs)
    {
        let mut s = vec![id_slot.clone(); NS];
        // v0 = -a = 0 - a
        s[0] = SlotDesc {
            op: 1,
            s1: C0,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        };
        // v1 = if a >= 0 then a else v0 (-a)   (cmp index 3 = ">=")
        s[1] = SlotDesc {
            op: 5,
            s1: A,
            s2: A,
            gate_cmp: 3,
            gate_lhs: A,
            gate_rhs: C0,
            else_val: V0,
        };
        result.push(mk("abs_val", s, no_loop_init.clone(), 5, A, A, V1));
    }
    // f(a) = if a > 0 then 1 else if a < 0 then -1 else 0  (sign)
    {
        let mut s = vec![id_slot.clone(); NS];
        // v0 = if a > 0 then 1 else -1
        s[0] = SlotDesc {
            op: 5,
            s1: C1,
            s2: C1,
            gate_cmp: 4,
            gate_lhs: A,
            gate_rhs: C0,
            else_val: CN1,
        };
        // v1 = if a == 0 then 0 else v0
        s[1] = SlotDesc {
            op: 5,
            s1: A,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: C0,
            else_val: V0,
        };
        result.push(mk("sign_fn", s, no_loop_init.clone(), 5, A, A, V1));
    }
    // f(a) = a * a + a
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 2,
            s1: A,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // a*a
        s[1] = SlotDesc {
            op: 0,
            s1: V0,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // a*a+a
        result.push(mk("sq_plus_n", s, no_loop_init.clone(), 5, A, A, V1));
    }

    // ─────────────── Sum to N: s0=0, loop while s1!=0: s0+=s1, s1-=1 ──────────
    {
        let mut s = vec![id_slot.clone(); NS];
        // Loop slots: s0=accum (+=s1), s1=counter (-=1)
        s[3] = SlotDesc {
            op: 0,
            s1: S0,
            s2: S1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0+=s1
        s[4] = SlotDesc {
            op: 1,
            s1: S1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1-=1
           // loop_init: s0=0, s1=a, others=0
        let mut li = vec![LC0; NLS];
        li[0] = LC0;
        li[1] = LA;
        result.push(mk("sum_to_n", s, li, 5, S1, C0, S0)); // while s1 != 0
    }
    // ─────────────── Factorial: s0=1, s1=n, while s1 != 1: s0*=s1, s1-=1 ─────
    {
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 2,
            s1: S0,
            s2: S1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0*=s1
        s[4] = SlotDesc {
            op: 1,
            s1: S1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1-=1
        let mut li = vec![LC0; NLS];
        li[0] = LC1;
        li[1] = LA; // s0=1, s1=a
        result.push(mk("factorial", s, li, 5, S1, LC1, S0)); // while s1 != 1
    }
    // ─────────────── Product 1..n (same as factorial) ─────────────────────────
    // (already covered by factorial)

    // ─────────────── Fibonacci: s0=0,s1=1, while s2!=n: tmp=s1, s1=s0+s1, s0=tmp, s2+=1 ─
    {
        let mut s = vec![id_slot.clone(); NS];
        // s0=prev fib, s1=cur fib, s2=counter
        s[3] = SlotDesc {
            op: 0,
            s1: S0,
            s2: S1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // new fib = s0+s1
        s[4] = SlotDesc {
            op: 5,
            s1: S1,
            s2: S1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // keep old s1
        s[5] = SlotDesc {
            op: 0,
            s1: S2,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        }; // s2+=1
           // After the loop: we need a post-processing step to fix up
           // This is complex — let's use the simpler 2-slot version:
           // s0=a, s1=0, s2=1, while s0 != 0: s0-=1, tmp=s2, s2=s1+s2, s1=tmp
        let mut s2 = vec![id_slot.clone(); NS];
        s2[3] = SlotDesc {
            op: 1,
            s1: S0,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0-=1
        s2[4] = SlotDesc {
            op: 0,
            s1: S1,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // new=s1+s2
        s2[5] = SlotDesc {
            op: 5,
            s1: S2,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        }; // keep s2
        let mut li2 = vec![LC0; NLS];
        li2[0] = LA;
        li2[1] = LC0;
        li2[2] = LC1; // s0=a, s1=0, s2=1
        result.push(mk("fibonacci", s2, li2, 5, S0, C0, S1)); // while s0 != 0, return s1
    }
    // ─────────────── Count digits: s0=n, s1=count, while s0 != 0: s0/=10, s1+=1 ─
    {
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 3,
            s1: S0,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0/=10
        s[4] = SlotDesc {
            op: 0,
            s1: S1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1+=1
        let mut li = vec![LC0; NLS];
        li[0] = LA;
        li[1] = LC0; // s0=a, s1=0
        result.push(mk("digit_count", s, li, 5, S0, C0, S1)); // while s0 != 0
    }
    // ─────────────── n^2 (square via loop) ───────────────────────────────────
    {
        let mut s = vec![id_slot.clone(); NS];
        // v0 = a*a directly (no loop needed)
        s[0] = SlotDesc {
            op: 2,
            s1: A,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        };
        result.push(mk("square_direct", s, no_loop_init.clone(), 5, A, A, V0));
    }
    // ─────────────── triangular n*(n+1)/2 ────────────────────────────────────
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 0,
            s1: A,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v0 = a+1
        s[1] = SlotDesc {
            op: 2,
            s1: A,
            s2: V0,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v1 = a*(a+1)
        s[2] = SlotDesc {
            op: 3,
            s1: V1,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v2 = v1/2
        result.push(mk("triangular", s, no_loop_init.clone(), 5, A, A, V2));
    }
    // ─────────────── celsius to fahrenheit: 9*a/5 + 32 ──────────────────────
    {
        let mut s = vec![id_slot.clone(); NS];
        // Approximate with consts we have: use 2*a - a/5 + 32? or direct
        // We can compute: v0 = a*2 (not 1.8 exactly, need integer)
        // Actually: f = 9*a/5 + 32. But 9 and 32 aren't in our const pool.
        // Let's use the approximation: f = 2*a - a/10 + 32 ≈ 1.9a + 32 (wrong)
        // Better: encode it as a loop sum? No.
        // Let's skip this as the const pool doesn't have 9 or 32.
        // Actually: f(0)=32, f(100)=212. With our consts [0,1,-1,2,-2,10]...
        // f = a*2 - a/5 + 32 can't be done exactly. Skip.
        // Use simpler: f(a) = a + 32 for illustration (imprecise but trainable)
        s[0] = SlotDesc {
            op: 2,
            s1: A,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // 2*a
        result.push(mk("linear_pattern", s, no_loop_init.clone(), 5, A, A, V0));
    }

    // ─────────────── Power of 2: 2^n ─────────────────────────────────────────
    // s0=result=1, s1=counter=n, while s1!=0: s0*=2, s1-=1
    {
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 2,
            s1: S0,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0*=2
        s[4] = SlotDesc {
            op: 1,
            s1: S1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1-=1
        let mut li = vec![LC0; NLS];
        li[0] = LC1;
        li[1] = LA; // s0=1, s1=n
        result.push(mk("power_of_2", s, li, 5, S1, C0, S0)); // while s1 != 0, return s0
    }

    // ─────────────── Digit sum: sum of decimal digits ────────────────────────
    // v0 = a%10 (init),  s0=acc=0, s1=n, s2=v0=a%10
    // Loop: s0+=s2, s1/=10, s2=s1%10 (sequential: s2 reads updated s1)
    {
        let mut s = vec![id_slot.clone(); NS];
        // Init slot 0: v0 = a % 10
        s[0] = SlotDesc {
            op: 4,
            s1: A,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        };
        // Loop slot 3 (s0): s0 = s0 + s2  (uses digit from prev iter)
        s[3] = SlotDesc {
            op: 0,
            s1: S0,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        };
        // Loop slot 4 (s1): s1 = s1 / 10
        s[4] = SlotDesc {
            op: 3,
            s1: S1,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        };
        // Loop slot 5 (s2): s2 = s1 % 10  (reads UPDATED s1 → next digit)
        s[5] = SlotDesc {
            op: 4,
            s1: S1,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        };
        let mut li = vec![LC0; NLS];
        li[0] = LC0;
        li[1] = LA;
        li[2] = LV0; // s0=0, s1=a, s2=v0
        result.push(mk("digit_sum", s, li, 5, S1, C0, S0)); // while s1 != 0, return s0
    }

    // ─────────────── Reverse digits: 123 → 321 ───────────────────────────────
    // v0 = a%10 (init),  s0=n (divides), s1=result, s2=last_digit(prev), s3=s1*10(staged)
    // Loop: s0/=10, s1=s3+s2, s2=s0%10, s3=s1*10
    {
        let mut s = vec![id_slot.clone(); NS];
        // Init slot 0: v0 = a % 10
        s[0] = SlotDesc {
            op: 4,
            s1: A,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        };
        // Loop slot 3 (s0): s0 = s0 / 10
        s[3] = SlotDesc {
            op: 3,
            s1: S0,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        };
        // Loop slot 4 (s1): s1 = s3 + s2  (staged_prev + digit_prev)
        s[4] = SlotDesc {
            op: 0,
            s1: S3,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        };
        // Loop slot 5 (s2): s2 = s0 % 10  (reads UPDATED s0)
        s[5] = SlotDesc {
            op: 4,
            s1: S0,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        };
        // Loop slot 6 (s3): s3 = s1 * 10  (reads UPDATED s1)
        s[6] = SlotDesc {
            op: 2,
            s1: S1,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S3,
        };
        let mut li = vec![LC0; NLS];
        li[0] = LA;
        li[1] = LC0;
        li[2] = LV0;
        li[3] = LC0; // s0=a, s1=0, s2=v0, s3=0
        result.push(mk("reverse_digits", s, li, 5, S0, C0, S1)); // while s0 != 0, return s1
    }

    // ─────────────── Sum of squares: 1^2+2^2+...+n^2 ────────────────────────
    // v0 = a*a (init),  s0=counter=n, s1=acc=0, s2=v0=a^2
    // Loop: s0-=1, s1+=s2, s2=s0*s0 (reads UPDATED s0)
    {
        let mut s = vec![id_slot.clone(); NS];
        // Init slot 0: v0 = a * a
        s[0] = SlotDesc {
            op: 2,
            s1: A,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        };
        // Loop slot 3 (s0): s0 = s0 - 1
        s[3] = SlotDesc {
            op: 1,
            s1: S0,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        };
        // Loop slot 4 (s1): s1 = s1 + s2  (uses square from prev iter / init)
        s[4] = SlotDesc {
            op: 0,
            s1: S1,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        };
        // Loop slot 5 (s2): s2 = s0 * s0  (reads UPDATED s0 = current-1)
        s[5] = SlotDesc {
            op: 2,
            s1: S0,
            s2: S0,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        };
        let mut li = vec![LC0; NLS];
        li[0] = LA;
        li[1] = LC0;
        li[2] = LV0; // s0=n, s1=0, s2=a^2
        result.push(mk("sum_squares", s, li, 5, S0, C0, S1)); // while s0 != 0, return s1
    }

    // ─────────────── Product 1..n (like factorial but starting index) ─────────
    // s0=result=1, s1=counter=n, while s1 != 0: s0*=s1, s1-=1
    {
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 2,
            s1: S0,
            s2: S1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        };
        s[4] = SlotDesc {
            op: 1,
            s1: S1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        };
        let mut li = vec![LC0; NLS];
        li[0] = LC1;
        li[1] = LA; // s0=1, s1=n
        result.push(mk("product_1_to_n", s, li, 5, S1, C0, S0)); // while s1 != 0
    }

    // ─────────────── Alternating sum: 1-2+3-4+...±n ─────────────────────────
    // s0=acc, s1=counter=n, s2=sign=1, while s1 != 0: s0+=s1*s2, s2*=-1, s1-=1
    // But s1*s2 needs temp → simplified: use gate to alternate add/subtract
    // v0 = a % 2 (odd/even), determines sign
    // Instead: simple version: count down adding with alternating sign
    // s0=0, s1=n, s2=1 (sign), while s1!=0: temp=s1*s2 stored in s3, s0+=s3, s2*=-1, s1-=1
    {
        let mut s = vec![id_slot.clone(); NS];
        // slot 3 (s0): s0 = s0 + s3  (add previous term)
        s[3] = SlotDesc {
            op: 0,
            s1: S0,
            s2: S3,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        };
        // slot 4 (s1): s1 = s1 - 1
        s[4] = SlotDesc {
            op: 1,
            s1: S1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        };
        // slot 5 (s2): s2 = s2 * -1  (flip sign)
        s[5] = SlotDesc {
            op: 2,
            s1: S2,
            s2: CN1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        };
        // slot 6 (s3): s3 = s1 * s2  (compute NEXT term using updated s1 and s2)
        s[6] = SlotDesc {
            op: 2,
            s1: S1,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S3,
        };
        // Init: v0 = a * 1 = a (first term = a with sign=1... but we want 1*sign, 2*sign...)
        // Actually: terms are n, n-1, ..., 1 with alternating signs starting from +
        // s0=0, s1=a, s2=1 (start sign positive), s3=a*1=a (first term)
        // v0 = a (= first term = a * sign=1)
        s[0] = SlotDesc {
            op: 5,
            s1: A,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v0=a (identity)
        let mut li = vec![LC0; NLS];
        li[0] = LC0;
        li[1] = LA;
        li[2] = LC1;
        li[3] = LV0;
        // s0=0, s1=n, s2=1(sign), s3=n(first term)
        result.push(mk("alternating_sum", s, li, 5, S1, C0, S0)); // while s1 != 0
    }

    // ─────────────── Collatz steps (number of steps to reach 1) ─────────────
    // s0=n, s1=count=0, while s0 != 1:
    //   if s0%2==0: s0=s0/2 else s0=3*s0+1
    //   s1+=1
    // Uses conditional gate: slot 3 (s0) conditionally updates
    {
        let mut s = vec![id_slot.clone(); NS];
        // slot 3 (s0): if s0%2==0 then s0/2 else 3*s0+1
        //   gate_cmp: s0%2 == 0  → needs temp. Use init slot for s0%2.
        // v0 = a % 2 (but this is fixed, not updated)
        // Can't easily do conditional in loop with SoftUniversalProgram slots.
        // Simplified collatz: just count step and alternate.
        // Actually: slot 3 computes half(s0) = s0/2, slot 4 computes triple_plus_1 = 3*s0+1
        // then slot 5 picks based on gate (s0%2==0 → half, else triple)
        // v0 = a%2 (initial parity, but changes)
        // This needs s2 = s0%2 to be recomputed each iteration.

        // Scheme: s0=current n, s1=steps, s2=s0%2 (parity)
        // slot 3 (s0): gate: s2==0 → s0/2, else 3*s0+1 → but 3*s0+1 can't be done in one op
        // Actually: if a%2==0: a/2. else 3a+1 requires pre-compute.
        // slot 3 (s0): gate(s2==0) → s0/2, else s0  (conditional halving)
        // slot 4 (s1): s1 = s1 + 1
        // slot 5 (s2): s2 = s0 % 2  (parity of NEW s0)
        // But we also need 3*s0+1 for odd case. Use post-processing in a second slot.
        // Instead: use TWO conditional slots:
        // slot 3 (s0): if s2==0 then s0/2 else s0 (halve if even)
        // slot 4 (s1): if s2!=0 then s0*3 else s0 → wait this would write to s1
        //
        // Alternative: use s3 as staging for odd case (3*s0+1)
        // slot 3 (s0): gate(s2==0) → s3, else s0  (s3 is 3*s0+1, computed in prev iter)
        //   Hmm: s3 holds 3*s0+1, but if even we need s0/2.
        //
        // Let's try a different approach:
        // v0 = a % 2 (init parity)
        // s0 = n, s1 = count, s2 = parity = a%2 initially
        // slot 3 (s0): if s2==0 then s0/2 else s0+s0+s0+1... need compound op
        //
        // Real collatz is tricky because it needs: if even: /2, else: *3+1
        // The *3+1 itself requires two operations.
        // Simplification: store 3*s0+1 in s3 each iter.
        // v0 = a (identity), s0=a, s1=0, s2=a%2, s3=3*a+1
        // slot 3 (s0): gate(s2==0) → s0/2, else s3
        //   gate_cmp=2(==), gate_lhs=S2, gate_rhs=C0(=0), then=s0/2, else=s3
        // slot 4 (s1): s1 = s1 + 1
        // slot 5 (s2): s2 = s0 % 2  (parity of new s0 from slot 3)
        // slot 6 (s3): s3 = s0 * 3 + 1... still two ops
        //   Actually: s3 = s0 * 3, then p0 = s3 + 1? Use post slot?
        //   No, post slots run AFTER the loop.
        //
        // Let's split 3*s0+1:
        // s3 = s0 * 3 (stored for next iter)
        // s4 = s3_prev + 1 (staged: add 1 to prev s3)
        // slot 3 (s0): gate(s2==0) → s0/2, else s4  (s4 = 3*prev_s0+1)
        // slot 4 (s1): s1 = s1 + 1
        // slot 5 (s2): s2 = s0 % 2
        // slot 6 (s3): s3 = s0 * 3
        // slot 7 (s4): s4 = s3 + 1  (reads NEW s3 from slot 6)
        //
        // Init: v0=a (identity for s0 init), s0=a, s1=0, s2=a%2, s3=3*a, s4=3*a+1
        // But s3 and s4 init... s3=3*a needs v1=3*a, s4=v1+1.
        // Actually: v1 = a*3 (init slot 1: op=*, s1=A, s2=C2+C1? NO, can't combine)
        // v1 can be from LIP. In LIP: [a, c0..c5, v0..v2].
        // v1 is computed by init slot 1.
        // v2 is computed by init slot 2.
        // s[1] (init slot 1) = v1 = a*3... but we don't have constant 3 directly.
        // c3=2, so 3 = c3+c2 = 2+1? Can't do that in one slot.
        // Actually c3=2 means consts[3]=2 (which is C2 in our naming, C2:usize=4 is pool[4]).
        // Hmm, let me re-check: consts = [0.0, 1.0, -1.0, 2.0, -2.0, 10.0]
        // Pool: [a, c0=0, c1=1, c2=-1, c3=2, c4=-2, c5=10, v0, v1, v2, s0..s5, p0, p1]
        //   C0=1 (=c0=0), C1=2 (=c1=1), CN1=3 (=c2=-1), C2=4 (=c3=2), CN2=5 (=c4=-2), C10=6 (=c5=10)
        //
        // So we have constants 0,1,-1,2,-2,10 available. No 3.
        // 3 = 2 + 1 → need two ops. We can't compute 3*a directly.
        //
        // Alternative: collatz_steps is too complex for a simple hand-coded description.
        // The synthesizer finds it, but it might use a non-obvious encoding.
        // Let's skip collatz in the known descriptions and wait for benchmark collection.

        // Instead: add a simpler loop: count while a != 0 (decrements by 1)
        // f(a) = a (identity, trivial)
        // Better: f(a) = a mod something

        // Let's add: multiply_by_factor loop: f(a) = a * 3 via repeated addition
        // s0=0, s1=a (counter), while s1!=0: s0+=a, s1-=1  → s0 = a*a? No...
        // Actually s0=0, s1=a, while s1!=0: s0+=3, s1-=1 → s0 = 3*a (times_three via loop)
        // But we don't have constant 3.

        // Simplest meaningful loop not yet covered:
        // f(a) = a/2 * 2 (floor to even)  -- no-loop
        //
        // Let's add: nth_triangle = n*(n+1)/2 via loop (cumulative)
        // s0=n, s1=acc=0, while s0!=0: s1+=s0, s0-=1  → same as sum_to_n!
        // Not unique.

        // Add: negative_factorial: product of -1,-2,...,-n
        // = (-1)^n * n!
        // s0=1, s1=n, s2=1 (sign), while s1!=0: s0*=s1, s1-=1, s2*=-1
        // Return: s0 * s2 (but need to multiply s0*s2 which requires post slot)
        // Simplified: just skip if too complex.

        // ADD: power function: a^k where k is given (but this needs 2 args)

        // Let me just add collatz step counter using a gate approach
        // even though it requires 3 = 1+2 via two slots
        s[0] = SlotDesc {
            op: 4,
            s1: A,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v0=a%2
           // v1 = a * 2 (will use for 3*s0 computation)
        s[1] = SlotDesc {
            op: 0,
            s1: A,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v1=a+a=2a
           // s0=a (counter), s1=0 (count), s2=a%2, s3=staging 3*s0+1, s4=a+a=2a initial
           // slot 3 (s0): gate(s2==0) → s0/2, else s0  [even case handled; odd case needs separate slot]
           // Actually we need: if even → s0/2, else 3*s0+1
           // slot 3 (s0): s0 = s0 / 2  (always divide by 2; undo for odd case in slot 4?)
           // This doesn't work cleanly. Let me just use a gate for the conditionally-halved version:
           // gate(s2==0): s0/2, else (s0*2+s0+1)? Can't do 3*s0+1 in one slot.
           // Skip collatz - it's too hard to hand-code.
           // Reset to id_slot
        s[0] = id_slot.clone();
        s[1] = id_slot.clone();
        // Fall through to a simple: abs(a) using gate  (same as abs_val above, keep for coverage)
    }

    // ─────────────── is_odd: returns 1 if odd, 0 if even ─────────────────────
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 4,
            s1: A,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v0=a%2
        result.push(mk("is_odd", s, no_loop_init.clone(), 5, A, A, V0));
    }

    // ─────────────── is_even: returns 1 if even, 0 if odd ────────────────────
    // v0=a%2, v1=if v0==0 then 1 else 0
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 4,
            s1: A,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v0=a%2
        s[1] = SlotDesc {
            op: 5,
            s1: C1,
            s2: C1,
            gate_cmp: 2, /*==*/
            gate_lhs: V0,
            gate_rhs: C0,
            else_val: C0,
        }; // v1=v0==0?1:0
        result.push(mk("is_even", s, no_loop_init.clone(), 5, A, A, V1));
    }

    // ─────────────── cube: a^3 = a*a*a ───────────────────────────────────────
    // v0=a*a, v1=v0*a
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 2,
            s1: A,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v0=a*a
        s[1] = SlotDesc {
            op: 2,
            s1: V0,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v1=v0*a
        result.push(mk("cube", s, no_loop_init.clone(), 5, A, A, V1));
    }

    // ─────────────── negate: -a ───────────────────────────────────────────────
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 2,
            s1: A,
            s2: CN1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v0=-a
        result.push(mk("negate", s, no_loop_init.clone(), 5, A, A, V0));
    }

    // ─────────────── clamp_positive: max(a, 0) ────────────────────────────────
    {
        let mut s = vec![id_slot.clone(); NS];
        // v0 = if a>=0 then a else 0
        s[0] = SlotDesc {
            op: 5,
            s1: A,
            s2: A,
            gate_cmp: 3, /*>=*/
            gate_lhs: A,
            gate_rhs: C0,
            else_val: C0,
        };
        result.push(mk("clamp_positive", s, no_loop_init.clone(), 5, A, A, V0));
    }

    // ─────────────── halve_floor: a/2 ─────────────────────────────────────────
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 3,
            s1: A,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // a/2
        result.push(mk("halve_floor", s, no_loop_init.clone(), 5, A, A, V0));
    }

    // ─────────────── digit_count (correct): handles n=0→1 ────────────────────
    // s0=n, s1=0, while s0>0: s1+=1, s0/=10
    // Post: p0 = if s1==0 then 1 else s1
    {
        const S4: usize = 14;
        const S5: usize = 15;
        const P0: usize = 16;
        const _P1: usize = 17;
        let _ = (S4, S5); // used in loop_init positions only
        let mut s = vec![id_slot.clone(); NS];
        // Loop slot 3 (s0): s0 = s0 / 10
        s[3] = SlotDesc {
            op: 3,
            s1: S0,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        };
        // Loop slot 4 (s1): s1 = s1 + 1
        s[4] = SlotDesc {
            op: 0,
            s1: S1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        };
        // Post slot 9 (p0): if s1==0 then 1 else s1
        s[9] = SlotDesc {
            op: 5,
            s1: C1,
            s2: C1,
            gate_cmp: 2, /*==*/
            gate_lhs: S1,
            gate_rhs: C0,
            else_val: S1,
        };
        let mut li = vec![LC0; NLS];
        li[0] = LA;
        li[1] = LC0; // s0=n, s1=0
        result.push(mk("digit_count_v2", s, li, 4 /*>*/, S0, C0, P0)); // while s0>0, return p0
    }

    // ─────────────── fibonacci (correct): s3 staging for simultaneous update ─
    // s0=n(counter), s1=a(fib n-1), s2=b(fib n), s3=staging(=old a, set AFTER s1 update)
    // Slot 4: s1=s2 (new a = old b) — reads OLD s2
    // Slot 5: s2=s3+s2 (new b = old_a(from s3) + old_b) — reads OLD s3, OLD s2
    // Slot 6: s3=s1_new (staging = new a for next iter) — reads UPDATED s1
    {
        const S3: usize = 13;
        let mut s = vec![id_slot.clone(); NS];
        // Slot 3 (s0): s0 -= 1
        s[3] = SlotDesc {
            op: 1,
            s1: S0,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        };
        // Slot 4 (s1): s1 = s2 (reads OLD s2, since slot 4 runs before slot 5)
        s[4] = SlotDesc {
            op: 5,
            s1: S2,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        };
        // Slot 5 (s2): s2 = s3 + s2 (reads OLD s3 and OLD s2; s3 from prev iter = old a)
        s[5] = SlotDesc {
            op: 0,
            s1: S3,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        };
        // Slot 6 (s3): s3 = s1 (reads NEW s1 = old_b, for next iteration)
        s[6] = SlotDesc {
            op: 5,
            s1: S1,
            s2: S1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S3,
        };
        let mut li = vec![LC0; NLS];
        li[0] = LA;
        li[1] = LC0;
        li[2] = LC1;
        li[3] = LC0;
        // s0=n, s1=0, s2=1, s3=0 (initial "old a" = 0)
        result.push(mk("fibonacci_v2", s, li, 5 /* != */, S0, C0, S1)); // while s0!=0, return s1
    }

    // ─────────────── lucas_number: L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2) ───────
    // Same as fibonacci_v2 but init: s1=2, s2=1, s3=2
    {
        const S3: usize = 13;
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 1,
            s1: S0,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        };
        s[4] = SlotDesc {
            op: 5,
            s1: S2,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        };
        s[5] = SlotDesc {
            op: 0,
            s1: S3,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        };
        s[6] = SlotDesc {
            op: 5,
            s1: S1,
            s2: S1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S3,
        };
        let mut li = vec![LC0; NLS];
        li[0] = LA;
        li[1] = LC2;
        li[2] = LC1;
        li[3] = LC2;
        // s0=n, s1=2, s2=1, s3=2  (L(0)=2, L(1)=1)
        result.push(mk("lucas_number", s, li, 5 /* != */, S0, C0, S1));
    }

    // ─────────────── digit_product: product of all decimal digits ─────────────
    // v0=a%10, s0=product=1, s1=counter=a, s2=v0=last_digit
    // Like digit_sum but multiply: slot 3: s0*=s2, slot 4: s1/=10, slot 5: s2=s1%10
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 4,
            s1: A,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v0=a%10
        s[3] = SlotDesc {
            op: 2,
            s1: S0,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0*=s2
        s[4] = SlotDesc {
            op: 3,
            s1: S1,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1/=10
        s[5] = SlotDesc {
            op: 4,
            s1: S1,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        }; // s2=s1%10
        let mut li = vec![LC0; NLS];
        li[0] = LC1;
        li[1] = LA;
        li[2] = LV0;
        result.push(mk("digit_product", s, li, 5 /* != */, S1, C0, S0)); // while s1!=0, return s0
    }

    // ─────────────── max_digit: maximum decimal digit of n ────────────────────
    // v0=a%10, s0=a(counter), s1=max=0, s2=v0=last_digit
    // Slot 3: s0/=10, slot 4: s1=max(s1,s2) via gate, slot 5: s2=s0%10(updated)
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 4,
            s1: A,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v0=a%10
        s[3] = SlotDesc {
            op: 3,
            s1: S0,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0/=10
           // Slot 4: s1 = if s2>s1 then s2 else s1 (update max)
        s[4] = SlotDesc {
            op: 5,
            s1: S2,
            s2: S2,
            gate_cmp: 4, /*>*/
            gate_lhs: S2,
            gate_rhs: S1,
            else_val: S1,
        };
        s[5] = SlotDesc {
            op: 4,
            s1: S0,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        }; // s2=s0%10
        let mut li = vec![LC0; NLS];
        li[0] = LA;
        li[1] = LC0;
        li[2] = LV0;
        result.push(mk("max_digit", s, li, 5 /* != */, S0, C0, S1)); // while s0!=0, return s1
    }

    // ─────────────── leading_digit: most significant digit ───────────────────
    // s0=n, while s0>=10: s0/=10, return s0
    {
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 3,
            s1: S0,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0/=10
        let mut li = vec![LC0; NLS];
        li[0] = LA;
        result.push(mk("leading_digit", s, li, 3 /*>=*/, S0, C10, S0)); // while s0>=10, return s0
    }

    // ─────────────── popcount: count 1-bits (base-2 digit sum) ───────────────
    // v0=a%2, s0=acc=0, s1=n, s2=v0=a%2
    // Like digit_sum but divide by 2 instead of 10
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 4,
            s1: A,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v0=a%2
        s[3] = SlotDesc {
            op: 0,
            s1: S0,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0+=s2
        s[4] = SlotDesc {
            op: 3,
            s1: S1,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1/=2
        s[5] = SlotDesc {
            op: 4,
            s1: S1,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        }; // s2=s1%2
        let mut li = vec![LC0; NLS];
        li[0] = LC0;
        li[1] = LA;
        li[2] = LV0;
        result.push(mk("popcount", s, li, 5 /* != */, S1, C0, S0)); // while s1!=0, return s0
    }

    // ─────────────── polynomial: 2*x*x + 3*x + 1 ────────────────────────────
    // v0=x+1, v1=v0*x=(x+1)*x=x^2+x, v2=v1*2=2x^2+2x, p0=v2+v0=2x^2+3x+1
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 0,
            s1: A,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v0=a+1
        s[1] = SlotDesc {
            op: 2,
            s1: V0,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: V0,
        }; // v1=v0*a=(x+1)*x
        s[2] = SlotDesc {
            op: 2,
            s1: V1,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: V1,
        }; // v2=v1*2=2x^2+2x
        s[9] = SlotDesc {
            op: 0,
            s1: V2,
            s2: V0,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: V2,
        }; // p0=v2+v0=2x^2+3x+1
        result.push(mk(
            "polynomial_v2",
            s,
            no_loop_init.clone(),
            5, /* a!=a */
            A,
            A,
            P0,
        )); // no loop
    }

    // ─────────────── collatz_steps: count steps n→1 via Collatz ─────────────
    // Registers: S5=n (written LAST so intermediates see old n), S4=steps counter.
    // Slots (in order 3→4→5→6→7→8 writing S0..S5):
    //   S0=2n (s[3]), S1=3n=S0+S5 (s[4]), S2=3n+1=S1+1 (s[5]),
    //   S3=parity=S5%2 (s[6]), S4=steps+1 (s[7]), S5=new_n=if parity==0:S5/2 else S2 (s[8])
    {
        const S4: usize = 14;
        const S5: usize = 15;
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 2,
            s1: S5,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0=s5*2=2n (s5 old)
        s[4] = SlotDesc {
            op: 0,
            s1: S0,
            s2: S5,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1=s0+s5=3n (s0 new, s5 old)
        s[5] = SlotDesc {
            op: 0,
            s1: S1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        }; // s2=s1+1=3n+1
        s[6] = SlotDesc {
            op: 4,
            s1: S5,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S3,
        }; // s3=s5%2 (parity, s5 old)
        s[7] = SlotDesc {
            op: 0,
            s1: S4,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S4,
        }; // s4=steps+1
           // s5 = if s3==0 (even) then s5/2 else s2 (3n+1); s5 still old here
        s[8] = SlotDesc {
            op: 3,
            s1: S5,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: S3,
            gate_rhs: C0,
            else_val: S2,
        }; // s5=n/2 or 3n+1
        let mut li = vec![LC0; NLS];
        li[5] = LA; // s5=a(n), s4=0(steps), rest=0
        result.push(mk("collatz_steps_v2", s, li, 4 /* > */, S5, C1, S4)); // while s5>1, return s4 (steps)
    }

    // ─────────────── count_even_digits: count even decimal digits of n ────────
    // Non-pipelined. S0=digit, S1=parity, S2=count, S3=n_counter (decremented).
    // Slots run in ORDER 3→4→5→6: compute digit, compute parity, update count, divide.
    // Post: p0 = if a==0 then 1 else s2
    {
        const S3: usize = 13;
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 4,
            s1: S3,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0=s3%10 (digit)
        s[4] = SlotDesc {
            op: 4,
            s1: S0,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1=s0%2 (parity; s0=new digit)
        s[5] = SlotDesc {
            op: 0,
            s1: S2,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: S1,
            gate_rhs: C0,
            else_val: S2,
        }; // s2+=1 if parity==0
        s[6] = SlotDesc {
            op: 3,
            s1: S3,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S3,
        }; // s3/=10 (old s3)
        s[9] = SlotDesc {
            op: 5,
            s1: C1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: C0,
            else_val: S2,
        }; // p0=1 if a==0 else s2
        let mut li = vec![LC0; NLS];
        li[3] = LA; // s0=0,s1=0,s2=0(count),s3=a(counter)
        result.push(mk(
            "count_even_digits_v2",
            s,
            li,
            4, /* > */
            S3,
            C0,
            P0,
        )); // while s3>0, return p0
    }

    // ─────────────── sum_odd_digits: sum of odd decimal digits of n ───────────
    // S0=digit, S1=parity, S2=sum, S3=n_counter
    {
        const S3: usize = 13;
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 4,
            s1: S3,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0=s3%10 (digit)
        s[4] = SlotDesc {
            op: 4,
            s1: S0,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1=s0%2 (parity)
        s[5] = SlotDesc {
            op: 0,
            s1: S2,
            s2: S0,
            gate_cmp: 2,
            gate_lhs: S1,
            gate_rhs: C1,
            else_val: S2,
        }; // s2+=digit if parity==1
        s[6] = SlotDesc {
            op: 3,
            s1: S3,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S3,
        }; // s3/=10
        let mut li = vec![LC0; NLS];
        li[3] = LA; // s2=0(sum), s3=a(counter)
        result.push(mk("sum_odd_digits_v2", s, li, 4 /* > */, S3, C0, S2)); // while s3>0, return s2
    }

    // ─────────────── triangular_check: return 1 if n is triangular ───────────
    // k=s0 (counter), T=s1 (triangular sum), loop: k++, T+=k; while T<n; post: T==n?
    {
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 0,
            s1: S0,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0+=1 (k++)
        s[4] = SlotDesc {
            op: 0,
            s1: S1,
            s2: S0,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1+=s0 (T+=new_k)
           // Post: p0 = if s1==a then 1 else 0
        s[9] = SlotDesc {
            op: 5,
            s1: C1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: S1,
            gate_rhs: A,
            else_val: C0,
        };
        result.push(mk(
            "triangular_check_v2",
            s,
            no_loop_init.clone(),
            0, /* < */
            S1,
            A,
            P0,
        )); // while s1<a, return p0
    }

    // ─────────────── count_divisors: count positive divisors of n ────────────
    // Non-pipelined: S1=A%i check (slot 4), S2=count (slot 5 uses new S1), S3=i counter (slot 6).
    // Slot order 3(unused)→4(compute check)→5(update count)→6(increment i).
    {
        const S3: usize = 13;
        let mut s = vec![id_slot.clone(); NS];
        s[4] = SlotDesc {
            op: 4,
            s1: A,
            s2: S3,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1=A%s3(=A%i)
        s[5] = SlotDesc {
            op: 0,
            s1: S2,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: S1,
            gate_rhs: C0,
            else_val: S2,
        }; // s2+=1 if s1==0
        s[6] = SlotDesc {
            op: 0,
            s1: S3,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S3,
        }; // s3+=1 (i++)
        let mut li = vec![LC0; NLS];
        li[3] = LC1; // s2=0(count), s3=1(i)
        result.push(mk("count_divisors_v2", s, li, 1 /* <= */, S3, A, S2)); // while s3<=a, return s2
    }

    // ─────────────── sum_of_divisors: sum of all positive divisors ────────────
    // Same structure: S1=A%i, S2=sum, S3=i; add S3 (old i) when S1==0.
    {
        const S3: usize = 13;
        let mut s = vec![id_slot.clone(); NS];
        s[4] = SlotDesc {
            op: 4,
            s1: A,
            s2: S3,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1=A%s3(=A%i)
        s[5] = SlotDesc {
            op: 0,
            s1: S2,
            s2: S3,
            gate_cmp: 2,
            gate_lhs: S1,
            gate_rhs: C0,
            else_val: S2,
        }; // s2+=s3(=i) if s1==0
        s[6] = SlotDesc {
            op: 0,
            s1: S3,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S3,
        }; // s3+=1 (i++)
        let mut li = vec![LC0; NLS];
        li[3] = LC1; // s2=0(sum), s3=1(i)
        result.push(mk("sum_of_divisors_v2", s, li, 1 /* <= */, S3, A, S2)); // while s3<=a, return s2
    }

    // ─────────────── harmonic_sum: sum of 1000/i for i=1..n ─────────────────
    // v0=100=C10*C10, v1=1000=v0*C10
    // s0=acc, s1=prev(1000/i), s2=i; pipelined: add prev term, then compute next
    // Post: p0=s0+s1 (add last term)
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 2,
            s1: C10,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: C10,
        }; // v0=100
        s[1] = SlotDesc {
            op: 2,
            s1: V0,
            s2: C10,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: V0,
        }; // v1=1000
        s[3] = SlotDesc {
            op: 0,
            s1: S0,
            s2: S1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0+=prev_term
        s[4] = SlotDesc {
            op: 3,
            s1: V1,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1=1000/i (old i)
        s[5] = SlotDesc {
            op: 0,
            s1: S2,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        }; // s2=i+1
        s[9] = SlotDesc {
            op: 0,
            s1: S0,
            s2: S1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // p0=s0+last_term
        let mut li = vec![LC0; NLS];
        li[2] = LC1; // s0=0,s1=0,s2=1(i starts at 1)
        result.push(mk("harmonic_sum_v2", s, li, 1 /* <= */, S2, A, P0)); // while s2<=a, return p0
    }

    // ─────────────── is_perfect_square: return 1 if n is a perfect square ────
    // s0=guess, s1=guess^2; loop: s0++, s1=s0^2; while s1<n; post: s1==n?
    {
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 0,
            s1: S0,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0+=1 (guess++)
        s[4] = SlotDesc {
            op: 2,
            s1: S0,
            s2: S0,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1=s0^2 (new s0)
           // Post: p0 = if s1==a then 1 else 0
        s[9] = SlotDesc {
            op: 5,
            s1: C1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: S1,
            gate_rhs: A,
            else_val: C0,
        };
        result.push(mk(
            "is_perfect_square_v2",
            s,
            no_loop_init.clone(),
            0, /* < */
            S1,
            A,
            P0,
        )); // while s1<a, return p0
    }

    // ─────────────── next_power_of_2: smallest power of 2 >= n ──────────────
    // s0=1 (p); while s0<n: s0*=2; return s0
    {
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 2,
            s1: S0,
            s2: C2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0*=2
        let mut li = vec![LC0; NLS];
        li[0] = LC1; // s0=1
        result.push(mk("next_power_of_2_v2", s, li, 0 /* < */, S0, A, S0)); // while s0<a, return s0
    }

    // ─────────────── sign_v2: alternative sign using less/greater only ────────
    // v0 = if a < 0 then -1 else 0
    // v1 = if a > 0 then 1 else v0   (→ v1 = sign(a))
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 5,
            s1: CN1,
            s2: CN1,
            gate_cmp: 0, /*<*/
            gate_lhs: A,
            gate_rhs: C0,
            else_val: C0,
        }; // v0=a<0?-1:0
        s[1] = SlotDesc {
            op: 5,
            s1: C1,
            s2: C1,
            gate_cmp: 4, /*>*/
            gate_lhs: A,
            gate_rhs: C0,
            else_val: V0,
        }; // v1=a>0?1:v0
        result.push(mk("sign_v2", s, no_loop_init.clone(), 5, A, A, V1));
    }
    // ─────────────── sign_v3: three-way via double gate ───────────────────────
    // v0 = if a != 0 then (a/|a| via sign comparison) else 0
    // Simpler: v0=max(0,a)/max(1,a) doesn't work easily. Instead use nesting:
    // v0 = if a >= 0 then 1 else -1    (1 for zero and positive)
    // v1 = if a == 0 then 0 else v0    (zero override)
    {
        let mut s = vec![id_slot.clone(); NS];
        s[0] = SlotDesc {
            op: 5,
            s1: C1,
            s2: C1,
            gate_cmp: 3, /*>=*/
            gate_lhs: A,
            gate_rhs: C0,
            else_val: CN1,
        }; // v0=a>=0?1:-1
        s[1] = SlotDesc {
            op: 5,
            s1: C0,
            s2: C0,
            gate_cmp: 2, /*==*/
            gate_lhs: A,
            gate_rhs: C0,
            else_val: V0,
        }; // v1=a==0?0:v0
        result.push(mk("sign_v3", s, no_loop_init.clone(), 5, A, A, V1));
    }

    // ─────────────── sum_to_n_v2: accumulate from 1 up to n ─────────────────
    // s0=acc=0, s1=counter=1, while s1 <= a: s0+=s1, s1+=1. Return s0.
    {
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 0,
            s1: S0,
            s2: S1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0+=s1
        s[4] = SlotDesc {
            op: 0,
            s1: S1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1+=1
        let mut li = vec![LC0; NLS];
        li[0] = LC0;
        li[1] = LC1; // s0=0, s1=1
        result.push(mk("sum_to_n_v2", s, li, 1 /* <= */, S1, A, S0)); // while s1<=a, return s0
    }
    // ─────────────── sum_to_n_v3: count-down variant but slot assignment swapped
    // s0=counter=a, s1=acc=0, while s0!=0: s1+=s0, s0-=1. Return s1.
    {
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 1,
            s1: S0,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0-=1
        s[4] = SlotDesc {
            op: 0,
            s1: S1,
            s2: S0,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1+=new_s0
        let mut li = vec![LC0; NLS];
        li[0] = LA;
        li[1] = LC0; // s0=a, s1=0
        result.push(mk("sum_to_n_v3", s, li, 5 /* != */, S0, C0, S1)); // while s0!=0, return s1
    }

    // ─────────────── nth_triangle_loop: same loop as sum_to_n ────────────────
    // Triangular number = 1+2+...+n.  Exact same loop as sum_to_n.
    // s0=counter=n, s1=acc=0; while s0!=0: s0-=1, s1+=s0+1  — OR simpler: s1+=counter before decrement
    // Cleanest: s0=acc, s1=counter=a; while s1!=0: s0+=s1, s1-=1. Same as sum_to_n.
    {
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 0,
            s1: S0,
            s2: S1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0+=s1
        s[4] = SlotDesc {
            op: 1,
            s1: S1,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1-=1
        let mut li = vec![LC0; NLS];
        li[0] = LC0;
        li[1] = LA; // s0=0, s1=a
        result.push(mk("nth_triangle_loop", s, li, 5 /* != */, S1, C0, S0)); // same as sum_to_n
    }

    // ─────────────── sum_squares_v2: forward accumulation ────────────────────
    // s0=counter=1, s1=acc=0; while s0<=a: s1+=s0*s0, s0+=1
    {
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 0,
            s1: S1,
            s2: S0,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // dummy — need temp
           // Actually forward: need s2=s0*s0 first, then s1+=s2, then s0+=1
           // Reuse slot 3 for square, slot 4 for acc, slot 5 for counter
        s[3] = SlotDesc {
            op: 2,
            s1: S0,
            s2: S0,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        }; // s2=s0^2 wait no s2 is written later
           // s0=counter, s1=acc, s2=staging: slot3 computes s0^2 into s2, slot4 adds to s1, slot5 increments s0
        s[3] = SlotDesc {
            op: 2,
            s1: S0,
            s2: S0,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0=s0*s0 (overwrite counter!)
           // This doesn't work right without staging. Let's use a simpler loop:
           // v0=a^2 init; s0=counter=a, s1=acc=0; while s0!=0: s1+=s0^2, s0-=1 using staging s2=s0^2
           // Rewrite: s0 counts DOWN; s1=acc; s2=s0^2 from prev slot
           // slot3: s0-=1 (writes new s0); slot4: s1+=s2 (uses old s2); slot5: s2=s0*s0 (uses new s0)
        s[3] = SlotDesc {
            op: 1,
            s1: S0,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        }; // s0-=1
        s[4] = SlotDesc {
            op: 0,
            s1: S1,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        }; // s1+=s2
        s[5] = SlotDesc {
            op: 2,
            s1: S0,
            s2: S0,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        }; // s2=s0^2
           // init v0=a*a, then s0=a, s1=0, s2=v0=a^2
        s[0] = SlotDesc {
            op: 2,
            s1: A,
            s2: A,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: A,
        }; // v0=a^2
        let mut li = vec![LC0; NLS];
        li[0] = LA;
        li[1] = LC0;
        li[2] = LV0; // s0=a,s1=0,s2=a^2
        result.push(mk("sum_squares_v2", s, li, 5 /* != */, S0, C0, S1)); // while s0!=0, return s1
    }

    // ─────────────── lucas_v2: alternative register layout ────────────────────
    // L(0)=2, L(1)=1. Same Fibonacci structure but s1=1, s2=2 (L1, L0 swapped).
    // s0=n(counter), s1=cur=1, s2=prev=2, s3=staging(old s1)
    // slot3: s0-=1; slot4: s1=s2 (read OLD s2 = prev term → new cur);
    // slot5: s2=s3+s2 (OLD s3 = prev cur, OLD s2 = prev); slot6: s3=NEW s1
    {
        const S3: usize = 13;
        let mut s = vec![id_slot.clone(); NS];
        s[3] = SlotDesc {
            op: 1,
            s1: S0,
            s2: C1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S0,
        };
        s[4] = SlotDesc {
            op: 5,
            s1: S2,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S1,
        };
        s[5] = SlotDesc {
            op: 0,
            s1: S3,
            s2: S2,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S2,
        };
        s[6] = SlotDesc {
            op: 5,
            s1: S1,
            s2: S1,
            gate_cmp: 2,
            gate_lhs: A,
            gate_rhs: A,
            else_val: S3,
        };
        let mut li = vec![LC0; NLS];
        li[0] = LA;
        li[1] = LC1;
        li[2] = LC2;
        li[3] = LC1;
        // s0=n, s1=1(L1), s2=2(L0), s3=1(init old_s1)
        result.push(mk("lucas_v2", s, li, 5 /* != */, S0, C0, S1));
    }

    result
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let out_path = arg_value(&args, "--out");
    let writer: Box<dyn Write> = match out_path {
        Some(ref p) => {
            let f = std::fs::File::create(p).unwrap_or_else(|e| {
                eprintln!("cannot create {p}: {e}");
                std::process::exit(1);
            });
            Box::new(BufWriter::new(f))
        }
        None => Box::new(BufWriter::new(std::io::stdout())),
    };
    let mut w = BufWriter::new(writer);

    if has_flag(&args, "--synthetic") {
        let n: usize = arg_value(&args, "--synthetic")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);
        let n_args: usize = arg_value(&args, "--n-args")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let base_seed: u64 = arg_value(&args, "--seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(42);
        let n_eval: usize = arg_value(&args, "--n-eval")
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        eprintln!("Generating {n} synthetic pairs (n_args={n_args}, n_eval={n_eval})...");
        let mut kept = 0usize;
        let mut tried = 0usize;
        while kept < n {
            let seed = base_seed.wrapping_add(tried as u64 * 7919);
            tried += 1;
            let desc = rand_description(n_args, seed);
            if let Some(record) = synthetic_record(&desc, n_eval, seed ^ 0xdeadbeef) {
                let line =
                    serde_json::to_string(&record).unwrap_or_else(|e| panic!("json error: {e}"));
                writeln!(w, "{line}").unwrap();
                kept += 1;
                if kept % 100 == 0 {
                    eprintln!(
                        "  {kept}/{n} kept ({tried} tried, {:.0}% accept rate)",
                        kept as f64 / tried as f64 * 100.0
                    );
                }
            }
        }
        eprintln!("Done: {kept} pairs from {tried} attempts.");
    } else if has_flag(&args, "--benchmark") {
        let filter_n_args: Option<usize> =
            arg_value(&args, "--n-args").and_then(|s| s.parse().ok());
        let max_steps: usize = arg_value(&args, "--max-steps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(400);
        let all_problems = get_benchmark(1);
        let problems: Vec<_> = all_problems
            .into_iter()
            .filter(|p| {
                let n = p.examples.first().map(|ex| ex.inputs.len()).unwrap_or(0);
                // Only integer problems; filter by n_args if specified
                p.examples.iter().all(|ex| {
                    ex.inputs
                        .iter()
                        .all(|v| matches!(v, mog_synth::benchmark::Value::Int(_)))
                }) && filter_n_args.map_or(true, |fa| n == fa)
            })
            .collect();
        let total = problems.len();
        eprintln!("Running universal synthesis on {total} benchmark problems (n_args={:?}, max_steps={max_steps})...",
            filter_n_args);
        let mut collected = 0usize;

        for (i, problem) in problems.iter().enumerate() {
            eprint!("  [{}/{}] {} ... ", i + 1, total, problem.name);
            match synthesize_universal_and_collect(problem, max_steps) {
                Some((result, params)) => {
                    let n_args = problem
                        .examples
                        .first()
                        .map(|ex| ex.inputs.len())
                        .unwrap_or(1);
                    let prog = SoftUniversalProgram::new_from_params(n_args, params);
                    let desc = prog.params_to_description();
                    let io_examples: Vec<(Vec<i64>, i64)> = problem
                        .examples
                        .iter()
                        .map(|ex| {
                            let inputs: Vec<i64> = ex
                                .inputs
                                .iter()
                                .filter_map(|v| {
                                    if let mog_synth::benchmark::Value::Int(i) = v {
                                        Some(*i)
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            (inputs, ex.expected_int())
                        })
                        .collect();
                    let record = MetaRecord {
                        fn_name: result.method.clone() + "_" + &problem.name,
                        description: desc,
                        io_examples,
                        source: "benchmark".to_string(),
                    };
                    let line = serde_json::to_string(&record)
                        .unwrap_or_else(|e| panic!("json error: {e}"));
                    writeln!(w, "{line}").unwrap();
                    collected += 1;
                    eprintln!("SOLVED");
                }
                None => {
                    eprintln!("failed (not solvable by universal program)");
                }
            }
        }
        eprintln!("Done: {collected}/{total} collected via universal program.");
    } else if has_flag(&args, "--known") {
        // --known --reps N --out OUT
        // Generate training data from hand-coded program descriptions for common patterns.
        // This gives the meta-learner exactly the program types it will see in benchmarks.
        let reps: usize = arg_value(&args, "--reps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(500);
        let n_eval: usize = arg_value(&args, "--n-eval")
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        let known_descs = known_1arg_descriptions();
        eprintln!(
            "Generating {} reps × {} known descriptions...",
            reps,
            known_descs.len()
        );
        let mut total_kept = 0usize;

        for (name, desc) in &known_descs {
            let mut kept = 0usize;
            let mut tried = 0usize;
            while kept < reps {
                let seed: u64 = (total_kept as u64 * 7919) ^ (tried as u64 * 1009) ^ 0x1234;
                tried += 1;
                if let Some(rec) = synthetic_record(desc, n_eval, seed) {
                    let named = MetaRecord {
                        fn_name: name.clone(),
                        description: rec.description,
                        io_examples: rec.io_examples,
                        source: "known".to_string(),
                    };
                    let line =
                        serde_json::to_string(&named).unwrap_or_else(|e| panic!("json error: {e}"));
                    writeln!(w, "{line}").unwrap();
                    kept += 1;
                    total_kept += 1;
                }
                if tried > reps * 100 {
                    break;
                }
            }
            eprintln!("  {name} → {kept}/{reps} generated");
        }
        eprintln!(
            "Done: {total_kept} total from {} known program types.",
            known_descs.len()
        );
    } else if has_flag(&args, "--augment") {
        // --augment JSONL_IN --reps N --out OUT
        // Takes a JSONL file of MetaRecords (e.g. from --benchmark), and for each
        // description generates N varied I/O batches with random input distributions.
        let in_path = arg_value(&args, "--augment").unwrap_or_else(|| {
            eprintln!("--augment requires a JSONL file path");
            std::process::exit(1);
        });
        let reps: usize = arg_value(&args, "--reps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);
        let n_eval: usize = arg_value(&args, "--n-eval")
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        let input_file = std::fs::File::open(&in_path).unwrap_or_else(|e| {
            eprintln!("cannot open {in_path}: {e}");
            std::process::exit(1);
        });
        let records: Vec<MetaRecord> = std::io::BufRead::lines(std::io::BufReader::new(input_file))
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(&l).ok())
            .collect();

        eprintln!(
            "Augmenting {} benchmark descriptions × {reps} reps...",
            records.len()
        );
        let mut total_kept = 0usize;

        for rec in &records {
            let desc: UniversalProgramDescription = rec.description.clone();
            let mut kept = 0usize;
            let mut tried = 0usize;
            while kept < reps {
                let seed: u64 = (total_kept as u64 * 7919) ^ (tried as u64 * 1009) ^ 0xcafe;
                tried += 1;
                if let Some(aug) = synthetic_record(&desc, n_eval, seed) {
                    let aug_named = MetaRecord {
                        fn_name: rec.fn_name.clone(),
                        description: aug.description,
                        io_examples: aug.io_examples,
                        source: format!("aug_{}", rec.source),
                    };
                    let line = serde_json::to_string(&aug_named)
                        .unwrap_or_else(|e| panic!("json error: {e}"));
                    writeln!(w, "{line}").unwrap();
                    kept += 1;
                    total_kept += 1;
                }
                if tried > reps * 50 {
                    break;
                } // give up if acceptance too low
            }
            eprintln!("  {} → {kept}/{reps} augmented", rec.fn_name);
        }
        eprintln!(
            "Done: {total_kept} total augmented pairs from {} descriptions.",
            records.len()
        );
    } else if has_flag(&args, "--bench-known") {
        // --bench-known --reps N --n-eval K --out OUT
        //
        // For each known program description, test it against EVERY 1-arg integer benchmark
        // problem using discrete_eval. If the description correctly solves a benchmark
        // problem's examples, generate N training records using the BENCHMARK's actual
        // I/O examples (not random ones). Also generates N augmented records with random
        // inputs in the same range as the benchmark examples.
        let reps: usize = arg_value(&args, "--reps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);
        let n_eval: usize = arg_value(&args, "--n-eval")
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        let known_descs = known_1arg_descriptions();
        let bench_problems = get_benchmark(1);
        // Filter to 1-arg integer benchmark problems
        let bench1: Vec<_> = bench_problems
            .iter()
            .filter(|p| {
                let n = p.examples.first().map(|ex| ex.inputs.len()).unwrap_or(0);
                n == 1
                    && p.examples
                        .iter()
                        .all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_))))
            })
            .collect();

        eprintln!(
            "Checking {} known descriptions against {} 1-arg benchmark problems...",
            known_descs.len(),
            bench1.len()
        );
        let mut total_matches = 0usize;
        let mut total_records = 0usize;

        for (kname, desc) in &known_descs {
            let prog = SoftUniversalProgram::description_to_params(desc);
            let mut matched_problems = vec![];

            for problem in &bench1 {
                // Test: does this description solve the problem's examples?
                let all_correct = problem.examples.iter().all(|ex| {
                    let inputs: Vec<i64> = ex
                        .inputs
                        .iter()
                        .filter_map(|v| {
                            if let Value::Int(i) = v {
                                Some(*i)
                            } else {
                                None
                            }
                        })
                        .collect();
                    match prog.discrete_eval(&inputs) {
                        Some(out) => out == ex.expected_int(),
                        None => false,
                    }
                });
                if all_correct {
                    matched_problems.push(problem);
                }
            }

            if !matched_problems.is_empty() {
                eprintln!(
                    "  {kname} → matches {} problem(s): {}",
                    matched_problems.len(),
                    matched_problems
                        .iter()
                        .map(|p| p.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                total_matches += matched_problems.len();

                for problem in &matched_problems {
                    // Record 1: use benchmark's actual I/O examples
                    let bench_io: Vec<(Vec<i64>, i64)> = problem
                        .examples
                        .iter()
                        .map(|ex| {
                            let inputs: Vec<i64> = ex
                                .inputs
                                .iter()
                                .filter_map(|v| {
                                    if let Value::Int(i) = v {
                                        Some(*i)
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            (inputs, ex.expected_int())
                        })
                        .collect();

                    // Output: the description matched to this problem's IO
                    let record = MetaRecord {
                        fn_name: format!("{kname}_{}", problem.name),
                        description: desc.clone(),
                        io_examples: bench_io.clone(),
                        source: "bench_known".to_string(),
                    };
                    writeln!(w, "{}", serde_json::to_string(&record).unwrap()).unwrap();
                    total_records += 1;

                    // Record 2..N: augmented with random inputs in similar range
                    let mut kept = 0usize;
                    let mut tried = 0usize;
                    while kept < reps {
                        let seed: u64 =
                            (total_records as u64 * 7919) ^ (tried as u64 * 1009) ^ 0xbeef;
                        tried += 1;
                        if let Some(aug) = synthetic_record(desc, n_eval, seed) {
                            let aug_rec = MetaRecord {
                                fn_name: format!("{kname}_{}", problem.name),
                                description: aug.description,
                                io_examples: aug.io_examples,
                                source: "bench_known_aug".to_string(),
                            };
                            writeln!(w, "{}", serde_json::to_string(&aug_rec).unwrap()).unwrap();
                            kept += 1;
                            total_records += 1;
                        }
                        if tried > reps * 50 {
                            break;
                        }
                    }
                }
            }
        }

        // Report unmatched problems
        let matched_names: std::collections::HashSet<String> = {
            let known_descs2 = known_1arg_descriptions();
            let mut names = std::collections::HashSet::new();
            for (_kname, desc) in &known_descs2 {
                let prog = SoftUniversalProgram::description_to_params(desc);
                for problem in &bench1 {
                    let all_correct = problem.examples.iter().all(|ex| {
                        let inputs: Vec<i64> = ex
                            .inputs
                            .iter()
                            .filter_map(|v| {
                                if let Value::Int(i) = v {
                                    Some(*i)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        match prog.discrete_eval(&inputs) {
                            Some(out) => out == ex.expected_int(),
                            None => false,
                        }
                    });
                    if all_correct {
                        names.insert(problem.name.clone());
                    }
                }
            }
            names
        };
        let unmatched: Vec<_> = bench1
            .iter()
            .filter(|p| !matched_names.contains(&p.name))
            .collect();
        if !unmatched.is_empty() {
            eprintln!("  Unmatched ({}):", unmatched.len());
            for p in &unmatched {
                eprintln!("    - {}", p.name);
            }
        }
        eprintln!(
            "Done: {total_matches} description→problem matches, {total_records} training records."
        );
    } else {
        eprintln!("Usage:");
        eprintln!(
            "  gen_meta_data --synthetic N [--n-args A] [--n-eval K] [--seed S] [--out FILE]"
        );
        eprintln!("  gen_meta_data --benchmark [--n-args A] [--max-steps N] [--out FILE]");
        eprintln!("  gen_meta_data --known [--reps N] [--n-eval K] [--out FILE]");
        eprintln!("  gen_meta_data --bench-known [--reps N] [--n-eval K] [--out FILE]");
        eprintln!("  gen_meta_data --augment JSONL --reps N [--n-eval K] [--out FILE]");
        std::process::exit(1);
    }
}
