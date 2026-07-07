//! Memorization-overfit detection for synthesized ops — the model-free, no-oracle
//! backstop for the SPECIFICATION WALL.
//!
//! A handful of examples do not DETERMINE a function. The solver can return a
//! program that reproduces every given example yet is wrong everywhere else. Two
//! defenses already exist and do NOT cover this last case:
//!
//!   * strict HOLDOUT (train/test split of the given examples) — USELESS here: the
//!     overfit is consistent with ALL given examples (it is the true function on
//!     that input subset); it only diverges at inputs the examples never covered,
//!     so every split finds the points mutually consistent. Proven on an LCG
//!     `scramble`: its affine closed form (minus the final mod) fits all 6 mined
//!     points and only breaks at x=5 / x=999.
//!   * SOURCE ORACLE (compile+run the repo's real code on fresh inputs,
//!     `foreign_op::eval_foreign_rust`) — the correct fix, but needs the source.
//!
//! When there is NO source and NO oracle, the only remaining model-free signal is
//! DESCRIPTION LENGTH (MDL / Occam). A genuine law COMPRESSES its data: `x*2`
//! explains any number of points with ~one bit of constant. A memorized fit does
//! not: `scramble`'s program encodes three 10-digit magic constants (~91 bits) to
//! reproduce 6 points (~186 bits) — the constants alone rival the data they
//! explain. That ratio is the tell, and it is exactly what separates a discovered
//! rule from a curve threaded through the points.
//!
//! This module is deliberately CONSERVATIVE — a false positive rejects a legit op
//! (a recall regression), so it fires only on the clear memorization signature:
//! TWO OR MORE genuinely large magic constants whose bits rival the output data.
//! A single large constant (e.g. a modulus `1_000_000_007`) never trips it.

use crate::benchmark::{Example, Value};

/// A constant is "magic" (a memorization candidate, not structural) only above
/// this magnitude. Bases, ASCII (256), small moduli, and type suffixes (`i64`,
/// `i128`) all fall below it and are exempt.
const MAGIC_MIN: i128 = 1 << 16; // 65_536

/// Fire only when the magic-constant bits are at least this fraction of the output
/// data bits — i.e. the program's constants carry a substantial share of the
/// information in the outputs they reproduce.
const OVERFIT_FRAC: f64 = 0.35;

/// Bits needed to encode a non-negative magnitude (0 -> 0 bits, 1 -> 1, 255 -> 8).
fn bits(mag: i128) -> f64 {
    let m = mag.unsigned_abs();
    if m == 0 {
        0.0
    } else {
        (128 - m.leading_zeros()) as f64
    }
}

/// Extract integer literals from `code`: maximal digit runs whose LEFT neighbor is
/// not an identifier char, so type suffixes (`i64`) and names (`x2`) are excluded.
/// Returns magnitudes (sign is irrelevant to encoding cost).
fn int_literals(code: &str) -> Vec<i128> {
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            // A digit run glued to the right of an identifier char (i64, x2) is not
            // a numeric literal.
            let prev_ident = i > 0 && (bytes[i - 1].is_ascii_alphabetic() || bytes[i - 1] == b'_');
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if !prev_ident {
                if let Ok(v) = code[start..i].parse::<i128>() {
                    out.push(v);
                }
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Total information (bits) in the example OUTPUTS — the data the program must
/// reproduce. Only integer outputs are counted (the synth domain here is `-> i64`).
fn data_bits(examples: &[Example]) -> (usize, f64) {
    let mut n = 0usize;
    let mut total = 0.0;
    for ex in examples {
        if let Value::Int(v) = ex.expected {
            n += 1;
            total += bits(v as i128).max(1.0); // each output carries >= 1 bit
        }
    }
    (n, total)
}

/// The magic-constant bits and count in `code`.
fn magic_bits(code: &str) -> (usize, f64) {
    let large: Vec<i128> = int_literals(code)
        .into_iter()
        .filter(|v| v.unsigned_abs() as i128 >= MAGIC_MIN)
        .collect();
    let bits_sum: f64 = large.iter().map(|&v| bits(v)).sum();
    (large.len(), bits_sum)
}

/// Compression margin: output data bits per magic-constant bit. `f64::INFINITY`
/// when the program uses no magic constants (a genuine law — maximal compression).
/// Below ~1 means the constants carry as much information as the data → memorized.
pub fn compression_margin(code: &str, examples: &[Example]) -> f64 {
    let (_, mbits) = magic_bits(code);
    if mbits <= 0.0 {
        return f64::INFINITY;
    }
    let (_, dbits) = data_bits(examples);
    dbits / mbits
}

/// True iff `code` looks like a MEMORIZED fit rather than a discovered law:
/// two or more large magic constants whose bits rival the output data. Conservative
/// by design — a single large constant (a modulus) or few examples never fires.
pub fn is_memorization_overfit(code: &str, examples: &[Example]) -> bool {
    let (n, dbits) = data_bits(examples);
    if n < 3 || dbits <= 0.0 {
        return false; // too little data to call anything a memorization
    }
    let (count, mbits) = magic_bits(code);
    if count < 2 {
        return false; // a lone large constant (e.g. a modulus) is not memorization
    }
    mbits >= OVERFIT_FRAC * dbits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(inputs: Vec<i64>, expected: i64) -> Example {
        Example {
            inputs: inputs.into_iter().map(Value::Int).collect(),
            expected: Value::Int(expected),
        }
    }

    #[test]
    fn flags_the_lcg_scramble_memorization() {
        // The overfit the specification wall produced live: an LCG affine closed
        // form with three 10-digit magic constants fit to 6 points.
        let code = "fn scramble(x: i64) -> i64 {\n    return (((-301564143 * x) + (2147483648 * (x / 7))) + 1449466924);\n}\n";
        let examples = vec![
            ex(vec![0], 1449466924),
            ex(vec![1], 1147902781),
            ex(vec![2], 846338638),
            ex(vec![3], 544774495),
            ex(vec![7], 1486001571),
            ex(vec![42], 1668674806),
        ];
        assert!(
            is_memorization_overfit(code, &examples),
            "3 magic constants explaining 6 points must be flagged as memorization"
        );
    }

    #[test]
    fn passes_a_genuine_law() {
        // x*2 explains any number of points with no magic constant — a real law.
        let code = "fn times_two(x: i64) -> i64 {\n    return (x * 2);\n}\n";
        let examples = vec![ex(vec![2], 4), ex(vec![5], 10), ex(vec![100], 200), ex(vec![7], 14)];
        assert!(!is_memorization_overfit(code, &examples));
        assert_eq!(compression_margin(code, &examples), f64::INFINITY);
    }

    #[test]
    fn passes_a_single_large_modulus() {
        // A lone big constant (a legitimate modulus) is NOT memorization — one
        // structural constant, however large, does not thread a curve through points.
        let code = "fn hash1(x: i64) -> i64 {\n    return ((x * x) % 1000000007);\n}\n";
        let examples = vec![
            ex(vec![3], 9),
            ex(vec![10], 100),
            ex(vec![50000], 344500001),
            ex(vec![99999], 1005999999 % 1000000007),
        ];
        assert!(!is_memorization_overfit(code, &examples), "single modulus must not be flagged");
    }

    #[test]
    fn passes_small_structural_constants() {
        // %10, /10, base 256 — all below the magic threshold, never counted.
        let code = "fn sum_digits(x: i64) -> i64 {\n    let mut s = 0;\n    let mut n = x;\n    while n > 0 { s = s + (n % 10); n = n / 10; }\n    return s;\n}\n";
        let examples = vec![ex(vec![123], 6), ex(vec![9], 9), ex(vec![38], 11), ex(vec![9999], 36)];
        assert!(!is_memorization_overfit(code, &examples));
    }

    #[test]
    fn ignores_type_suffix_digits() {
        // `i64` / `i128` must not be read as the literals 64 / 128.
        let lits = int_literals("fn f(x: i64) -> i128 { return x; }");
        assert!(!lits.contains(&64) && !lits.contains(&128), "type suffixes are not literals: {lits:?}");
    }

    #[test]
    fn too_few_examples_never_fires() {
        // Two points can be threaded by anything — but with so little data we make
        // NO claim (the caller should gather more), rather than false-flagging.
        let code = "fn f(x: i64) -> i64 { return ((123456789 * x) + 987654321); }";
        let examples = vec![ex(vec![0], 987654321), ex(vec![1], 1111111110)];
        assert!(!is_memorization_overfit(code, &examples));
    }
}
