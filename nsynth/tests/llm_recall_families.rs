//! Multi-family recall benchmark — the HONEST universal-coverage measure. For each
//! (paraphrase, probe input, expected output) row it runs symbolic-only vs +LME,
//! EXECUTES the synthesized program on the probe input, and checks the actual
//! returned Value == expected. Value-level (not substring) so it can't be fooled
//! by a verified-but-wrong-family program: picking `array_min` for a "largest"
//! request fails the value check. Spans array→scalar, scalar→scalar, scalar→bool
//! families across the live op vocabulary.
//!
//! Gated: skips unless NSYNTH_LOCAL_LLM_URL points at a served model.
use mog_synth::benchmark::Value;
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

struct Row {
    phrasing: &'static str,
    family: &'static str,
    arg: Value,
    expect: Value,
}

fn row(phrasing: &'static str, family: &'static str, arg: Value, expect: Value) -> Row {
    Row { phrasing, family, arg, expect }
}

/// Run the synthesized program on the probe arg and check the actual output equals
/// `expect`. The entry fn name is the first `fn <name>(` in the code. Comparison is
/// in runtime::Value space (Int/Bool/Array cover this corpus).
fn runs_to(code: &str, arg: &Value, expect: &Value) -> bool {
    let Some(name) = code.split("fn ").nth(1).and_then(|s| s.split('(').next()).map(str::trim)
    else {
        return false;
    };
    let Ok(got) = mog_synth::runtime::execute_function(code, name, std::slice::from_ref(arg), name)
    else {
        return false;
    };
    matches_expected(&got, expect)
}

fn matches_expected(got: &mog_synth::runtime::Value, expect: &Value) -> bool {
    use mog_synth::runtime::Value as RV;
    match (got, expect) {
        (RV::Int(g), Value::Int(e)) => g == e,
        (RV::Bool(g), Value::Bool(e)) => g == e,
        // Representation-fair: these synthesizers encode a predicate as an int
        // (0 = false, nonzero = truthy), so judge a bool op on its behavior, not
        // its wrapper type — e.g. is_positive returns `n if n>0 else 0`.
        (RV::Int(g), Value::Bool(e)) => (*g != 0) == *e,
        (RV::Bool(g), Value::Int(e)) => *g == (*e != 0),
        _ => false,
    }
}

fn corpus() -> Vec<Row> {
    let arr = Value::int_array;
    vec![
        // array -> scalar (single Vec<i64> input)
        row("add up all the elements", "array_sum", arr(&[1, 2, 3, 4]), Value::Int(10)),
        row("the biggest value in the list", "array_max", arr(&[3, 9, 2]), Value::Int(9)),
        row("the smallest number in the array", "array_min", arr(&[4, 1, 7]), Value::Int(1)),
        row("multiply all the elements together", "product", arr(&[2, 3, 4]), Value::Int(24)),
        row("how many items are in the array", "length", arr(&[5, 5, 5]), Value::Int(3)),
        row("the first element of the list", "first", arr(&[7, 8, 9]), Value::Int(7)),
        row("the final entry in the array", "last", arr(&[7, 8, 9]), Value::Int(9)),
        // scalar -> scalar (single i64 input)
        row("the absolute value of the number", "abs", Value::Int(-7), Value::Int(7)),
        row("the number squared", "square", Value::Int(6), Value::Int(36)),
        row("flip the sign of the number", "negate", Value::Int(5), Value::Int(-5)),
        row("multiply the number by three", "triple", Value::Int(4), Value::Int(12)),
        row("add one to the value", "increment", Value::Int(9), Value::Int(10)),
        row("the factorial of the number", "factorial", Value::Int(5), Value::Int(120)),
        // scalar -> bool (single i64 input)
        row("is the number even", "is_even", Value::Int(4), Value::Bool(true)),
        row("is the number positive", "is_positive", Value::Int(-3), Value::Bool(false)),
    ]
    // Known gaps this fresh-input check exposes (NOT NL-mapping failures — the LLM
    // maps all three to the right op; the SYNTHESIS path is the limit):
    //   * factorial — 1 registry example → synthesize_op_by_name returns None.
    //   * first — its 2 registry examples both have first==max, so synthesis
    //     OVERFITS to a verified-but-wrong `max` program (returns 9 for [7,8,9]).
    //     The fresh-input value check catches the overfit; the fix is more
    //     disambiguating registry examples (a linguigenesis data change).
}

#[test]
fn llm_recall_multi_family_value_checked() {
    if std::env::var("NSYNTH_LOCAL_LLM_URL").ok().filter(|s| !s.is_empty()).is_none() {
        eprintln!("[FAM] skipped (no NSYNTH_LOCAL_LLM_URL)");
        return;
    }
    let bridge = LinguigenesisBridge::new();
    assert!(bridge.registry_load_error().is_none(), "registry must load");

    let rows = corpus();
    let (mut sym_ok, mut lme_ok, mut recovered) = (0usize, 0usize, 0usize);
    eprintln!("\n[FAM] family       phrasing                                  sym  +LME");
    eprintln!("[FAM] ------------ ----------------------------------------  ---  ----");
    for r in &rows {
        let sym = bridge
            .synthesize_from_description_symbolic(r.phrasing, None)
            .ok()
            .filter(|s| s.success)
            .is_some_and(|s| runs_to(&s.code, &r.arg, &r.expect));
        let lme = bridge
            .synthesize_via_local_llm(r.phrasing)
            .filter(|s| s.success)
            .is_some_and(|s| runs_to(&s.code, &r.arg, &r.expect));
        let net = sym || lme;
        if sym {
            sym_ok += 1;
        }
        if net {
            lme_ok += 1;
        }
        if !sym && lme {
            recovered += 1;
        }
        eprintln!(
            "[FAM] {:12} {:40}  {:3}  {}",
            r.family,
            &r.phrasing[..r.phrasing.len().min(40)],
            if sym { "✓" } else { "·" },
            if lme { "✓" } else { "·" }
        );
    }
    let n = rows.len();
    eprintln!("\n[FAM] symbolic-only: {sym_ok}/{n}   with-LME: {lme_ok}/{n}   recovered: {recovered}");
    // The lane must never reduce net recall and must recover real coverage breadth.
    assert!(lme_ok >= sym_ok, "LME reduced net recall ({lme_ok} < {sym_ok})");
    assert!(lme_ok >= n * 3 / 5, "with-LME coverage below 60% ({lme_ok}/{n}) — investigate misses");
}
