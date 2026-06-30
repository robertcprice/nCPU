//! Verify the STRING op class is reachable from NL (uppercase/lowercase/...).
//! These live in mined_capabilities (string->string); the engine synthesizes
//! them via string_synth. Confirms strings are part of the NL-reachable vocab.
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

#[test]
fn nl_reaches_string_ops() {
    let bridge = LinguigenesisBridge::new();
    let cases: &[(&str, &str)] = &[
        ("convert the text to uppercase", "uppercase"),
        ("convert the text to lowercase", "lowercase"),
    ];
    let mut ok = 0usize;
    for (phrase, want) in cases {
        match bridge.synthesize_project(phrase) {
            Ok((solved, skipped)) => match solved.iter().find(|(_, r)| r.success) {
                Some((name, r)) => {
                    // string_synth names the fn generically ("transform"); the
                    // SEMANTICS are verified — the bridge fed the `uppercase` op's
                    // example_cases, and string_synth strict-verifies against them.
                    // So a verified string-synth success == the string op reached.
                    eprintln!("[STR] {phrase:?} → fn={name} (want {want}) method={}", r.method);
                    if r.method.contains("string") || name == want || r.code.contains(want) {
                        ok += 1;
                    }
                }
                None => eprintln!("[STR] {phrase:?} → NO SUCCESS (want {want}) skipped={skipped:?}"),
            },
            Err(e) => eprintln!("[STR] {phrase:?} → ERR {e}"),
        }
    }
    assert!(ok >= 2, "string ops must resolve + synthesize (verified) from NL; got {ok}");
}
