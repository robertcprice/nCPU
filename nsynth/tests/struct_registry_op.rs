//! FRONT-DOOR struct op: a registry entity whose example_cases are STRUCT
//! literals flows through parse_example_cases -> LiteralValue::Struct ->
//! Value::Struct -> the field-wise struct synthesizer, via the same
//! synthesize_op_by_name path every component leaf uses. Proves the registry
//! plumbing end to end (the last audit gap: "a struct op cannot be spec'd").

use mog_synth::linguigenesis_bridge::LinguigenesisBridge;

#[test]
fn registry_struct_op_synthesizes_fieldwise() {
    // A minimal base registry containing ONE struct-output op: bounds(x) ->
    // Bounds { hi: x+1, lo: x-1 } (canonical name order: hi < lo).
    let ex = |x: i64| {
        format!(
            r#"{{"inputs":[{{"type":"Int","value":{x}}}],"expected":{{"type":"Struct","value":{{"hi":{{"type":"Int","value":{}}},"lo":{{"type":"Int","value":{}}}}}}}}}"#,
            x + 1,
            x - 1
        )
    };
    let cases = format!("[{},{},{}]", ex(3), ex(10), ex(-2));
    let registry = serde_json::json!({
        "entities": {
            "bounds": {
                "word": "bounds",
                "entity_type": "function",
                "definitions": ["the bounds of a number"],
                "attributes": {
                    "arity": "1",
                    "input_types": "i64",
                    "output_type": "Bounds",
                    "default_fn_name": "bounds",
                    "signature_template": "fn {name}({params}) -> {return}",
                    "example_cases": cases,
                },
                "relations": {}
            }
        }
    })
    .to_string();
    let reg_path = std::env::temp_dir().join(format!(
        "nsynth_struct_reg_{}.json",
        std::process::id()
    ));
    std::fs::write(&reg_path, &registry).expect("write registry");
    std::env::set_var("NSYNTH_BASE_REGISTRY", &reg_path);

    let bridge = LinguigenesisBridge::new();
    let r = bridge
        .synthesize_op_by_name("bounds")
        .expect("struct op reachable by name");
    std::env::remove_var("NSYNTH_BASE_REGISTRY");
    let _ = std::fs::remove_file(&reg_path);

    assert!(r.success, "err: {:?}", r.error);
    assert!(
        r.method.starts_with("struct_fieldwise"),
        "expected the struct synthesizer, got: {}",
        r.method
    );
    assert!(r.code.contains("struct "), "struct decl emitted: {}", r.code);
    assert!(r.code.contains("return "), "constructor present: {}", r.code);
}
