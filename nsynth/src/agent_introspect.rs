//! Runtime capability introspection — derived from the live registry, miner
//! overlay, prose router, sandbox policy, and tensor surface. No hand-maintained
//! capability marketing lists.

use crate::agent::tools::secure_runtime::SecureToolRuntime;
use crate::agent::GuardrailPolicy;
use crate::backend_intake::prose_door_catalog;
use crate::linguigenesis_bridge::LinguigenesisBridge;
use crate::tensor_nl::{classify, forward_ops, is_training_request, TensorRouteOutcome};
use linguigenesis_core::coding_requirements::parse_example_cases;
use linguigenesis_core::entity_resolution::is_operation;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

/// JSON snapshot of what the engine can reach right now (registry-derived).
pub fn engine_capabilities_json(root: &Path, policy: GuardrailPolicy) -> Value {
    let bridge = LinguigenesisBridge::new();
    let sandbox = SecureToolRuntime::for_general_agent(root, policy);

    let mut registry_summary = json!({
        "loaded": false,
        "entity_count": 0,
        "operations_with_example_cases": 0,
        "output_type_counts": {},
        "load_error": bridge.registry_load_error(),
    });

    if let Ok(registry) = bridge.registry_clone() {
        let mut output_type_counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut ops_with_examples = 0usize;
        let mut entity_count = 0usize;
        for entity in registry.all_entities() {
            entity_count += 1;
            if !is_operation(&entity) {
                continue;
            }
            if entity.get_property("output_type").is_some()
                || entity.get_property("input_types").is_some()
            {
                let out = entity
                    .get_property("output_type")
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                *output_type_counts.entry(out).or_default() += 1;
            }
            if !parse_example_cases(&entity).is_empty() {
                ops_with_examples += 1;
            }
        }
        registry_summary = json!({
            "loaded": true,
            "entity_count": entity_count,
            "operations_with_example_cases": ops_with_examples,
            "output_type_counts": output_type_counts,
            "load_error": bridge.registry_load_error(),
        });
    }

    let mined_path = LinguigenesisBridge::data_file_path("mined_capabilities.json");
    let mined_summary = mined_path
        .as_ref()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .map(|doc| {
            let entities = doc
                .get("entities")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            json!({
                "path": mined_path.as_ref().map(|p| p.display().to_string()),
                "entity_count": entities,
            })
        })
        .unwrap_or_else(|| {
            json!({
                "path": mined_path.as_ref().map(|p| p.display().to_string()),
                "entity_count": 0,
                "warning": "mined_capabilities.json unavailable",
            })
        });

    let tensor_ops: Vec<Value> = forward_ops()
        .iter()
        .map(|op| {
            json!({
                "lemma": op.lemma,
                "arity": op.arity,
                "requires_tensor_context": op.requires_tensor_context,
                "provenance": op.provenance,
            })
        })
        .collect();

    let prose_doors: Vec<Value> = prose_door_catalog()
        .iter()
        .map(|(tag, mechanism)| json!({ "tag": tag, "mechanism": mechanism }))
        .collect();

    json!({
        "source": "runtime_introspection",
        "registry": registry_summary,
        "mined_capabilities": mined_summary,
        "prose_backend_doors": prose_doors,
        "sandbox_tools": sandbox.allowed_capabilities(),
        "tensor": {
            "forward_ops": tensor_ops,
            "training_request_gate_active": is_training_request("train a model"),
            "sample_training_refusal": match classify("train a model") {
                TensorRouteOutcome::RefuseTraining => "refused",
                _ => "not_training",
            },
        },
        "communication_notes": [
            "capabilities are derived from the loaded registry + miner overlay, not a static list",
            "call agent_query to probe a specific NL request; route and synthesis_method report what fired",
            "honest refusals return success=false with an explanatory response string",
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn introspection_json_has_registry_and_doors() {
        let root = env::temp_dir().join(format!("nsynth_intro_{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let doc = engine_capabilities_json(&root, GuardrailPolicy::default());
        assert_eq!(doc["source"], "runtime_introspection");
        assert!(doc["prose_backend_doors"].as_array().unwrap().len() >= 6);
        assert!(doc["sandbox_tools"].as_array().unwrap().len() >= 5);
    }
}
