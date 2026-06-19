//! TypeScript Target for Multi-Language Code Generation

use super::{
    lang::{TargetLang, LanguageTarget, MogType, MogOp, common_type_map},
    js::JavaScriptTarget,
};

/// TypeScript code generator
pub struct TypeScriptTarget {
    js: JavaScriptTarget,
    strict_types: bool,
    strict_null_checks: bool,
}

impl Default for TypeScriptTarget {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeScriptTarget {
    pub fn new() -> Self {
        Self {
            js: JavaScriptTarget::new().with_types(true),
            strict_types: true,
            strict_null_checks: true,
        }
    }

    pub fn with_strict_types(mut self, strict: bool) -> Self {
        self.strict_types = strict;
        self
    }

    pub fn with_strict_null_checks(mut self, strict: bool) -> Self {
        self.strict_null_checks = strict;
        self
    }

    fn ts_type_map(&self, mog_type: &MogType) -> String {
        match mog_type {
            MogType::Unit => {
                if self.strict_null_checks {
                    "void | null".to_string()
                } else {
                    "void".to_string()
                }
            }
            MogType::Array(inner) => {
                let inner_str = self.ts_type_map(inner);
                format!("{}[]", inner_str)
            }
            MogType::Function(params, ret) => {
                let param_str: Vec<String> = params.iter()
                    .map(|p| format!("_: {}", self.ts_type_map(p)))
                    .collect();
                let ret_str = self.ts_type_map(ret);
                format!("({}) => {}", param_str.join(", "), ret_str)
            }
            MogType::Tuple(types) => {
                let types_str: Vec<String> = types.iter()
                    .map(|t| self.ts_type_map(t))
                    .collect();
                format!("[{}]", types_str.join(", "))
            }
            _ => common_type_map(TargetLang::TypeScript, mog_type),
        }
    }
}

impl LanguageTarget for TypeScriptTarget {
    fn type_map(&self, mog_type: &MogType) -> String {
        self.ts_type_map(mog_type)
    }

    fn op_map(&self, op: &MogOp) -> String {
        self.js.op_map(op)
    }

    fn stdlib(&self) -> String {
        format!(r#"// TypeScript Standard Library
// @ts-check
{}"#, self.js.stdlib())
    }

    fn format_function(&self, name: &str, params: &[(String, MogType)], ret: &MogType, body: &str) -> String {
        let mut output = String::new();

        // Add JSDoc comment
        output.push_str("/**\n");
        output.push_str(&format!(" * Function {}\n", name));
        output.push_str(&format!(" * @returns {{{}}}\n", self.type_map(ret)));
        output.push_str(" */\n");

        // Function signature
        output.push_str("function ");
        output.push_str(name);
        output.push_str("(");

        let param_strs: Vec<String> = params.iter()
            .map(|(n, ty)| format!("{}: {}", n, self.type_map(ty)))
            .collect();

        output.push_str(&param_strs.join(", "));
        output.push_str("): ");
        output.push_str(&self.type_map(ret));
        output.push_str(" {\n");
        output.push_str(body);
        output.push_str("}");

        output
    }

    fn format_var(&self, name: &str, ty: &MogType, value: Option<&str>) -> String {
        let mut output = String::new();

        output.push_str("let ");
        output.push_str(name);
        output.push_str(": ");
        output.push_str(&self.type_map(ty));

        if let Some(v) = value {
            output.push_str(" = ");
            output.push_str(v);
        }

        output
    }

    fn format_call(&self, func: &str, args: &[String]) -> String {
        self.js.format_call(func, args)
    }

    fn format_if(&self, cond: &str, then_block: &str, else_block: Option<&str>) -> String {
        self.js.format_if(cond, then_block, else_block)
    }

    fn format_while(&self, cond: &str, body: &str) -> String {
        self.js.format_while(cond, body)
    }

    fn target(&self) -> TargetLang {
        TargetLang::TypeScript
    }

    fn explicit_types(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ts_target() {
        let target = TypeScriptTarget::new();
        assert_eq!(target.target(), TargetLang::TypeScript);

        assert_eq!(target.op_map(&MogOp::Add), "+");
        assert_eq!(target.op_map(&MogOp::Eq), "===");
    }

    #[test]
    fn test_ts_type_map() {
        let target = TypeScriptTarget::new();
        assert_eq!(target.type_map(&MogType::Int), "number");
        assert_eq!(target.type_map(&MogType::String), "string");
        assert_eq!(target.type_map(&MogType::Array(Box::new(MogType::Int))), "number[]");
    }

    #[test]
    fn test_ts_function_format() {
        let target = TypeScriptTarget::new();
        let formatted = target.format_function(
            "add",
            &[("a".to_string(), MogType::Int), ("b".to_string(), MogType::Int)],
            &MogType::Int,
            "    return a + b;\n",
        );

        assert!(formatted.contains("function add"));
        assert!(formatted.contains("a: number"));
        assert!(formatted.contains("): number"));
        assert!(formatted.contains("@returns {number}"));
    }

    #[test]
    fn test_ts_var_format() {
        let target = TypeScriptTarget::new();
        let formatted = target.format_var("x", &MogType::Int, Some("42"));

        assert!(formatted.contains("let x: number = 42"));
    }
}
