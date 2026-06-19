//! JavaScript Target for Multi-Language Code Generation

use super::{
    lang::{TargetLang, LanguageTarget, MogType, MogOp, common_type_map},
};
use std::collections::HashMap;

/// JavaScript code generator
pub struct JavaScriptTarget {
    types: bool,
    comments: bool,
    minify: bool,
}

impl JavaScriptTarget {
    pub fn new() -> Self {
        Self {
            types: false,
            comments: true,
            minify: false,
        }
    }

    pub fn with_types(mut self, types: bool) -> Self {
        self.types = types;
        self
    }

    pub fn with_comments(mut self, comments: bool) -> Self {
        self.comments = comments;
        self
    }

    pub fn with_minify(mut self, minify: bool) -> Self {
        self.minify = minify;
        self
    }

    fn add_jsdoc(&self, output: &mut String, text: &str) {
        if self.comments && !self.minify {
            output.push_str("/** ");
            output.push_str(text);
            output.push_str(" */\n");
        }
    }
}

impl Default for JavaScriptTarget {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageTarget for JavaScriptTarget {
    fn type_map(&self, mog_type: &MogType) -> String {
        common_type_map(TargetLang::JavaScript, mog_type)
    }

    fn op_map(&self, op: &MogOp) -> String {
        match op {
            MogOp::Negate => "-".to_string(),
            MogOp::Not => "!".to_string(),
            MogOp::Abs => "Math.abs".to_string(),
            MogOp::Sqrt => "Math.sqrt".to_string(),
            MogOp::Log => "Math.log".to_string(),
            MogOp::Exp => "Math.exp".to_string(),
            MogOp::Sin => "Math.sin".to_string(),
            MogOp::Cos => "Math.cos".to_string(),
            MogOp::Tan => "Math.tan".to_string(),

            MogOp::Add => "+".to_string(),
            MogOp::Sub => "-".to_string(),
            MogOp::Mul => "*".to_string(),
            MogOp::Div => "/".to_string(),
            MogOp::Mod => "%".to_string(),
            MogOp::Pow => "**".to_string(),
            MogOp::Eq => "===".to_string(),
            MogOp::Ne => "!==".to_string(),
            MogOp::Lt => "<".to_string(),
            MogOp::Le => "<=".to_string(),
            MogOp::Gt => ">".to_string(),
            MogOp::Ge => ">=".to_string(),
            MogOp::And => "&&".to_string(),
            MogOp::Or => "||".to_string(),
            MogOp::BitAnd => "&".to_string(),
            MogOp::BitOr => "|".to_string(),
            MogOp::BitXor => "^".to_string(),
            MogOp::ShiftLeft => "<<".to_string(),
            MogOp::ShiftRight => ">>".to_string(),

            MogOp::If => "if".to_string(),
            MogOp::While => "while".to_string(),
            MogOp::For => "for".to_string(),
            MogOp::Loop => "while".to_string(),
            MogOp::Break => "break".to_string(),
            MogOp::Continue => "continue".to_string(),

            MogOp::Call => "call".to_string(),
            MogOp::Closure => "=>".to_string(),
            MogOp::Recurse => "recurse".to_string(),

            MogOp::Array => "[]".to_string(),
            MogOp::ArrayPush => ".push".to_string(),
            MogOp::ArrayPop => ".pop".to_string(),
            MogOp::ArrayLen => ".length".to_string(),

            MogOp::Assign => "=".to_string(),
            MogOp::Let => "let".to_string(),
            MogOp::Mut => "let".to_string(),
            MogOp::Ref => "&".to_string(),
            MogOp::Deref => "*".to_string(),

            MogOp::Print => "console.log".to_string(),
            MogOp::Println => "console.log".to_string(),
            MogOp::Return => "return".to_string(),

            _ => format!("{:?}", op), // Fallback using Debug
        }
    }

    fn stdlib(&self) -> String {
        if self.minify {
            String::new()
        } else {
            "// JavaScript Standard Library\n".to_string()
        }
    }

    fn format_function(&self, name: &str, params: &[(String, MogType)], ret: &MogType, body: &str) -> String {
        let mut output = String::new();

        self.add_jsdoc(&mut output, &format!("Function {}", name));

        let param_list: Vec<String> = params.iter()
            .map(|(n, _)| n.clone())
            .collect();

        output.push_str("function ");
        output.push_str(name);
        output.push_str("(");
        output.push_str(&param_list.join(", "));
        output.push_str(") {\n");
        output.push_str(body);
        output.push_str("}");

        if self.minify {
            // Remove newlines for minified output
            output = output.replace('\n', "");
        }

        output
    }

    fn format_var(&self, name: &str, ty: &MogType, value: Option<&str>) -> String {
        let mut output = String::new();

        output.push_str("let ");
        output.push_str(name);

        if let Some(v) = value {
            output.push_str(" = ");
            output.push_str(v);
        }

        if self.types {
            output.push_str(" // : ");
            output.push_str(&self.type_map(ty));
        }

        output
    }

    fn format_call(&self, func: &str, args: &[String]) -> String {
        format!("{}({})", func, args.join(", "))
    }

    fn format_if(&self, cond: &str, then_block: &str, else_block: Option<&str>) -> String {
        let mut output = String::new();

        output.push_str("if (");
        output.push_str(cond);
        output.push_str(") {\n");
        output.push_str(then_block);
        output.push_str("}");

        if let Some(else_blk) = else_block {
            output.push_str(" else {\n");
            output.push_str(else_blk);
            output.push_str("}");
        }

        output
    }

    fn format_while(&self, cond: &str, body: &str) -> String {
        let mut output = String::new();

        output.push_str("while (");
        output.push_str(cond);
        output.push_str(") {\n");
        output.push_str(body);
        output.push_str("}");

        output
    }

    fn target(&self) -> TargetLang {
        TargetLang::JavaScript
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_js_target() {
        let target = JavaScriptTarget::new();
        assert_eq!(target.target(), TargetLang::JavaScript);

        assert_eq!(target.op_map(&MogOp::Add), "+");
        assert_eq!(target.op_map(&MogOp::Eq), "===");
    }

    #[test]
    fn test_js_function_format() {
        let target = JavaScriptTarget::new();
        let formatted = target.format_function(
            "add",
            &[("a".to_string(), MogType::Int), ("b".to_string(), MogType::Int)],
            &MogType::Int,
            "    return a + b;\n",
        );

        assert!(formatted.contains("function add"));
        assert!(formatted.contains("a, b"));
    }

    #[test]
    fn test_js_var_format() {
        let target = JavaScriptTarget::new();
        let formatted = target.format_var("x", &MogType::Int, Some("42"));

        assert!(formatted.contains("let x = 42"));
    }
}
