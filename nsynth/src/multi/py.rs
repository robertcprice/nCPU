//! Python Target for Multi-Language Code Generation

use super::{
    lang::{TargetLang, LanguageTarget, MogType, MogOp, common_type_map},
};

/// Python code generator
pub struct PythonTarget {
    types: bool,
    comments: bool,
    version: PythonVersion,
}

#[derive(Debug, Clone, Copy)]
pub enum PythonVersion {
    Python3_8,
    Python3_9,
    Python3_10,
    Python3_11,
}

impl Default for PythonTarget {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonTarget {
    pub fn new() -> Self {
        Self {
            types: false,
            comments: true,
            version: PythonVersion::Python3_10,
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

    pub fn with_version(mut self, version: PythonVersion) -> Self {
        self.version = version;
        self
    }

    fn needs_type_hints(&self) -> bool {
        self.types && matches!(self.version, PythonVersion::Python3_9 | PythonVersion::Python3_10 | PythonVersion::Python3_11)
    }
}

impl LanguageTarget for PythonTarget {
    fn type_map(&self, mog_type: &MogType) -> String {
        common_type_map(TargetLang::Python, mog_type)
    }

    fn op_map(&self, op: &MogOp) -> String {
        match op {
            MogOp::Negate => "-".to_string(),
            MogOp::Not => "not ".to_string(),
            MogOp::Abs => "abs".to_string(),
            MogOp::Sqrt => "math.sqrt".to_string(),
            MogOp::Log => "math.log".to_string(),
            MogOp::Exp => "math.exp".to_string(),
            MogOp::Sin => "math.sin".to_string(),
            MogOp::Cos => "math.cos".to_string(),
            MogOp::Tan => "math.tan".to_string(),

            MogOp::Add => "+".to_string(),
            MogOp::Sub => "-".to_string(),
            MogOp::Mul => "*".to_string(),
            MogOp::Div => "/".to_string(),
            MogOp::Mod => "%".to_string(),
            MogOp::Pow => "**".to_string(),
            MogOp::Eq => "==".to_string(),
            MogOp::Ne => "!=".to_string(),
            MogOp::Lt => "<".to_string(),
            MogOp::Le => "<=".to_string(),
            MogOp::Gt => ">".to_string(),
            MogOp::Ge => ">=".to_string(),
            MogOp::And => " and ".to_string(),
            MogOp::Or => " or ".to_string(),
            MogOp::BitAnd => "&".to_string(),
            MogOp::BitOr => "|".to_string(),
            MogOp::BitXor => "^".to_string(),
            MogOp::ShiftLeft => "<<".to_string(),
            MogOp::ShiftRight => ">>".to_string(),

            MogOp::If => "if".to_string(),
            MogOp::While => "while".to_string(),
            MogOp::For => "for".to_string(),
            MogOp::Loop => "while True".to_string(),
            MogOp::Break => "break".to_string(),
            MogOp::Continue => "continue".to_string(),

            MogOp::Call => "".to_string(),
            MogOp::Closure => "lambda".to_string(),
            MogOp::Recurse => "recurse".to_string(),

            MogOp::Array => "list".to_string(),
            MogOp::ArrayPush => ".append".to_string(),
            MogOp::ArrayPop => ".pop".to_string(),
            MogOp::ArrayLen => "len".to_string(),

            MogOp::Assign => "=".to_string(),
            MogOp::Let => "".to_string(),
            MogOp::Mut => "".to_string(),
            MogOp::Ref => "".to_string(),
            MogOp::Deref => "*".to_string(),

            MogOp::Print => "print".to_string(),
            MogOp::Println => "print".to_string(),
            MogOp::Return => "return".to_string(),

            _ => format!("{:?}", op),
        }
    }

    fn stdlib(&self) -> String {
        format!(r#"# Python Standard Library
import math
from typing import List, Tuple, Optional, Callable, Any
"#)
    }

    fn format_function(&self, name: &str, params: &[(String, MogType)], ret: &MogType, body: &str) -> String {
        let mut output = String::new();

        if self.comments {
            output.push_str("# Function ");
            output.push_str(name);
            output.push_str("\n");
        }

        output.push_str("def ");
        output.push_str(name);
        output.push_str("(");

        let param_strs: Vec<String> = params.iter()
            .map(|(n, ty)| {
                if self.needs_type_hints() {
                    format!("{}: {}", n, self.type_map(ty))
                } else {
                    n.clone()
                }
            })
            .collect();

        output.push_str(&param_strs.join(", "));

        if self.needs_type_hints() {
            output.push_str(") -> ");
            output.push_str(&self.type_map(ret));
        } else {
            output.push_str(")");
        }

        output.push_str(":\n");
        output.push_str(body);

        output
    }

    fn format_var(&self, name: &str, ty: &MogType, value: Option<&str>) -> String {
        let mut output = String::new();

        output.push_str(name);

        if let Some(v) = value {
            output.push_str(" = ");
            output.push_str(v);
        }

        if self.types {
            output.push_str(" # : ");
            output.push_str(&self.type_map(ty));
        }

        output
    }

    fn format_call(&self, func: &str, args: &[String]) -> String {
        format!("{}({})", func, args.join(", "))
    }

    fn format_if(&self, cond: &str, then_block: &str, else_block: Option<&str>) -> String {
        let mut output = String::new();

        output.push_str("if ");
        output.push_str(cond);
        output.push_str(":\n");
        output.push_str(then_block);

        if let Some(else_blk) = else_block {
            output.push_str("else:\n");
            output.push_str(else_blk);
        }

        output
    }

    fn format_while(&self, cond: &str, body: &str) -> String {
        let mut output = String::new();

        output.push_str("while ");
        output.push_str(cond);
        output.push_str(":\n");
        output.push_str(body);

        output
    }

    fn target(&self) -> TargetLang {
        TargetLang::Python
    }

    fn explicit_types(&self) -> bool {
        false // Python is dynamically typed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_target() {
        let target = PythonTarget::new();
        assert_eq!(target.target(), TargetLang::Python);

        assert_eq!(target.op_map(&MogOp::Add), "+");
        assert_eq!(target.op_map(&MogOp::Eq), "==");
        assert_eq!(target.op_map(&MogOp::And), " and ");
    }

    #[test]
    fn test_py_function_format() {
        let target = PythonTarget::new().with_types(true);
        let formatted = target.format_function(
            "add",
            &[("a".to_string(), MogType::Int), ("b".to_string(), MogType::Int)],
            &MogType::Int,
            "    return a + b\n",
        );

        assert!(formatted.contains("def add"));
        assert!(formatted.contains("a: int"));
        assert!(formatted.contains("-> int"));
    }

    #[test]
    fn test_py_stdlib() {
        let target = PythonTarget::new();
        let stdlib = target.stdlib();
        assert!(stdlib.contains("import math"));
        assert!(stdlib.contains("from typing import"));
    }
}
