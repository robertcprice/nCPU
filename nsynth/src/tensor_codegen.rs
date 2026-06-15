//! Stage 2: Tensor Code Generation for nsynth.
//!
//! This module synthesizes Mog programs that operate on multi-dimensional
//! tensors with shape constraints and dtype tracking. Stage 2 extends
//! Stage 1 (scalar/array) with:
//!
//! **Supported Mog Tensor Operations:**
//! - `broadcast(scalar: int) -> tensor<N, i64>` — replicate scalar to all N elements
//! - `dot(a: tensor<N, i64>, b: tensor<N, i64>) -> i64` — element-wise product sum
//! - `matmul(a: tensor<N, M>, b: tensor<M, K>) -> tensor<N, K>` — standard MM
//! - `transpose(a: tensor<N, M>) -> tensor<M, N>` — swap dimensions
//! - `slice_row(a: tensor<N, M>, i: int) -> tensor<M, i64>` — extract row i
//!
//! **Emitted Syntax:**
//! ```text
//! let t1 = broadcast(3) as tensor<4, i64>;
//! let t2 = input_arr as tensor<4, i64>;
//! let dot_result = dot(t1, t2) as i64;
//! let mm_result = matmul(matrix_a, matrix_b) as tensor<3, 4>;
//! let row_slice = slice_row(matrix_c, 0) as tensor<5, i64>;
//! ```
//!
//! **Architecture:**
//! - `TensorType`: Tracks shape dims and element dtype
//! - `TensorCodegen`: Stateful builder emitting properly-scoped Mog programs
//! - Individual `codegen_*` functions for each op (broadcast, dot, matmul, transpose, slice_row)
//! - Type-safe tensor variable binding with alias tracking
//!
//! **Integration Points:**
//! - Called from `solver::solve_problem` when problem input/output includes tensor types
//! - Composes with existing scalar/array gradients via problem embedding
//! - Validates emitted code via `runtime::verify_problem_code_strict`

use std::collections::HashMap;

/// Represents a multi-dimensional tensor's static shape and element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    /// Ordered dimension sizes (e.g., [3, 4] for 3x4 matrix, [5] for vector).
    pub shape: Vec<usize>,
    /// Element data type: "i64" or "f64" (extensible).
    pub dtype: String,
}

impl TensorType {
    /// Construct a new tensor type with given shape and dtype.
    pub fn new(shape: Vec<usize>, dtype: impl Into<String>) -> Self {
        TensorType {
            shape,
            dtype: dtype.into(),
        }
    }

    /// Shorthand for a 1D vector of given length.
    pub fn vector(len: usize) -> Self {
        TensorType {
            shape: vec![len],
            dtype: "i64".to_string(),
        }
    }

    /// Shorthand for a 2D matrix of given rows × cols.
    pub fn matrix(rows: usize, cols: usize) -> Self {
        TensorType {
            shape: vec![rows, cols],
            dtype: "i64".to_string(),
        }
    }

    /// Total element count (product of shape dims).
    pub fn total_elements(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    /// Format as Mog type annotation: `tensor<3, 4, i64>`.
    pub fn mog_annotation(&self) -> String {
        let dims = self.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
        format!("tensor<{}, {}>", dims, self.dtype)
    }
}

/// Stateful code generator for tensor operations.
///
/// Tracks emitted variable bindings, maintains a stable variable counter,
/// and constructs properly-scoped Mog function bodies.
#[derive(Debug, Clone)]
pub struct TensorCodegen {
    /// Accumulated program lines in order.
    lines: Vec<String>,
    /// Mapping from variable name to its declared `TensorType`.
    variables: HashMap<String, TensorType>,
    /// Counter for generated temporary variable names.
    var_counter: usize,
}

impl TensorCodegen {
    /// Create a new empty tensor code generator.
    pub fn new() -> Self {
        TensorCodegen {
            lines: Vec::new(),
            variables: HashMap::new(),
            var_counter: 0,
        }
    }

    /// Generate a fresh variable name (e.g., `t_0`, `t_1`).
    fn fresh_var(&mut self) -> String {
        let name = format!("t_{}", self.var_counter);
        self.var_counter += 1;
        name
    }

    /// Register a variable with its tensor type.
    fn bind_var(&mut self, name: &str, ty: TensorType) {
        self.variables.insert(name.to_string(), ty);
    }

    /// Emit a line of code and return the variable name for the result.
    fn emit(&mut self, line: String) -> String {
        self.lines.push(line);
        // The last var_counter-1 is the one just created by fresh_var.
        format!("t_{}", self.var_counter - 1)
    }

    /// Finalize and return the complete Mog function body.
    pub fn finish(&self) -> String {
        self.lines.join("\n")
    }

    /// **Public API: Broadcast a scalar to fill a tensor.**
    ///
    /// ```text
    /// let t_0 = broadcast(5) as tensor<4, i64>;
    /// ```
    pub fn broadcast(&mut self, scalar: i64, target_shape: Vec<usize>) -> Result<String, String> {
        let target_type = TensorType::new(target_shape, "i64");
        let var_name = self.fresh_var();
        let line = format!(
            "let {} = broadcast({}) as {};",
            var_name,
            scalar,
            target_type.mog_annotation()
        );
        self.lines.push(line);
        self.bind_var(&var_name, target_type);
        Ok(var_name)
    }

    /// **Public API: Compute dot product of two 1D tensors.**
    ///
    /// Both tensors must have matching shapes. Result is a scalar (i64).
    ///
    /// ```text
    /// let t_0 = dot(a, b) as i64;
    /// ```
    pub fn dot(&mut self, a: &str, b: &str) -> Result<String, String> {
        let a_type = self.variables.get(a)
            .ok_or_else(|| format!("variable {} not found", a))?;
        let b_type = self.variables.get(b)
            .ok_or_else(|| format!("variable {} not found", b))?;

        if a_type.shape != b_type.shape {
            return Err(format!(
                "dot: shape mismatch: {} vs {}",
                a_type.mog_annotation(),
                b_type.mog_annotation()
            ));
        }

        let var_name = self.fresh_var();
        let line = format!("let {} = dot({}, {}) as i64;", var_name, a, b);
        self.lines.push(line);
        self.bind_var(&var_name, TensorType::new(vec![], "i64")); // scalar
        Ok(var_name)
    }

    /// **Public API: Matrix multiply A (N×M) × B (M×K) → result (N×K).**
    ///
    /// ```text
    /// let t_0 = matmul(a, b) as tensor<3, 4>;
    /// ```
    pub fn matmul(&mut self, a: &str, b: &str) -> Result<String, String> {
        let a_type = self.variables.get(a)
            .ok_or_else(|| format!("variable {} not found", a))?;
        let b_type = self.variables.get(b)
            .ok_or_else(|| format!("variable {} not found", b))?;

        if a_type.shape.len() != 2 || b_type.shape.len() != 2 {
            return Err(format!(
                "matmul: both inputs must be 2D matrices, got {} and {}",
                a_type.mog_annotation(),
                b_type.mog_annotation()
            ));
        }

        let a_cols = a_type.shape[1];
        let b_rows = b_type.shape[0];
        if a_cols != b_rows {
            return Err(format!(
                "matmul: inner dimensions mismatch: {} cols of A vs {} rows of B",
                a_cols, b_rows
            ));
        }

        let result_shape = vec![a_type.shape[0], b_type.shape[1]];
        let result_type = TensorType::new(result_shape, "i64");

        let var_name = self.fresh_var();
        let line = format!(
            "let {} = matmul({}, {}) as {};",
            var_name,
            a,
            b,
            result_type.mog_annotation()
        );
        self.lines.push(line);
        self.bind_var(&var_name, result_type);
        Ok(var_name)
    }

    /// **Public API: Transpose a 2D matrix.**
    ///
    /// ```text
    /// let t_0 = transpose(a) as tensor<4, 3>;
    /// ```
    pub fn transpose(&mut self, a: &str) -> Result<String, String> {
        let a_type = self.variables.get(a)
            .ok_or_else(|| format!("variable {} not found", a))?;

        if a_type.shape.len() != 2 {
            return Err(format!(
                "transpose: input must be 2D, got {}",
                a_type.mog_annotation()
            ));
        }

        let result_shape = vec![a_type.shape[1], a_type.shape[0]];
        let result_type = TensorType::new(result_shape, "i64");

        let var_name = self.fresh_var();
        let line = format!(
            "let {} = transpose({}) as {};",
            var_name,
            a,
            result_type.mog_annotation()
        );
        self.lines.push(line);
        self.bind_var(&var_name, result_type);
        Ok(var_name)
    }

    /// **Public API: Extract a single row from a 2D matrix.**
    ///
    /// ```text
    /// let t_0 = slice_row(matrix, 1) as tensor<5, i64>;
    /// ```
    pub fn slice_row(&mut self, matrix: &str, row_idx: usize) -> Result<String, String> {
        let mat_type = self.variables.get(matrix)
            .ok_or_else(|| format!("variable {} not found", matrix))?;

        if mat_type.shape.len() != 2 {
            return Err(format!(
                "slice_row: input must be 2D, got {}",
                mat_type.mog_annotation()
            ));
        }

        if row_idx >= mat_type.shape[0] {
            return Err(format!(
                "slice_row: row index {} out of bounds (matrix has {} rows)",
                row_idx, mat_type.shape[0]
            ));
        }

        let result_shape = vec![mat_type.shape[1]];
        let result_type = TensorType::new(result_shape, "i64");

        let var_name = self.fresh_var();
        let line = format!(
            "let {} = slice_row({}, {}) as {};",
            var_name,
            matrix,
            row_idx,
            result_type.mog_annotation()
        );
        self.lines.push(line);
        self.bind_var(&var_name, result_type);
        Ok(var_name)
    }

    /// Get the type of a variable (for consumers to understand result shapes).
    pub fn get_type(&self, var_name: &str) -> Option<TensorType> {
        self.variables.get(var_name).cloned()
    }

    /// Get all registered variables and their types.
    pub fn all_variables(&self) -> &HashMap<String, TensorType> {
        &self.variables
    }
}

impl Default for TensorCodegen {
    fn default() -> Self {
        Self::new()
    }
}

/// Standalone function: generate complete Mog code for a broadcast operation.
///
/// This is used by higher-level synthesis algorithms.
pub fn codegen_broadcast(scalar: i64, target_shape: Vec<usize>, output_var: &str) -> String {
    let shape_str = target_shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ");
    format!(
        "let {} = broadcast({}) as tensor<{}, i64>;",
        output_var, scalar, shape_str
    )
}

/// Standalone function: generate complete Mog code for a dot product.
pub fn codegen_dot(a_var: &str, b_var: &str, output_var: &str) -> String {
    format!("let {} = dot({}, {}) as i64;", output_var, a_var, b_var)
}

/// Standalone function: generate complete Mog code for matrix multiplication.
///
/// Caller is responsible for ensuring shapes are compatible (a cols == b rows).
pub fn codegen_matmul(
    a_var: &str,
    b_var: &str,
    result_rows: usize,
    result_cols: usize,
    output_var: &str,
) -> String {
    format!(
        "let {} = matmul({}, {}) as tensor<{}, {}>;",
        output_var, a_var, b_var, result_rows, result_cols
    )
}

/// Standalone function: generate complete Mog code for transpose.
pub fn codegen_transpose(a_var: &str, result_rows: usize, result_cols: usize, output_var: &str) -> String {
    format!(
        "let {} = transpose({}) as tensor<{}, {}>;",
        output_var, a_var, result_rows, result_cols
    )
}

/// Standalone function: generate complete Mog code for row slicing.
pub fn codegen_slice_row(matrix_var: &str, row_idx: usize, num_cols: usize, output_var: &str) -> String {
    format!(
        "let {} = slice_row({}, {}) as tensor<{}, i64>;",
        output_var, matrix_var, row_idx, num_cols
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_type_vector() {
        let vec = TensorType::vector(5);
        assert_eq!(vec.shape, vec![5]);
        assert_eq!(vec.dtype, "i64");
        assert_eq!(vec.mog_annotation(), "tensor<5, i64>");
    }

    #[test]
    fn test_tensor_type_matrix() {
        let mat = TensorType::matrix(3, 4);
        assert_eq!(mat.shape, vec![3, 4]);
        assert_eq!(mat.mog_annotation(), "tensor<3, 4, i64>");
    }

    #[test]
    fn test_tensor_type_total_elements() {
        assert_eq!(TensorType::vector(5).total_elements(), 5);
        assert_eq!(TensorType::matrix(3, 4).total_elements(), 12);
    }

    #[test]
    fn test_codegen_broadcast() {
        let mut gen = TensorCodegen::new();
        let result = gen.broadcast(42, vec![4, 5]).unwrap();
        let code = gen.finish();
        assert!(code.contains("broadcast(42)"));
        assert!(code.contains("tensor<4, 5, i64>"));
        assert!(code.contains(&format!("let {} =", result)));
    }

    #[test]
    fn test_codegen_dot_product() {
        let mut gen = TensorCodegen::new();
        gen.broadcast(1, vec![3]).unwrap();
        gen.broadcast(2, vec![3]).unwrap();
        let _result = gen.dot("t_0", "t_1").unwrap();
        let code = gen.finish();
        assert!(code.contains("dot(t_0, t_1)"));
        assert!(code.contains("as i64"));
    }

    #[test]
    fn test_codegen_dot_shape_mismatch() {
        let mut gen = TensorCodegen::new();
        gen.broadcast(1, vec![3]).unwrap();
        gen.broadcast(2, vec![4]).unwrap();
        let err = gen.dot("t_0", "t_1");
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("shape mismatch"));
    }

    #[test]
    fn test_codegen_matmul() {
        let mut gen = TensorCodegen::new();
        gen.variables.insert("a".to_string(), TensorType::matrix(3, 4));
        gen.variables.insert("b".to_string(), TensorType::matrix(4, 5));
        let result = gen.matmul("a", "b").unwrap();
        let code = gen.finish();
        assert!(code.contains("matmul(a, b)"));
        assert!(code.contains("tensor<3, 5"));
        // Verify the result type is 3x5
        assert_eq!(gen.get_type(&result).unwrap().shape, vec![3, 5]);
    }

    #[test]
    fn test_codegen_matmul_dimension_mismatch() {
        let mut gen = TensorCodegen::new();
        gen.variables.insert("a".to_string(), TensorType::matrix(3, 4));
        gen.variables.insert("b".to_string(), TensorType::matrix(5, 6));
        let err = gen.matmul("a", "b");
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("inner dimensions mismatch"));
    }

    #[test]
    fn test_codegen_transpose() {
        let mut gen = TensorCodegen::new();
        gen.variables.insert("a".to_string(), TensorType::matrix(3, 4));
        let result = gen.transpose("a").unwrap();
        let code = gen.finish();
        assert!(code.contains("transpose(a)"));
        assert!(code.contains("tensor<4, 3"));
        assert_eq!(gen.get_type(&result).unwrap().shape, vec![4, 3]);
    }

    #[test]
    fn test_codegen_slice_row() {
        let mut gen = TensorCodegen::new();
        gen.variables.insert("matrix".to_string(), TensorType::matrix(5, 3));
        let result = gen.slice_row("matrix", 2).unwrap();
        let code = gen.finish();
        assert!(code.contains("slice_row(matrix, 2)"));
        assert!(code.contains("tensor<3, i64>"));
        assert_eq!(gen.get_type(&result).unwrap().shape, vec![3]);
    }

    #[test]
    fn test_codegen_slice_row_out_of_bounds() {
        let mut gen = TensorCodegen::new();
        gen.variables.insert("matrix".to_string(), TensorType::matrix(3, 4));
        let err = gen.slice_row("matrix", 5);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("out of bounds"));
    }

    #[test]
    fn test_codegen_sequence() {
        let mut gen = TensorCodegen::new();
        let b1 = gen.broadcast(1, vec![3]).unwrap();
        let b2 = gen.broadcast(2, vec![3]).unwrap();
        let dot = gen.dot(&b1, &b2).unwrap();
        let code = gen.finish();

        assert!(code.contains("let t_0 = broadcast(1)"));
        assert!(code.contains("let t_1 = broadcast(2)"));
        assert!(code.contains("let t_2 = dot(t_0, t_1)"));

        // Verify the dot result is a scalar (shape empty)
        let empty: Vec<usize> = vec![];
        assert_eq!(gen.get_type(&dot).unwrap().shape, empty);
    }

    #[test]
    fn test_standalone_codegen_functions() {
        let bc = codegen_broadcast(5, vec![2, 3], "out");
        assert!(bc.contains("broadcast(5)"));
        assert!(bc.contains("tensor<2, 3, i64>"));

        let dot = codegen_dot("a", "b", "result");
        assert!(dot.contains("dot(a, b)"));

        let mm = codegen_matmul("x", "y", 3, 4, "z");
        assert!(mm.contains("matmul(x, y)"));
        assert!(mm.contains("tensor<3, 4>"));

        let tr = codegen_transpose("m", 5, 6, "m_t");
        assert!(tr.contains("transpose(m)"));
        assert!(tr.contains("tensor<5, 6>"));

        let sr = codegen_slice_row("mat", 1, 7, "row");
        assert!(sr.contains("slice_row(mat, 1)"));
    }
}
