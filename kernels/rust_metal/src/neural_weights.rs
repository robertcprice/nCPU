// neural_weights.rs - Neural model weight management for GPU buffers
//
// This module handles loading and managing neural network weights
// that are passed from Python (loaded via PyTorch) into GPU buffers.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use pyo3::prelude::*;

use crate::MetalError;

/// Weight container for passing flattened weights from Python
#[pyclass]
#[derive(Clone)]
pub struct ModelWeights {
    #[pyo3(get, set)]
    pub weights: Vec<f32>,

    #[pyo3(get, set)]
    pub shape: Vec<usize>,
}

#[pymethods]
impl ModelWeights {
    #[new]
    fn new(weights: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { weights, shape }
    }

    fn size(&self) -> usize {
        self.weights.len()
    }

    fn total_params(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Creates a GPU buffer from Python-provided weights
pub fn create_buffer_from_weights(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    weights: &ModelWeights,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let byte_size = weights.weights.len() * std::mem::size_of::<f32>();

    let options = MTLResourceOptions::StorageModeShared;
    let buffer = device
        .newBufferWithLength_options(byte_size, options)
        .ok_or(MetalError::BufferCreationFailed)?;

    // Copy weights to GPU buffer
    unsafe {
        let ptr = buffer.contents().as_ptr() as *mut f32;
        for (i, &weight) in weights.weights.iter().enumerate() {
            *ptr.add(i) = weight;
        }
    }

    Ok(buffer)
}

/// Weight collection for all neural models
#[pyclass]
pub struct NeuralWeightCollection {
    pub dispatch_weights: Option<ModelWeights>,
    pub embedding_weights: Option<ModelWeights>, // 100% accurate dispatch (10,279 params)
    pub loop_detector_weights: Option<ModelWeights>,
    pub memory_oracle_weights: Option<ModelWeights>,
    pub pattern_recognizer_weights: Option<ModelWeights>,
}

#[pymethods]
impl NeuralWeightCollection {
    #[new]
    fn new() -> Self {
        Self {
            dispatch_weights: None,
            embedding_weights: None,
            loop_detector_weights: None,
            memory_oracle_weights: None,
            pattern_recognizer_weights: None,
        }
    }

    fn set_dispatch_weights(&mut self, weights: ModelWeights) {
        self.dispatch_weights = Some(weights);
    }

    fn set_embedding_weights(&mut self, weights: ModelWeights) {
        self.embedding_weights = Some(weights);
    }

    fn set_loop_detector_weights(&mut self, weights: ModelWeights) {
        self.loop_detector_weights = Some(weights);
    }

    fn set_memory_oracle_weights(&mut self, weights: ModelWeights) {
        self.memory_oracle_weights = Some(weights);
    }

    fn set_pattern_recognizer_weights(&mut self, weights: ModelWeights) {
        self.pattern_recognizer_weights = Some(weights);
    }

    fn is_complete(&self) -> bool {
        self.dispatch_weights.is_some()
            && self.loop_detector_weights.is_some()
            && self.memory_oracle_weights.is_some()
    }

    fn total_params(&self) -> usize {
        let mut total = 0;
        if let Some(ref w) = self.dispatch_weights {
            total += w.total_params();
        }
        if let Some(ref w) = self.embedding_weights {
            total += w.total_params();
        }
        if let Some(ref w) = self.loop_detector_weights {
            total += w.total_params();
        }
        if let Some(ref w) = self.memory_oracle_weights {
            total += w.total_params();
        }
        if let Some(ref w) = self.pattern_recognizer_weights {
            total += w.total_params();
        }
        total
    }

    /// Save all weights to disk for persistence
    fn save_to_path(&self, path: &str) -> PyResult<usize> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to create file: {}", e))
        })?;

        let mut saved = 0;

        // Save dispatch weights
        if let Some(ref w) = self.dispatch_weights {
            let header = b"DISP";
            file.write_all(header).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let len = (w.weights.len() as u32).to_le_bytes();
            file.write_all(&len).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let bytes: Vec<u8> = w.weights.iter().flat_map(|f| f.to_le_bytes()).collect();
            file.write_all(&bytes).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            saved += 1;
        }

        // Save embedding weights
        if let Some(ref w) = self.embedding_weights {
            let header = b"EMBD";
            file.write_all(header).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let len = (w.weights.len() as u32).to_le_bytes();
            file.write_all(&len).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let bytes: Vec<u8> = w.weights.iter().flat_map(|f| f.to_le_bytes()).collect();
            file.write_all(&bytes).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            saved += 1;
        }

        // Save loop detector weights
        if let Some(ref w) = self.loop_detector_weights {
            let header = b"LOOP";
            file.write_all(header).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let len = (w.weights.len() as u32).to_le_bytes();
            file.write_all(&len).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let bytes: Vec<u8> = w.weights.iter().flat_map(|f| f.to_le_bytes()).collect();
            file.write_all(&bytes).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            saved += 1;
        }

        // Save memory oracle weights
        if let Some(ref w) = self.memory_oracle_weights {
            let header = b"MEMO";
            file.write_all(header).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let len = (w.weights.len() as u32).to_le_bytes();
            file.write_all(&len).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let bytes: Vec<u8> = w.weights.iter().flat_map(|f| f.to_le_bytes()).collect();
            file.write_all(&bytes).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            saved += 1;
        }

        // Save pattern recognizer weights
        if let Some(ref w) = self.pattern_recognizer_weights {
            let header = b"PATT";
            file.write_all(header).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let len = (w.weights.len() as u32).to_le_bytes();
            file.write_all(&len).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let bytes: Vec<u8> = w.weights.iter().flat_map(|f| f.to_le_bytes()).collect();
            file.write_all(&bytes).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            saved += 1;
        }

        Ok(saved)
    }

    /// Load all weights from disk
    fn load_from_path(&mut self, path: &str) -> PyResult<usize> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to open file: {}", e))
        })?;

        let mut loaded = 0;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to read file: {}", e))
        })?;

        let mut offset = 0;
        while offset + 8 <= buffer.len() {
            let header = &buffer[offset..offset + 4];
            let len = u32::from_le_bytes([buffer[offset + 4], buffer[offset + 5], buffer[offset + 6], buffer[offset + 7]]) as usize;

            if offset + 8 + len * 4 > buffer.len() {
                break;
            }

            let mut weights = Vec::with_capacity(len);
            for i in 0..len {
                let float_bytes = [
                    buffer[offset + 8 + i * 4],
                    buffer[offset + 9 + i * 4],
                    buffer[offset + 10 + i * 4],
                    buffer[offset + 11 + i * 4],
                ];
                weights.push(f32::from_le_bytes(float_bytes));
            }

            match header {
                b"DISP" => {
                    self.dispatch_weights = Some(ModelWeights::new(weights, vec![weights.len()]));
                    loaded += 1;
                }
                b"EMBD" => {
                    self.embedding_weights = Some(ModelWeights::new(weights, vec![weights.len()]));
                    loaded += 1;
                }
                b"LOOP" => {
                    self.loop_detector_weights = Some(ModelWeights::new(weights, vec![weights.len()]));
                    loaded += 1;
                }
                b"MEMO" => {
                    self.memory_oracle_weights = Some(ModelWeights::new(weights, vec![weights.len()]));
                    loaded += 1;
                }
                b"PATT" => {
                    self.pattern_recognizer_weights = Some(ModelWeights::new(weights, vec![weights.len()]));
                    loaded += 1;
                }
                _ => {}
            }

            offset += 8 + len * 4;
        }

        Ok(loaded)
    }
}

pub fn register_weights(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ModelWeights>()?;
    m.add_class::<NeuralWeightCollection>()?;
    Ok(())
}
