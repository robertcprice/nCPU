//! Data Loading for nCPU/nSynth
//!
//! DataSet, DataLoader, batching utilities.

use super::ops::{Shape, Tensor};

// ============================================================================
// DATASET TRAIT
// ============================================================================

/// Dataset trait for data sources
pub trait Dataset {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<(Tensor, Tensor)>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Simple in-memory dataset
#[derive(Debug, Clone)]
pub struct SimpleDataset {
    pub data: Vec<(Tensor, Tensor)>,
}

impl SimpleDataset {
    pub fn new(data: Vec<(Tensor, Tensor)>) -> Self {
        Self { data }
    }
}

impl Dataset for SimpleDataset {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Option<(Tensor, Tensor)> {
        self.data.get(index).cloned()
    }
}

// ============================================================================
// ITERABLE DATASET
// ============================================================================

/// Iterable dataset for streaming data
pub trait IterableDataset {
    fn iter(&self) -> Box<dyn Iterator<Item = (Tensor, Tensor)> + '_>;
}

/// Streaming dataset from iterator
#[derive(Debug)]
pub struct StreamingDataset<F, I>
where
    F: Fn() -> I,
    I: Iterator<Item = (Tensor, Tensor)>,
{
    pub source: F,
}

impl<F, I> StreamingDataset<F, I>
where
    F: Fn() -> I,
    I: Iterator<Item = (Tensor, Tensor)>,
{
    pub fn new(source: F) -> Self {
        Self { source }
    }
}

impl<F, I> IterableDataset for StreamingDataset<F, I>
where
    F: Fn() -> I,
    I: Iterator<Item = (Tensor, Tensor)>,
{
    fn iter(&self) -> Box<dyn Iterator<Item = (Tensor, Tensor)> + '_> {
        Box::new((self.source)())
    }
}

// ============================================================================
// DATALOADER
// ============================================================================

/// DataLoader for batching and shuffling
#[derive(Debug)]
pub struct DataLoader<D: Dataset> {
    pub dataset: D,
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
    pub epoch: usize,
    pub indices: Vec<usize>,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        let len = dataset.len();
        let indices = (0..len).collect();

        Self {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
            epoch: 0,
            indices,
        }
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn reset(&mut self) {
        self.epoch += 1;
        self.indices = (0..self.dataset.len()).collect();
        if self.shuffle {
            // Simple Fisher-Yates shuffle
            let n = self.indices.len();
            for i in (1..n).rev() {
                let j = (self.epoch * i + i) % (i + 1); // Pseudo-random based on epoch
                self.indices.swap(i, j);
            }
        }
    }
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = Vec<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.epoch == 0 || self.indices.is_empty() {
            self.reset();
        }

        if self.indices.is_empty() {
            return None;
        }

        let batch_indices: Vec<usize> = self
            .indices
            .drain(0..self.batch_size.min(self.indices.len()))
            .collect();

        if self.drop_last && batch_indices.len() < self.batch_size {
            return None;
        }

        let batch: Vec<(Tensor, Tensor)> = batch_indices
            .iter()
            .filter_map(|&i| self.dataset.get(i))
            .collect();

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
}

// ============================================================================
// COLLATE FUNCTIONS
// ============================================================================

/// Default collate: stack tensors
pub fn default_collate(batch: &[(Tensor, Tensor)]) -> Option<(Tensor, Tensor)> {
    if batch.is_empty() {
        return None;
    }

    let inputs: Vec<&Tensor> = batch.iter().map(|(x, _)| x).collect();
    let targets: Vec<&Tensor> = batch.iter().map(|(_, y)| y).collect();

    let stacked_inputs = stack_tensors(&inputs);
    let stacked_targets = stack_tensors(&targets);

    Some((stacked_inputs, stacked_targets))
}

/// Stack tensors along new dimension
pub fn stack_tensors(tensors: &[&Tensor]) -> Tensor {
    if tensors.is_empty() {
        return Tensor::scalar(0.0);
    }

    if tensors.len() == 1 {
        return tensors[0].clone();
    }

    let mut all_data = Vec::new();
    let shape = &tensors[0].shape;

    for tensor in tensors {
        all_data.extend(&tensor.data);
    }

    let mut new_shape = vec![tensors.len()];
    new_shape.extend(shape.dims.clone());

    Tensor::new(all_data, Shape::new(new_shape))
}

/// Collate with padding for variable length sequences
pub fn pad_collate(batch: &[(Tensor, Tensor)], pad_value: f64) -> Option<(Tensor, Tensor)> {
    if batch.is_empty() {
        return None;
    }

    let max_len = batch.iter().map(|(x, _)| x.data.len()).max().unwrap_or(0);
    let feature_dim = batch[0].0.shape.dims.get(1).copied().unwrap_or(1);

    let mut padded_inputs = Vec::new();
    let mut targets = Vec::new();

    for (x, y) in batch {
        let mut padded = vec![pad_value; max_len * feature_dim];
        for (i, &v) in x.data.iter().enumerate() {
            padded[i] = v;
        }
        padded_inputs.push(Tensor::new(
            padded.clone(),
            Shape::new(vec![max_len, feature_dim]),
        ));
        targets.push(y.clone());
    }

    let inputs_refs: Vec<&Tensor> = padded_inputs.iter().collect();
    let targets_refs: Vec<&Tensor> = targets.iter().collect();
    let stacked_inputs = stack_tensors(&inputs_refs);
    let stacked_targets = stack_tensors(&targets_refs);

    Some((stacked_inputs, stacked_targets))
}

// ============================================================================
// BATCH SAMPLING
// ============================================================================

/// Batch sampler for custom batching strategies
#[derive(Debug)]
pub struct BatchSampler {
    pub indices: Vec<usize>,
    pub batch_size: usize,
    pub drop_last: bool,
    pub shuffle: bool,
}

impl BatchSampler {
    pub fn new(dataset_size: usize, batch_size: usize) -> Self {
        let indices = (0..dataset_size).collect();
        Self {
            indices,
            batch_size,
            drop_last: false,
            shuffle: false,
        }
    }

    pub fn with_shuffle(mut self) -> Self {
        self.shuffle = true;
        self
    }

    pub fn with_drop_last(mut self) -> Self {
        self.drop_last = true;
        self
    }

    pub fn generate_batches(&mut self) -> Vec<Vec<usize>> {
        if self.shuffle {}

        let mut batches = Vec::new();
        let mut i = 0;

        while i < self.indices.len() {
            let end = (i + self.batch_size).min(self.indices.len());
            let batch: Vec<usize> = self.indices[i..end].to_vec();

            if !self.drop_last || batch.len() == self.batch_size {
                batches.push(batch);
            }

            i = end;
        }

        batches
    }
}

// ============================================================================
// DISTRIBUTED SAMPLING (SIMULATED)
// ============================================================================

/// Distributed sampler for multi-GPU training
#[derive(Debug)]
pub struct DistributedSampler {
    pub num_replicas: usize,
    pub rank: usize,
    pub dataset_size: usize,
    pub shuffle: bool,
    pub epoch: usize,
}

impl DistributedSampler {
    pub fn new(dataset_size: usize, num_replicas: usize, rank: usize) -> Self {
        Self {
            num_replicas,
            rank,
            dataset_size,
            shuffle: false,
            epoch: 0,
        }
    }

    pub fn with_shuffle(mut self) -> Self {
        self.shuffle = true;
        self
    }

    pub fn indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.dataset_size).collect();

        if self.shuffle {
            // Use epoch as seed for reproducibility - simple Fisher-Yates
            let n = indices.len();
            let mut seed = self.epoch as u64;
            for i in (1..n).rev() {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let j = (seed % (i as u64 + 1)) as usize;
                indices.swap(i, j);
            }
        }

        // Distribute indices across replicas
        indices
            .into_iter()
            .skip(self.rank)
            .step_by(self.num_replicas)
            .collect()
    }

    pub fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
    }
}

// ============================================================================
// WEIGHTED RANDOM SAMPLING
// ============================================================================

/// Weighted random sampler for imbalanced datasets
#[derive(Debug)]
pub struct WeightedRandomSampler {
    pub weights: Vec<f64>,
    pub num_samples: usize,
    pub replacement: bool,
}

impl WeightedRandomSampler {
    pub fn new(weights: Vec<f64>, num_samples: usize, replacement: bool) -> Self {
        Self {
            weights,
            num_samples,
            replacement,
        }
    }

    pub fn sample(&self) -> Vec<usize> {
        let total: f64 = self.weights.iter().sum();
        let normalized: Vec<f64> = self.weights.iter().map(|&w| w / total).collect();

        let mut indices = Vec::new();
        let mut seed = 12345_u64;

        for _ in 0..self.num_samples {
            // Simple deterministic PRNG
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let rand_val = (seed % 10000) as f64 / 10000.0;
            let mut cumsum = 0.0;

            for (i, &p) in normalized.iter().enumerate() {
                cumsum += p;
                if rand_val <= cumsum {
                    indices.push(i);
                    break;
                }
            }
        }

        indices
    }
}

// ============================================================================
// DATA TRANSFORMS
// ============================================================================

/// Transformation pipeline
pub struct TransformPipeline {
    pub transforms: Vec<Box<dyn Fn(&Tensor) -> Tensor>>,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    pub fn add(mut self, transform: Box<dyn Fn(&Tensor) -> Tensor>) -> Self {
        self.transforms.push(transform);
        self
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();
        for transform in &self.transforms {
            out = transform(&out);
        }
        out
    }
}

/// Normalization transform
pub fn normalize(mean: f64, std: f64) -> Box<dyn Fn(&Tensor) -> Tensor> {
    Box::new(move |x: &Tensor| {
        let normalized = x.sub(&mean.into()).unwrap();
        normalized.div(&std.into()).unwrap()
    })
}

/// Standardization (per-dimension)
pub fn standardize(means: &[f64], stds: &[f64]) -> Box<dyn Fn(&Tensor) -> Tensor> {
    let means = means.to_vec();
    let stds = stds.to_vec();

    Box::new(move |x: &Tensor| {
        let mut normalized = x.clone();
        for (i, v) in normalized.data.iter_mut().enumerate() {
            let idx = i % means.len();
            *v = (*v - means[idx]) / stds[idx];
        }
        normalized
    })
}

/// One-hot encoding
pub fn one_hot(num_classes: usize) -> Box<dyn Fn(&Tensor) -> Tensor> {
    Box::new(move |x: &Tensor| {
        let mut one_hot = vec![0.0; x.data.len() * num_classes];

        for (i, &v) in x.data.iter().enumerate() {
            let class = v as usize % num_classes;
            one_hot[i * num_classes + class] = 1.0;
        }

        Tensor::new(one_hot, Shape::new(vec![x.data.len(), num_classes]))
    })
}

// ============================================================================
// DATASET UTILITIES
// ============================================================================

/// Split dataset into train/val/test
pub fn split_dataset<D: Dataset + Clone>(
    dataset: &D,
    train_ratio: f64,
    val_ratio: f64,
) -> (
    Vec<(Tensor, Tensor)>,
    Vec<(Tensor, Tensor)>,
    Vec<(Tensor, Tensor)>,
) {
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();

    let total = dataset.len();
    let train_size = (total as f64 * train_ratio) as usize;
    let val_size = (total as f64 * val_ratio) as usize;

    for i in 0..total {
        if let Some(sample) = dataset.get(i) {
            if i < train_size {
                train.push(sample);
            } else if i < train_size + val_size {
                validation.push(sample);
            } else {
                test.push(sample);
            }
        }
    }

    (train, validation, test)
}

/// K-fold cross validation indices
pub fn k_fold_indices(n: usize, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let fold_size = n / k;
    let mut folds = Vec::new();

    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 { n } else { (i + 1) * fold_size };

        let test_indices: Vec<usize> = (start..end).collect();
        let train_indices: Vec<usize> = (0..start).chain(end..n).collect();

        folds.push((train_indices, test_indices));
    }

    folds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_dataset() {
        let data = vec![
            (Tensor::vector(vec![1.0]), Tensor::scalar(0.0)),
            (Tensor::vector(vec![2.0]), Tensor::scalar(1.0)),
        ];
        let dataset = SimpleDataset::new(data);
        assert_eq!(dataset.len(), 2);
        assert!(dataset.get(0).is_some());
    }

    #[test]
    fn test_dataloader() {
        let data = vec![
            (Tensor::vector(vec![1.0]), Tensor::scalar(0.0)),
            (Tensor::vector(vec![2.0]), Tensor::scalar(1.0)),
            (Tensor::vector(vec![3.0]), Tensor::scalar(0.0)),
            (Tensor::vector(vec![4.0]), Tensor::scalar(1.0)),
        ];
        let dataset = SimpleDataset::new(data);
        let mut loader = DataLoader::new(dataset, 2);

        let batch1 = loader.next();
        assert!(batch1.is_some());
        assert_eq!(batch1.unwrap().len(), 2);
    }

    #[ignore = "experimental tensor stack — Package O; see docs/TENSOR_QUARANTINE.md"]
    #[test]
    fn test_pad_collate() {
        let batch = vec![
            (Tensor::vector(vec![1.0, 2.0]), Tensor::scalar(0.0)),
            (Tensor::vector(vec![3.0]), Tensor::scalar(1.0)),
        ];
        let collated = pad_collate(&batch, 0.0);
        assert!(collated.is_some());
        let (inputs, _) = collated.unwrap();
        assert_eq!(inputs.shape, Shape::new(vec![2, 2]));
    }

    #[test]
    fn test_transform_pipeline() {
        let pipeline = TransformPipeline::new().add(Box::new(|x| x.mul(&2.0.into()).unwrap()));

        let x = Tensor::vector(vec![1.0, 2.0]);
        let transformed = pipeline.apply(&x);
        assert_eq!(transformed.data, vec![2.0, 4.0]);
    }
}
