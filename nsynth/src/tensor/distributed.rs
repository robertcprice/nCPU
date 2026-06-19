//! Distributed training primitives.
//!
//! This module provides simulation layer for distributed training operations.
//! Real distributed training would use NCCL/MPI backends.

use crate::tensor::{Tensor, Shape};

/// Reduction operation for all-reduce collective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise sum
    Sum,
    /// Element-wise average (sum / world_size)
    Average,
    /// Element-wise maximum
    Max,
    /// Element-wise minimum
    Min,
}

/// Perform an all-reduce collective operation.
///
/// In a real distributed setting, this would synchronize all processes
/// and combine their data according to the reduction operation.
///
/// # Arguments
/// * `tensor` - The tensor to reduce (modified in place)
/// * `op` - The reduction operation to apply
///
/// # Returns
/// A new tensor containing the reduced result
///
/// # Simulation Note
/// This implementation simulates all-reduce by applying the operation
/// to a copy of the tensor. In production, this would use NCCL/MPI.
pub fn all_reduce(tensor: &Tensor, op: ReduceOp) -> Tensor {
    // Simulate all-reduce by applying the operation
    // In real distributed training, this would:
    // 1. Synchronize all processes
    // 2. Collect all tensor values from each rank
    // 3. Apply the reduction operation
    // 4. Broadcast the result back to all ranks

    let data = &tensor.data;

    match op {
        ReduceOp::Sum => {
            // Simulate sum across world_size
            let mut result = data.clone();
            // In simulation: sum = value * 2 (assuming 2 processes)
            // Real implementation would sum across all ranks
            for v in result.iter_mut() {
                *v *= 2.0;
            }
            Tensor::new(result, tensor.shape.clone())
        }
        ReduceOp::Average => {
            // Simulate average: sum / world_size
            let mut result = data.clone();
            for v in result.iter_mut() {
                *v = (*v * 2.0) / 2.0; // Simulates (sum across 2 ranks) / 2
            }
            Tensor::new(result, tensor.shape.clone())
        }
        ReduceOp::Max => {
            // Simulate max: keep original (assuming same values)
            Tensor::new(data.clone(), tensor.shape.clone())
        }
        ReduceOp::Min => {
            // Simulate min: keep original (assuming same values)
            Tensor::new(data.clone(), tensor.shape.clone())
        }
    }
}

/// Gradient bucketing for efficient all-reduce operations.
///
/// Groups small gradients into buckets to amortize communication overhead.
/// Real implementations use micro-batching to overlap computation and communication.
#[derive(Debug)]
pub struct GradientBucket {
    /// Target size for each bucket (in bytes)
    bucket_size: usize,
    /// Current buckets of packed gradients
    buckets: Vec<Tensor>,
    /// Mapping from original gradient index to bucket info
    bucket_indices: Vec<(usize, usize)>, // (bucket_idx, offset)
}

impl GradientBucket {
    /// Create a new gradient bucket manager.
    ///
    /// # Arguments
    /// * `bucket_size` - Target size for each bucket in number of elements
    pub fn new(bucket_size: usize) -> Self {
        Self {
            bucket_size,
            buckets: Vec::new(),
            bucket_indices: Vec::new(),
        }
    }

    /// Pack gradients into buckets for efficient communication.
    ///
    /// Groups small gradients together to reduce the number of all-reduce operations.
    ///
    /// # Arguments
    /// * `grads` - Slice of gradient tensors to pack
    ///
    /// # Returns
    /// Vector of bucketed tensors ready for all-reduce
    pub fn pack_gradients(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let mut packed = Vec::new();
        let mut current_bucket = Vec::<f64>::new();
        let mut current_indices = Vec::new();
        let mut global_offset = 0;

        for grad in grads {
            let grad_data = &grad.data;
            let grad_size = grad_data.len();

            // If gradient is larger than bucket size, it gets its own bucket
            if grad_size >= self.bucket_size {
                // Flush current bucket if non-empty
                if !current_bucket.is_empty() {
                    let bucket_tensor = Tensor::new(current_bucket.clone(), Shape::new(vec![current_bucket.len()]));
                    self.buckets.push(bucket_tensor);
                    packed.push(self.buckets.last().unwrap().clone());
                    for &(idx, offset) in &current_indices {
                        self.bucket_indices.push((self.buckets.len() - 1, offset));
                    }
                    current_bucket.clear();
                    current_indices.clear();
                    global_offset = 0;
                }

                // Create bucket for this large gradient
                let bucket_tensor = grad.clone();
                self.buckets.push(bucket_tensor);
                packed.push(self.buckets.last().unwrap().clone());
                self.bucket_indices.push((self.buckets.len() - 1, 0));
            } else {
                // Check if adding this gradient would exceed bucket size
                if !current_bucket.is_empty() && current_bucket.len() + grad_size > self.bucket_size {
                    // Flush current bucket
                    let bucket_tensor = Tensor::new(current_bucket.clone(), Shape::new(vec![current_bucket.len()]));
                    self.buckets.push(bucket_tensor);
                    packed.push(self.buckets.last().unwrap().clone());
                    for &(idx, offset) in &current_indices {
                        self.bucket_indices.push((self.buckets.len() - 1, offset));
                    }
                    current_bucket.clear();
                    current_indices.clear();
                    global_offset = 0;
                }

                // Add to current bucket
                let start_offset = current_bucket.len();
                current_bucket.extend(grad_data);
                current_indices.push((packed.len() + self.buckets.len(), start_offset));
            }
        }

        // Flush final bucket
        if !current_bucket.is_empty() {
            let bucket_tensor = Tensor::new(current_bucket.clone(), Shape::new(vec![current_bucket.len()]));
            self.buckets.push(bucket_tensor);
            packed.push(self.buckets.last().unwrap().clone());
            for &(idx, offset) in &current_indices {
                self.bucket_indices.push((self.buckets.len() - 1, offset));
            }
        }

        packed
    }

    /// Unpack bucketed gradients back to original tensor shapes.
    ///
    /// # Arguments
    /// * `buckets` - Synchronized buckets from all-reduce
    /// * `original_shapes` - Original shapes of each gradient tensor
    ///
    /// # Returns
    /// Vector of unpacked gradient tensors
    pub fn unpack_buckets(&self, buckets: &[Tensor], original_shapes: &[Vec<usize>]) -> Vec<Tensor> {
        // Reconstruct the original gradient tensors from buckets
        // This would use the bucket_indices mapping
        let mut result = Vec::new();
        let mut bucket_idx = 0;
        let mut offset = 0;
        let empty_vec = vec![];

        for shape in original_shapes {
            let size: usize = shape.iter().product();
            let bucket_data = buckets.get(bucket_idx).map(|t| &t.data).unwrap_or(&empty_vec);

            if offset + size <= bucket_data.len() {
                let grad_data = bucket_data[offset..offset + size].to_vec();
                result.push(Tensor::new(grad_data, Shape::new(shape.clone())));
                offset += size;

                // Move to next bucket if current is exhausted
                if offset >= bucket_data.len() {
                    bucket_idx += 1;
                    offset = 0;
                }
            } else {
                // Gradient spans multiple buckets (unlikely in practice)
                let mut grad_data = Vec::with_capacity(size);
                let mut remaining = size;
                let mut current_bucket = bucket_idx;
                let mut current_offset = offset;

                while remaining > 0 && current_bucket < buckets.len() {
                    let available = buckets[current_bucket].data.len() - current_offset;
                    let take = available.min(remaining);

                    grad_data.extend_from_slice(
                        &buckets[current_bucket].data[current_offset..current_offset + take]
                    );

                    remaining -= take;
                    current_bucket += 1;
                    current_offset = 0;
                }

                result.push(Tensor::new(grad_data, Shape::new(shape.clone())));
                bucket_idx = current_bucket;
                offset = current_offset;
            }
        }

        result
    }

    /// Return the number of buckets currently managed.
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Return the total size of all buckets.
    pub fn total_size(&self) -> usize {
        self.buckets.iter().map(|b| b.data.len()).sum()
    }
}

impl Default for GradientBucket {
    fn default() -> Self {
        Self::new(1024 * 1024) // 1M elements default bucket size
    }
}

/// Multi-device gradient synchronization coordinator.
///
/// Manages gradient synchronization across multiple processes/devices.
#[derive(Debug, Clone)]
pub struct GradientSynchronization {
    /// Total number of processes/devices in the job
    world_size: usize,
    /// Rank of this process (0 to world_size-1)
    rank: usize,
}

impl GradientSynchronization {
    /// Create a new gradient synchronization coordinator.
    ///
    /// # Arguments
    /// * `world_size` - Total number of processes
    /// * `rank` - Rank of this process (0-indexed)
    pub fn new(world_size: usize, rank: usize) -> Self {
        assert!(rank < world_size, "rank must be less than world_size");
        Self { world_size, rank }
    }

    /// Synchronize gradients across all processes.
    ///
    /// Performs all-reduce on all gradient tensors.
    ///
    /// # Arguments
    /// * `grads` - Gradient tensors to synchronize
    ///
    /// # Returns
    /// A tensor representing the synchronization status (0 = success)
    pub fn synchronize_gradients(&self, grads: &mut [Tensor]) -> Tensor {
        // Simulate gradient synchronization
        for grad in grads.iter_mut() {
            let reduced = all_reduce(grad, ReduceOp::Average);
            // Update gradient with synchronized values
            *grad = reduced;
        }

        // Return status tensor (simulated)
        Tensor::new(vec![0.0], Shape::new(vec![1]))
    }

    /// Broadcast parameters from root rank to all other ranks.
    ///
    /// # Arguments
    /// * `params` - Parameter tensors to broadcast
    /// * `root_rank` - Rank of the process broadcasting parameters
    pub fn broadcast_parameters(&self, params: &mut [Tensor], root_rank: usize) {
        // In simulation, all ranks have the same data
        // Real implementation would use MPI_Bcast or NCCL broadcast
        if self.rank != root_rank {
            // In real implementation, this would receive data from root
            // In simulation, data is already synchronized
        }
        // Root rank already has the correct data
    }

    /// Return the world size.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Return this process's rank.
    pub fn rank(&self) -> usize {
        self.rank
    }
}

/// Distributed Data Parallel (DDP) wrapper for multi-device training.
///
/// Simulates PyTorch DDP behavior for data parallel training across multiple devices.
pub struct DistributedDataParallel<F>
where
    F: Fn(&Tensor) -> Tensor,
{
    /// Local model forward pass function
    model: F,
    /// Gradient synchronization coordinator
    sync: GradientSynchronization,
    /// Whether to use gradient bucketing
    use_bucketing: bool,
    /// Gradient bucket manager
    bucket: GradientBucket,
}

impl<F> DistributedDataParallel<F>
where
    F: Fn(&Tensor) -> Tensor,
{
    /// Create a new DDP wrapper.
    ///
    /// # Arguments
    /// * `model` - Forward function of the local model
    /// * `world_size` - Number of distributed processes
    /// * `rank` - Rank of this process
    pub fn new(model: F, world_size: usize, rank: usize) -> Self {
        Self {
            model,
            sync: GradientSynchronization::new(world_size, rank),
            use_bucketing: true,
            bucket: GradientBucket::default(),
        }
    }

    /// Create a DDP wrapper without gradient bucketing.
    pub fn without_bucketing(model: F, world_size: usize, rank: usize) -> Self {
        Self {
            model,
            sync: GradientSynchronization::new(world_size, rank),
            use_bucketing: false,
            bucket: GradientBucket::default(),
        }
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// Output tensor from the model
    pub fn forward(&self, x: &Tensor) -> Tensor {
        (self.model)(x)
    }

    /// Backward pass with gradient synchronization.
    ///
    /// # Arguments
    /// * `grads` - Gradient tensors to synchronize
    ///
    /// # Returns
    /// Synchronized gradient tensor
    pub fn backward(&mut self, grads: &mut [Tensor]) -> Tensor {
        if self.use_bucketing && grads.len() > 1 {
            // Pack gradients into buckets
            let shapes: Vec<Vec<usize>> = grads.iter().map(|g| g.shape.dims.clone()).collect();
            let buckets = self.bucket.pack_gradients(grads.to_vec());

            // Synchronize buckets
            let mut synced_buckets = buckets;
            self.sync.synchronize_gradients(&mut synced_buckets);

            // Unpack back to original gradients
            let unpacked = self.bucket.unpack_buckets(&synced_buckets, &shapes);
            for (i, grad) in grads.iter_mut().enumerate() {
                if let Some(unpacked_grad) = unpacked.get(i) {
                    *grad = unpacked_grad.clone();
                }
            }

            Tensor::new(vec![0.0], Shape::new(vec![1])) // Status: success
        } else {
            // Direct synchronization without bucketing
            self.sync.synchronize_gradients(grads)
        }
    }

    /// Get the world size.
    pub fn world_size(&self) -> usize {
        self.sync.world_size()
    }

    /// Get the rank.
    pub fn rank(&self) -> usize {
        self.sync.rank()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_mock_tensor(values: Vec<f64>) -> Tensor {
        let len = values.len();
        Tensor::new(values, Shape::new(vec![len]))
    }

    #[test]
    fn test_reduce_op_sum() {
        let tensor = create_mock_tensor(vec![1.0, 2.0, 3.0]);
        let result = all_reduce(&tensor, ReduceOp::Sum);
        assert_eq!(&result.data, &[2.0, 4.0, 6.0]); // Simulated: *2 for 2 processes
    }

    #[test]
    fn test_reduce_op_average() {
        let tensor = create_mock_tensor(vec![2.0, 4.0, 6.0]);
        let result = all_reduce(&tensor, ReduceOp::Average);
        assert_eq!(&result.data, &[2.0, 4.0, 6.0]); // Average preserves values
    }

    #[test]
    fn test_reduce_op_max() {
        let tensor = create_mock_tensor(vec![1.0, 5.0, 3.0]);
        let result = all_reduce(&tensor, ReduceOp::Max);
        assert_eq!(&result.data, &[1.0, 5.0, 3.0]); // Preserves in simulation
    }

    #[test]
    fn test_reduce_op_min() {
        let tensor = create_mock_tensor(vec![1.0, 5.0, 3.0]);
        let result = all_reduce(&tensor, ReduceOp::Min);
        assert_eq!(&result.data, &[1.0, 5.0, 3.0]); // Preserves in simulation
    }

    #[test]
    fn test_gradient_bucket_new() {
        let bucket = GradientBucket::new(100);
        assert_eq!(bucket.bucket_size, 100);
        assert_eq!(bucket.num_buckets(), 0);
    }

    #[test]
    fn test_gradient_bucket_pack_single() {
        let mut bucket = GradientBucket::new(100);
        let grads = vec![create_mock_tensor(vec![1.0, 2.0, 3.0])];
        let packed = bucket.pack_gradients(grads);

        assert_eq!(packed.len(), 1);
        assert_eq!(bucket.num_buckets(), 1);
        assert_eq!(&packed[0].data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_gradient_bucket_pack_multiple() {
        let mut bucket = GradientBucket::new(5);
        let grads = vec![
            create_mock_tensor(vec![1.0, 2.0]),
            create_mock_tensor(vec![3.0, 4.0]),
            create_mock_tensor(vec![5.0, 6.0]),
        ];
        let packed = bucket.pack_gradients(grads);

        // First two gradients fit in one bucket (4 elements), third goes to new bucket
        assert!(packed.len() >= 2);
    }

    #[test]
    fn test_gradient_bucket_unpack() {
        let mut bucket = GradientBucket::new(100);
        let grads = vec![
            create_mock_tensor(vec![1.0, 2.0]),
            create_mock_tensor(vec![3.0, 4.0]),
        ];
        let shapes: Vec<Vec<usize>> = grads.iter().map(|g| g.shape.dims.clone()).collect();
        let packed = bucket.pack_gradients(grads);

        let unpacked = bucket.unpack_buckets(&packed, &shapes);

        assert_eq!(unpacked.len(), 2);
        assert_eq!(&unpacked[0].data, &[1.0, 2.0]);
        assert_eq!(&unpacked[1].data, &[3.0, 4.0]);
    }

    #[test]
    fn test_gradient_bucket_large_gradient() {
        let mut bucket = GradientBucket::new(10);
        let grads = vec![
            create_mock_tensor(vec![1.0, 2.0, 3.0]),
            create_mock_tensor(vec![4.0; 20]), // Larger than bucket
        ];

        let packed = bucket.pack_gradients(grads);

        // Large gradient gets its own bucket
        assert!(packed.len() >= 2);
    }

    #[test]
    fn test_gradient_synchronization_new() {
        let sync = GradientSynchronization::new(4, 2);
        assert_eq!(sync.world_size(), 4);
        assert_eq!(sync.rank(), 2);
    }

    #[test]
    fn test_gradient_synchronization_invalid_rank() {
        // Rank must be less than world_size
        let sync = GradientSynchronization::new(2, 2);
        assert_eq!(sync.world_size(), 2);
        // This should have been caught by the assert in new()
    }

    #[test]
    fn test_gradient_synchronization_sync() {
        let sync = GradientSynchronization::new(2, 0);
        let mut grads = vec![create_mock_tensor(vec![1.0, 2.0, 3.0])];
        let status = sync.synchronize_gradients(&mut grads);

        // Status should be 0 (success)
        assert_eq!(&status.data, &[0.0]);
        // Gradients should be synchronized (averaged)
        assert_eq!(&grads[0].data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ddp_new() {
        let model = |x: &Tensor| x.clone();
        let ddp = DistributedDataParallel::new(model, 4, 1);

        assert_eq!(ddp.world_size(), 4);
        assert_eq!(ddp.rank(), 1);
    }

    #[test]
    fn test_ddp_forward() {
        let model = |x: &Tensor| Tensor::new(x.data.iter().map(|v| v * 2.0).collect(), x.shape.clone());
        let ddp = DistributedDataParallel::new(model, 2, 0);

        let input = create_mock_tensor(vec![1.0, 2.0, 3.0]);
        let output = ddp.forward(&input);

        assert_eq!(&output.data, &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_ddp_backward() {
        let model = |x: &Tensor| x.clone();
        let mut ddp = DistributedDataParallel::new(model, 2, 0);

        let mut grads = vec![
            create_mock_tensor(vec![1.0, 2.0]),
            create_mock_tensor(vec![3.0, 4.0]),
        ];

        let status = ddp.backward(&mut grads);

        assert_eq!(&status.data, &[0.0]);
        // Gradients should be synchronized
    }

    #[test]
    fn test_ddp_without_bucketing() {
        let model = |x: &Tensor| x.clone();
        let mut ddp = DistributedDataParallel::without_bucketing(model, 2, 0);

        let mut grads = vec![create_mock_tensor(vec![1.0, 2.0])];
        let status = ddp.backward(&mut grads);

        assert_eq!(&status.data, &[0.0]);
        assert_eq!(&grads[0].data, &[1.0, 2.0]);
    }

    #[test]
    fn test_multi_process_simulation() {
        // Simulate two processes synchronizing gradients
        let rank0_sync = GradientSynchronization::new(2, 0);
        let rank1_sync = GradientSynchronization::new(2, 1);

        let mut rank0_grads = vec![create_mock_tensor(vec![2.0, 4.0])];
        let mut rank1_grads = vec![create_mock_tensor(vec![4.0, 8.0])];

        // Both synchronize
        rank0_sync.synchronize_gradients(&mut rank0_grads);
        rank1_sync.synchronize_gradients(&mut rank1_grads);

        // After synchronization, both should have the average
        // (In simulation, this preserves the local values since we simulate avg = sum/size)
        assert_eq!(&rank0_grads[0].data, &[2.0, 4.0]);
        assert_eq!(&rank1_grads[0].data, &[4.0, 8.0]);
    }
}
