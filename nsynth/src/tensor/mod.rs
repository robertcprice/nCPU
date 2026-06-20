//! ML/Tensor Engine for nCPU/nSynth
//!
//! Comprehensive tensor operations, autodiff, models, training, layers.

pub mod activations;
pub mod advanced_ops;
pub mod autodiff;
pub mod layers;
pub mod losses;
pub mod model;
pub mod ops;
pub mod schedulers;
pub mod special_ops;
pub mod train;

// Advanced ML primitives
pub mod advanced_layers;
pub mod advanced_losses;
pub mod advanced_optimizers;
pub mod composition_primitives;
pub mod data_loading;
pub mod gnn_layers;
pub mod initializers;
pub mod metrics;
pub mod training_utils;

// Attention mechanisms
pub mod attention;
pub mod efficient_attention;

pub use autodiff::{ComputeGraph, Node, Op};
pub use model::{Activation, Conv2D, Linear, Model, MLP};
pub use ops::{DType, Shape, Tensor};
pub use train::{LossFn, Optimizer, OptimizerState, Trainer, TrainingConfig};

// Attention primitives
pub use attention::{
    flash_attention, ALiBi, LayerNorm as AttentionLayerNorm, MultiHeadAttention,
    PositionalEncoding, RoPE,
};

// Diffusion model primitives
pub mod diffusion;
pub use diffusion::{score_matching_loss, GaussianDiffusion, ScheduleType, UNet1D};

// Neural ODE and Energy-Based Models
pub mod neural_ode;
pub use neural_ode::{EBMLoss, EnergyModel, NeuralODE, ODESolver};

// Normalizing Flows for Density Estimation
pub mod flows;
pub use flows::{
    flow_loss, kl_divergence_loss, standard_normal_log_prob, CouplingMLP, MADEBlock, MaskedLinear,
    RealNVP, MAF,
};

// Fourier Features and NeRF primitives
pub mod fourier_nerf;
pub use fourier_nerf::{
    ray_sampling, volume_rendering, FourierFeatures, NeRF, RandomFourierFeatures,
};

// Special vision operations
pub use special_ops::{
    AvgPool3D, Conv3D, Conv3DTranspose, DeformableConv2D, DeformablePooling, MaxPool3D, PoolType,
};

// Reinforcement Learning buffers and actor-critic methods
pub mod rl_buffer;
pub use rl_buffer::{
    Experience, PrioritizedExperience, PrioritizedReplay, ReplayBuffer, RolloutBuffer, A3C,
};

// Reinforcement Learning primitives
pub mod rl_core;
pub use rl_core::{AdvantageEstimation, PPOClip, PolicyGradient, ValueFunction};

// Meta-Learning primitives for few-shot learning
pub mod metalearning;
pub use metalearning::{MetaLoss, Reptile, TaskBatch, MAML};

// Neural Architecture Search primitives
pub mod nas;
pub use nas::{AdjacencyMatrix, DARTSCell, LayerOptions, NasOptimizer, SearchSpace, ENAS};

// Probabilistic primitives and variational inference
pub mod probabilistic;
pub use probabilistic::{
    conditional_entropy, entropy, kl_divergence, kl_divergence_continuous,
    kl_divergence_diag_gaussian, mutual_information, NormalDistribution, VariationalInference,
};

// Bayesian Neural Network primitives
pub mod bayesian_nn;
pub use bayesian_nn::{bbb_loss, reparameterize, BayesianLinear, MCDropout, VariationalLayer};

// Model compression primitives
pub mod compression;
pub use compression::{
    KnowledgeDistillation, QuantMode, Quantization, StructuredPruning, UnstructuredPruning,
};

// Distributed training primitives
pub mod distributed;
pub use distributed::{
    all_reduce, DistributedDataParallel, GradientBucket, GradientSynchronization, ReduceOp,
};
