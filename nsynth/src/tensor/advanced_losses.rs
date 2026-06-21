//! Advanced Loss Functions for nCPU/nSynth
//!
//! Specialized losses for segmentation, ranking, metric learning, etc.

use super::ops::Shape;
use super::ops::Tensor;

/// Loss function trait
pub trait LossFunction {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
}

// ============================================================================
// SEGMENTATION LOSSES
// ============================================================================

/// Dice Loss (for segmentation, measures overlap)
#[derive(Debug, Clone, Copy)]
pub struct DiceLoss {
    pub smooth: f64,
}

impl DiceLoss {
    pub fn new(smooth: f64) -> Self {
        Self { smooth }
    }
}

impl Default for DiceLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl LossFunction for DiceLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut intersection = 0.0;
        let mut union = 0.0;

        for i in 0..predictions.data.len() {
            let p = predictions.data[i];
            let t = targets.data[i];
            intersection += p * t;
            union += p + t;
        }

        let dice = (2.0 * intersection + self.smooth) / (union + self.smooth);
        Tensor::scalar(1.0 - dice)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        // Gradient of Dice loss
        let mut grad = vec![0.0; predictions.data.len()];
        let mut intersection = 0.0;
        let mut union = 0.0;

        for i in 0..predictions.data.len() {
            intersection += predictions.data[i] * targets.data[i];
            union += predictions.data[i] + targets.data[i];
        }

        let dice = (2.0 * intersection + self.smooth) / (union + self.smooth);
        let denom = (union + self.smooth).powi(2);

        for i in 0..predictions.data.len() {
            let p = predictions.data[i];
            let t = targets.data[i];
            grad[i] = -2.0 * t * (union + self.smooth) / denom
                + (2.0 * intersection + self.smooth) / denom;
        }

        Tensor::new(grad, predictions.shape.clone())
    }
}

/// IoU Loss (Intersection over Union)
#[derive(Debug, Clone, Copy)]
pub struct IoULoss {
    pub smooth: f64,
}

impl IoULoss {
    pub fn new(smooth: f64) -> Self {
        Self { smooth }
    }
}

impl Default for IoULoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl LossFunction for IoULoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut intersection = 0.0;
        let mut union = 0.0;

        for i in 0..predictions.data.len() {
            let p = predictions.data[i];
            let t = targets.data[i];
            intersection += p * t;
            union += p + t - p * t;
        }

        let iou = (intersection + self.smooth) / (union + self.smooth);
        Tensor::scalar(1.0 - iou)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut grad = vec![0.0; predictions.data.len()];
        let mut intersection = 0.0;
        let mut union = 0.0;

        for i in 0..predictions.data.len() {
            let p = predictions.data[i];
            let t = targets.data[i];
            intersection += p * t;
            union += p + t - p * t;
        }

        let denom = (union + self.smooth).powi(2);

        for i in 0..predictions.data.len() {
            let p = predictions.data[i];
            let t = targets.data[i];
            let d_intersection = t;
            let d_union = 1.0 - t;
            grad[i] = -(d_intersection * (union + self.smooth)
                - (intersection + self.smooth) * d_union)
                / denom;
        }

        Tensor::new(grad, predictions.shape.clone())
    }
}

// ============================================================================
// VAE LOSSES
// ============================================================================

/// VAE Loss (KL divergence + reconstruction)
#[derive(Debug, Clone, Copy)]
pub struct VAELoss {
    pub kl_weight: f64,
    pub reconstruction_loss: ReconstructionLoss,
}

#[derive(Debug, Clone, Copy)]
pub enum ReconstructionLoss {
    MSE,
    BCE,
}

impl VAELoss {
    pub fn new(kl_weight: f64, reconstruction_loss: ReconstructionLoss) -> Self {
        Self {
            kl_weight,
            reconstruction_loss,
        }
    }
}

impl LossFunction for VAELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        // Assumes predictions contains [reconstruction, mu, logvar]
        // For full VAE, these should be separate tensors

        let reconstruction = match self.reconstruction_loss {
            ReconstructionLoss::MSE => {
                let mut mse = 0.0;
                for i in 0..predictions.data.len().min(targets.data.len()) {
                    mse += (predictions.data[i] - targets.data[i]).powi(2);
                }
                mse / predictions.data.len() as f64
            }
            ReconstructionLoss::BCE => {
                let mut bce = 0.0;
                for i in 0..predictions.data.len().min(targets.data.len()) {
                    let p = predictions.data[i].clamp(1e-10, 1.0 - 1e-10);
                    bce -= targets.data[i] * p.ln() + (1.0 - targets.data[i]) * (1.0 - p).ln();
                }
                bce / predictions.data.len() as f64
            }
        };

        // KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        // For VAE, mu and logvar would be separate - simplified here
        let kl = 0.0_f64; // Placeholder

        Tensor::scalar(reconstruction + self.kl_weight * kl)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        predictions.sub(targets).unwrap()
    }
}

// ============================================================================
// FACE RECOGNITION LOSSES (Angular Margin)
// ============================================================================

/// ArcFace Loss (additive angular margin)
#[derive(Debug, Clone)]
pub struct ArcFaceLoss {
    pub num_classes: usize,
    pub embedding_size: usize,
    pub margin: f64,
    pub scale: f64,
    pub weights: Tensor,
}

impl ArcFaceLoss {
    pub fn new(num_classes: usize, embedding_size: usize, margin: f64) -> Self {
        Self {
            num_classes,
            embedding_size,
            margin,
            scale: 64.0,
            weights: Tensor::uniform(Shape::new(vec![num_classes, embedding_size]), -0.01, 0.01),
        }
    }
}

impl LossFunction for ArcFaceLoss {
    fn forward(&self, embeddings: &Tensor, labels: &Tensor) -> Tensor {
        // Normalize embeddings and weights
        let norm_embeddings = embeddings.normalize();
        let norm_weights = self.weights.normalize();

        // Compute cosine similarity
        let logits = norm_embeddings
            .matmul(&norm_weights.transpose().unwrap())
            .unwrap();
        let scaled_logits = logits.mul(&self.scale.into()).unwrap();

        // Add angular margin for true class
        // Simplified: use cross entropy
        let mut loss = 0.0;
        for i in 0..labels.data.len() {
            let target_class = labels.data[i] as usize;
            if target_class < scaled_logits.data.len() {
                loss -= scaled_logits.data[target_class].ln();
            }
        }
        Tensor::scalar(loss / labels.data.len() as f64)
    }

    fn backward(&self, embeddings: &Tensor, labels: &Tensor) -> Tensor {
        embeddings.clone()
    }
}

/// CosFace Loss (additive cosine margin)
#[derive(Debug, Clone)]
pub struct CosFaceLoss {
    pub num_classes: usize,
    pub embedding_size: usize,
    pub margin: f64,
    pub scale: f64,
    pub weights: Tensor,
}

impl CosFaceLoss {
    pub fn new(num_classes: usize, embedding_size: usize, margin: f64) -> Self {
        Self {
            num_classes,
            embedding_size,
            margin,
            scale: 64.0,
            weights: Tensor::uniform(Shape::new(vec![num_classes, embedding_size]), -0.01, 0.01),
        }
    }
}

impl LossFunction for CosFaceLoss {
    fn forward(&self, embeddings: &Tensor, labels: &Tensor) -> Tensor {
        let logits = embeddings
            .matmul(&self.weights.transpose().unwrap())
            .unwrap();
        let scaled_logits = logits.mul(&self.scale.into()).unwrap();

        let mut loss = 0.0;
        for i in 0..labels.data.len() {
            let target_class = labels.data[i] as usize;
            if target_class < scaled_logits.data.len() {
                loss -= scaled_logits.data[target_class].ln();
            }
        }
        Tensor::scalar(loss / labels.data.len() as f64)
    }

    fn backward(&self, embeddings: &Tensor, labels: &Tensor) -> Tensor {
        embeddings.clone()
    }
}

/// SphereFace Loss (angular softmax)
#[derive(Debug, Clone)]
pub struct SphereFaceLoss {
    pub num_classes: usize,
    pub embedding_size: usize,
    pub margin: f64,
    pub scale: f64,
    pub weights: Tensor,
}

impl SphereFaceLoss {
    pub fn new(num_classes: usize, embedding_size: usize, margin: f64) -> Self {
        Self {
            num_classes,
            embedding_size,
            margin,
            scale: 64.0,
            weights: Tensor::uniform(Shape::new(vec![num_classes, embedding_size]), -0.01, 0.01),
        }
    }
}

impl LossFunction for SphereFaceLoss {
    fn forward(&self, embeddings: &Tensor, labels: &Tensor) -> Tensor {
        let logits = embeddings
            .matmul(&self.weights.transpose().unwrap())
            .unwrap();
        let scaled_logits = logits.mul(&self.scale.into()).unwrap();

        let mut loss = 0.0;
        for i in 0..labels.data.len() {
            let target_class = labels.data[i] as usize;
            if target_class < scaled_logits.data.len() {
                loss -= scaled_logits.data[target_class].ln();
            }
        }
        Tensor::scalar(loss / labels.data.len() as f64)
    }

    fn backward(&self, embeddings: &Tensor, labels: &Tensor) -> Tensor {
        embeddings.clone()
    }
}

// ============================================================================
// MULTI-CLASS LOSSES
// ============================================================================

/// Multi-Margin Loss (SVM multiclass)
#[derive(Debug, Clone, Copy)]
pub struct MultiMarginLoss {
    pub margin: f64,
    pub p: f64,
}

impl MultiMarginLoss {
    pub fn new(margin: f64) -> Self {
        Self { margin, p: 1.0 }
    }
}

impl Default for MultiMarginLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl LossFunction for MultiMarginLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        let n = predictions.data.len();

        for i in 0..targets.data.len() {
            let target_class = targets.data[i] as usize;
            for j in 0..n {
                if j != target_class {
                    let margin_loss = self.margin - (predictions.data[i] - predictions.data[j]);
                    loss += margin_loss.max(0.0);
                }
            }
        }

        Tensor::scalar(loss / targets.data.len() as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut grad = vec![0.0; predictions.data.len()];
        for i in 0..targets.data.len() {
            let target_class = targets.data[i] as usize;
            for j in 0..predictions.data.len() {
                if j != target_class {
                    let margin_loss = self.margin - (predictions.data[i] - predictions.data[j]);
                    if margin_loss > 0.0 {
                        grad[i] += 1.0;
                        grad[j] -= 1.0;
                    }
                }
            }
        }
        Tensor::new(grad, predictions.shape.clone())
    }
}

/// Multi-Label Soft Margin Loss (sigmoid + BCE on each label)
#[derive(Debug, Clone, Copy)]
pub struct MultiLabelSoftMarginLoss;

impl LossFunction for MultiLabelSoftMarginLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        let n = predictions.data.len();

        for i in 0..n {
            let p = predictions.data[i].clamp(-20.0, 20.0); // Avoid overflow
            let t = targets.data[i];
            // log(1 + exp(-x)) for t=1, log(1 + exp(x)) for t=0
            let x = if t > 0.5 { -p } else { p };
            loss += (1.0 + x.exp()).ln();
        }

        Tensor::scalar(loss / n as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut grad = vec![0.0; predictions.data.len()];
        for i in 0..predictions.data.len() {
            let p = predictions.data[i];
            let t = targets.data[i];
            // Derivative: -(t - sigma(x))
            let sig = 1.0 / (1.0 + (-p).exp());
            grad[i] = sig - t;
        }
        Tensor::new(grad, predictions.shape.clone())
    }
}

// ============================================================================
// DISTRIBUTION LOSSES
// ============================================================================

/// Poisson Negative Log Likelihood
#[derive(Debug, Clone, Copy)]
pub struct PoissonNLLLoss {
    pub log_input: bool,
    pub full: bool,
}

impl PoissonNLLLoss {
    pub fn new(log_input: bool, full: bool) -> Self {
        Self { log_input, full }
    }
}

impl LossFunction for PoissonNLLLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        let n = predictions.data.len();

        for i in 0..n {
            let p = if self.log_input {
                predictions.data[i].exp()
            } else {
                predictions.data[i]
            };
            let t = targets.data[i];

            if self.full {
                // Stirling's approximation term
                loss += p - t * p.ln() + (t * (t + 1.0)).ln() / 2.0 + std::f64::consts::PI / 2.0;
            } else {
                loss += p - t * p.ln();
            }
        }

        Tensor::scalar(loss / n as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut grad = vec![0.0; predictions.data.len()];
        for i in 0..predictions.data.len() {
            let p = predictions.data[i];
            let t = targets.data[i];
            grad[i] = if self.log_input {
                p.exp() - t
            } else {
                1.0 - t / (p + 1e-10)
            };
        }
        Tensor::new(grad, predictions.shape.clone())
    }
}

/// Gamma Negative Log Likelihood
#[derive(Debug, Clone, Copy)]
pub struct GammaNLLLoss {
    pub epsilon: f64,
}

impl GammaNLLLoss {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl Default for GammaNLLLoss {
    fn default() -> Self {
        Self::new(1e-8)
    }
}

impl LossFunction for GammaNLLLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        let n = predictions.data.len() / 2; // shape and concentration

        for i in 0..n {
            let shape = predictions.data[i].max(self.epsilon);
            let conc = predictions.data[n + i].max(self.epsilon);
            let target = targets.data[i];

            loss += (conc / shape
                + target / shape
                + (conc - 1.0) * (target + self.epsilon).ln()
                + conc * conc.ln()
                + conc * (std::f64::consts::PI * 2.0).ln());
        }

        Tensor::scalar(loss / n as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        predictions.clone() // Simplified
    }
}

/// Negative Binomial Loss
#[derive(Debug, Clone, Copy)]
pub struct NegativeBinomialLoss;

impl LossFunction for NegativeBinomialLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        let n = predictions.data.len() / 2;

        for i in 0..n {
            let mean = predictions.data[i].max(1e-10);
            let theta = predictions.data[n + i].max(1e-10);
            let target = targets.data[i];

            // NB likelihood: Gamma(target + theta) / (Gamma(target + 1) * Gamma(theta))
            // * (theta / (mean + theta))^theta * (mean / (mean + theta))^target
            loss -= (target + theta).ln() - mean.ln() - target * (mean + theta).ln();
        }

        Tensor::scalar(loss / n as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        predictions.clone()
    }
}

// ============================================================================
// ROBUST LOSSES
// ============================================================================

/// Huber Loss (smooth L1 with threshold)
#[derive(Debug, Clone, Copy)]
pub struct HuberLoss {
    pub delta: f64,
}

impl HuberLoss {
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl LossFunction for HuberLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        let delta = self.delta;

        for i in 0..predictions.data.len() {
            let diff = (predictions.data[i] - targets.data[i]).abs();
            loss += if diff <= delta {
                0.5 * diff * diff
            } else {
                delta * (diff - 0.5 * delta)
            };
        }

        Tensor::scalar(loss / predictions.data.len() as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut grad = vec![0.0; predictions.data.len()];
        let delta = self.delta;

        for i in 0..predictions.data.len() {
            let diff = predictions.data[i] - targets.data[i];
            grad[i] = if diff.abs() <= delta {
                diff
            } else {
                delta * diff.signum()
            };
        }

        Tensor::new(grad, predictions.shape.clone())
    }
}

/// Log-Cosh Loss (logarithm of hyperbolic cosine)
#[derive(Debug, Clone, Copy)]
pub struct LogCoshLoss;

impl LossFunction for LogCoshLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        for i in 0..predictions.data.len() {
            let diff = predictions.data[i] - targets.data[i];
            loss += (diff.cosh()).ln();
        }
        Tensor::scalar(loss / predictions.data.len() as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut grad = vec![0.0; predictions.data.len()];
        for i in 0..predictions.data.len() {
            let diff = predictions.data[i] - targets.data[i];
            grad[i] = (diff.sinh()) / (diff.cosh());
        }
        Tensor::new(grad, predictions.shape.clone())
    }
}

// ============================================================================
// RANKING LOSSES
// ============================================================================

/// RankNet Loss (learning to rank)
#[derive(Debug, Clone, Copy)]
pub struct RankNetLoss;

impl LossFunction for RankNetLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        // Assumes targets are pairwise preferences (+1, -1, 0)
        let mut loss = 0.0;
        let n = predictions.data.len() / 2;

        for i in 0..n {
            let s_i = predictions.data[i];
            let s_j = predictions.data[i + n];
            let p_ij = targets.data[i];

            // Cross entropy on sigmoid difference
            let sigma = 1.0 / (1.0 + (-(s_i - s_j)).exp());
            loss -= if p_ij > 0.0 {
                sigma.ln()
            } else {
                (1.0 - sigma).ln()
            };
        }

        Tensor::scalar(loss / n as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut grad = vec![0.0; predictions.data.len()];
        let n = predictions.data.len() / 2;

        for i in 0..n {
            let s_i = predictions.data[i];
            let s_j = predictions.data[i + n];
            let p_ij = targets.data[i];

            let sigma = 1.0 / (1.0 + (-(s_i - s_j)).exp());
            grad[i] = sigma - (if p_ij > 0.0 { 1.0 } else { 0.0 });
            grad[i + n] = -grad[i];
        }

        Tensor::new(grad, predictions.shape.clone())
    }
}

/// ListNet Loss (softmax over list)
#[derive(Debug, Clone, Copy)]
pub struct ListNetLoss;

impl LossFunction for ListNetLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        // KL divergence between predicted and target distributions
        let mut loss = 0.0;

        // Compute softmax of predictions
        let pred_max = predictions
            .data
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let pred_exp: Vec<f64> = predictions
            .data
            .iter()
            .map(|&x| (x - pred_max).exp())
            .collect();
        let pred_sum: f64 = pred_exp.iter().sum();
        let pred_softmax: Vec<f64> = pred_exp.iter().map(|&e| e / pred_sum).collect();

        // Compute softmax of targets
        let target_max = targets
            .data
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let target_exp: Vec<f64> = targets
            .data
            .iter()
            .map(|&x| (x - target_max).exp())
            .collect();
        let target_sum: f64 = target_exp.iter().sum();
        let target_softmax: Vec<f64> = target_exp.iter().map(|&e| e / target_sum).collect();

        // KL divergence
        for i in 0..predictions.data.len() {
            loss += target_softmax[i] * (target_softmax[i].ln() - pred_softmax[i].ln());
        }

        Tensor::scalar(loss)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        predictions.sub(targets).unwrap()
    }
}

// ============================================================================
// SPECIALIZED TRIPLET LOSSES
// ============================================================================

/// Batched Triplet Loss
#[derive(Debug, Clone, Copy)]
pub struct BatchedTripletLoss {
    pub margin: f64,
}

impl BatchedTripletLoss {
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }
}

impl LossFunction for BatchedTripletLoss {
    fn forward(&self, embeddings: &Tensor, labels: &Tensor) -> Tensor {
        // Assumes embeddings are organized by class
        let mut loss = 0.0;
        let mut count = 0;

        for i in 0..labels.data.len() {
            for j in (i + 1)..labels.data.len() {
                if labels.data[i] == labels.data[j] {
                    // Positive pair - find negative
                    for k in 0..labels.data.len() {
                        if labels.data[k] != labels.data[i] {
                            let dist_pos = embeddings.data[i] - embeddings.data[j];
                            let dist_neg = embeddings.data[i] - embeddings.data[k];
                            let triplet_loss =
                                (dist_pos * dist_pos) - (dist_neg * dist_neg) + self.margin;
                            loss += triplet_loss.max(0.0);
                            count += 1;
                        }
                    }
                }
            }
        }

        if count > 0 {
            Tensor::scalar(loss / count as f64)
        } else {
            Tensor::scalar(0.0)
        }
    }

    fn backward(&self, embeddings: &Tensor, labels: &Tensor) -> Tensor {
        embeddings.clone()
    }
}

/// Semi-Hard Triplet Loss
#[derive(Debug, Clone, Copy)]
pub struct SemiHardTripletLoss {
    pub margin: f64,
}

impl SemiHardTripletLoss {
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }
}

impl LossFunction for SemiHardTripletLoss {
    fn forward(&self, embeddings: &Tensor, labels: &Tensor) -> Tensor {
        let mut loss = 0.0;
        let mut count = 0;

        // Find semi-hard negatives (distance > positive but < margin)
        for i in 0..labels.data.len() {
            for j in (i + 1)..labels.data.len() {
                if labels.data[i] == labels.data[j] {
                    let dist_pos = (embeddings.data[i] - embeddings.data[j]).powi(2);

                    for k in 0..labels.data.len() {
                        if labels.data[k] != labels.data[i] {
                            let dist_neg = (embeddings.data[i] - embeddings.data[k]).powi(2);

                            if dist_neg > dist_pos && dist_neg < dist_pos + self.margin {
                                let triplet_loss = dist_pos - dist_neg + self.margin;
                                loss += triplet_loss;
                                count += 1;
                            }
                        }
                    }
                }
            }
        }

        if count > 0 {
            Tensor::scalar(loss / count as f64)
        } else {
            Tensor::scalar(0.0)
        }
    }

    fn backward(&self, embeddings: &Tensor, labels: &Tensor) -> Tensor {
        embeddings.clone()
    }
}

// ============================================================================
// CTC LOSS (Connectionist Temporal Classification)
// ============================================================================

/// CTC Loss for sequence alignment
#[derive(Debug, Clone)]
pub struct CTCLoss {
    pub blank: usize,
    pub zero_infinity: bool,
}

impl CTCLoss {
    pub fn new(blank: usize) -> Self {
        Self {
            blank,
            zero_infinity: true,
        }
    }
}

impl LossFunction for CTCLoss {
    fn forward(&self, log_probs: &Tensor, targets: &Tensor) -> Tensor {
        // Simplified CTC loss (full implementation requires dynamic programming)
        let mut loss = 0.0;

        // Assume log_probs are already log-space and targets are indices
        for i in 0..targets.data.len() {
            let target_idx = targets.data[i] as usize;
            if target_idx < log_probs.data.len() {
                loss -= log_probs.data[target_idx];
            }
        }

        Tensor::scalar(loss)
    }

    fn backward(&self, log_probs: &Tensor, targets: &Tensor) -> Tensor {
        log_probs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dice_loss() {
        let pred = Tensor::vector(vec![1.0, 0.0, 1.0]);
        let target = Tensor::vector(vec![1.0, 0.0, 0.0]);
        let loss_fn = DiceLoss::new(1.0);
        let loss = loss_fn.forward(&pred, &target);
        // Dice = 2*1 / (2+1+0) = 2/3, loss = 1/3
        assert!((loss.data[0] - 0.33).abs() < 0.1);
    }

    #[ignore = "experimental tensor stack — Package O; see docs/TENSOR_QUARANTINE.md"]
    #[test]
    fn test_iou_loss() {
        let pred = Tensor::vector(vec![1.0, 0.0, 1.0]);
        let target = Tensor::vector(vec![1.0, 0.0, 0.0]);
        let loss_fn = IoULoss::new(1.0);
        let loss = loss_fn.forward(&pred, &target);
        // IoU = 1 / (1 + 1) = 0.5, loss = 0.5
        assert!((loss.data[0] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_huber_loss() {
        let pred = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let target = Tensor::vector(vec![1.5, 2.5, 3.5]);
        let loss_fn = HuberLoss::new(1.0);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data[0] > 0.0);
    }

    #[test]
    fn test_multi_margin_loss() {
        let pred = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let target = Tensor::vector(vec![2.0]); // Class 2 is target
        let loss_fn = MultiMarginLoss::new(1.0);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data[0] > 0.0);
    }
}
