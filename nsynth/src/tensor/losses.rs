//! Comprehensive Loss Functions for nCPU/nSynth
//!
//! All common loss functions with gradients.

use super::ops::Tensor;

/// Loss function trait
pub trait LossFunction {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
}

/// Mean Squared Error (MSE)
#[derive(Debug, Clone, Copy)]
pub struct MSELoss;

impl LossFunction for MSELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        for i in 0..predictions.data.len() {
            loss += (predictions.data[i] - targets.data[i]).powi(2);
        }
        Tensor::scalar(loss / predictions.data.len() as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let n = predictions.data.len() as f64;
        Tensor::new(
            predictions
                .data
                .iter()
                .zip(targets.data.iter())
                .map(|(&p, &t)| 2.0 * (p - t) / n)
                .collect(),
            predictions.shape.clone(),
        )
    }
}

/// Mean Absolute Error (MAE)
#[derive(Debug, Clone, Copy)]
pub struct MAELoss;

impl LossFunction for MAELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        for i in 0..predictions.data.len() {
            loss += (predictions.data[i] - targets.data[i]).abs();
        }
        Tensor::scalar(loss / predictions.data.len() as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let n = predictions.data.len() as f64;
        Tensor::new(
            predictions
                .data
                .iter()
                .zip(targets.data.iter())
                .map(|(&p, &t)| {
                    if p > t {
                        1.0 / n
                    } else if p < t {
                        -1.0 / n
                    } else {
                        0.0
                    }
                })
                .collect(),
            predictions.shape.clone(),
        )
    }
}

/// Cross-Entropy Loss
#[derive(Debug, Clone, Copy)]
pub struct CrossEntropyLoss;

impl LossFunction for CrossEntropyLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        for i in 0..predictions.data.len() {
            let p = predictions.data[i].max(1e-10);
            let t = targets.data[i];
            loss -= t * p.ln();
        }
        Tensor::scalar(loss / predictions.data.len() as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        predictions.sub(targets).unwrap()
    }
}

/// Binary Cross-Entropy Loss
#[derive(Debug, Clone, Copy)]
pub struct BCELoss;

impl LossFunction for BCELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        for i in 0..predictions.data.len() {
            let p = predictions.data[i].clamp(1e-10, 1.0 - 1e-10);
            let t = targets.data[i];
            loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
        }
        Tensor::scalar(loss / predictions.data.len() as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let n = predictions.data.len() as f64;
        Tensor::new(
            predictions
                .data
                .iter()
                .zip(targets.data.iter())
                .map(|(&p, &t)| {
                    let p_clamped = p.clamp(1e-10, 1.0 - 1e-10);
                    -(t / p_clamped - (1.0 - t) / (1.0 - p_clamped)) / n
                })
                .collect(),
            predictions.shape.clone(),
        )
    }
}

/// Hinge Loss (for SVM)
#[derive(Debug, Clone, Copy)]
pub struct HingeLoss;

impl LossFunction for HingeLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        for i in 0..predictions.data.len() {
            let val = 1.0 - targets.data[i] * predictions.data[i];
            loss += val.max(0.0);
        }
        Tensor::scalar(loss / predictions.data.len() as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        Tensor::new(
            predictions
                .data
                .iter()
                .zip(targets.data.iter())
                .map(|(&p, &t)| {
                    let val = 1.0 - t * p;
                    if val > 0.0 {
                        -t
                    } else {
                        0.0
                    }
                })
                .collect(),
            predictions.shape.clone(),
        )
    }
}

/// Kullback-Leibler Divergence
#[derive(Debug, Clone, Copy)]
pub struct KLDivLoss;

impl LossFunction for KLDivLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        for i in 0..predictions.data.len() {
            let p = predictions.data[i].clamp(1e-10, 1.0);
            let t = targets.data[i].clamp(1e-10, 1.0);
            loss += t * (t.ln() - p.ln());
        }
        Tensor::scalar(loss / predictions.data.len() as f64)
    }

    fn backward(&self, predictions: &Tensor, _targets: &Tensor) -> Tensor {
        Tensor::new(
            predictions
                .data
                .iter()
                .map(|&p| -1.0 / p.max(1e-10))
                .collect(),
            predictions.shape.clone(),
        )
    }
}

/// Cosine Embedding Loss
#[derive(Debug, Clone, Copy)]
pub struct CosineEmbeddingLoss {
    pub margin: f64,
}

impl CosineEmbeddingLoss {
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }
}

impl LossFunction for CosineEmbeddingLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        // Simplified for 1D: cos_sim = dot(p, t) / (||p|| * ||t||)
        let dot = predictions
            .data
            .iter()
            .zip(targets.data.iter())
            .map(|(&p, &t)| p * t)
            .sum::<f64>();

        let p_norm = predictions.data.iter().map(|&p| p * p).sum::<f64>().sqrt();
        let t_norm = targets.data.iter().map(|&t| t * t).sum::<f64>().sqrt();
        let cos_sim = dot / (p_norm * t_norm + 1e-10);

        Tensor::scalar((1.0_f64 - cos_sim).max(0.0_f64))
    }

    fn backward(&self, predictions: &Tensor, _targets: &Tensor) -> Tensor {
        // Simplified gradient
        predictions.clone()
    }
}

/// Triplet Margin Loss
#[derive(Debug, Clone, Copy)]
pub struct TripletMarginLoss {
    pub margin: f64,
    pub p: f64,
}

impl TripletMarginLoss {
    pub fn new(margin: f64) -> Self {
        Self { margin, p: 2.0 }
    }
}

impl LossFunction for TripletMarginLoss {
    fn forward(&self, anchor: &Tensor, positive: &Tensor) -> Tensor {
        // For simplified case: loss = max(0, ||anchor - positive|| - margin)
        let mut dist = 0.0;
        for i in 0..anchor.data.len() {
            dist += (anchor.data[i] - positive.data[i]).abs().powf(self.p);
        }
        let loss = dist.powf(1.0 / self.p) - self.margin;
        Tensor::scalar(loss.max(0.0_f64))
    }

    fn backward(&self, anchor: &Tensor, positive: &Tensor) -> Tensor {
        let diff = Tensor::new(
            anchor
                .data
                .iter()
                .zip(positive.data.iter())
                .map(|(&a, &p)| a - p)
                .collect(),
            anchor.shape.clone(),
        );
        diff.mul(&Tensor::scalar(self.margin.max(0.0_f64))).unwrap()
    }
}

/// Smooth L1 Loss (Huber loss)
#[derive(Debug, Clone, Copy)]
pub struct SmoothL1Loss {
    pub beta: f64,
}

impl SmoothL1Loss {
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl LossFunction for SmoothL1Loss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        for i in 0..predictions.data.len() {
            let diff = (predictions.data[i] - targets.data[i]).abs();
            if diff < self.beta {
                loss += 0.5 * diff * diff / self.beta;
            } else {
                loss += diff - 0.5 * self.beta;
            }
        }
        Tensor::scalar(loss / predictions.data.len() as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        Tensor::new(
            predictions
                .data
                .iter()
                .zip(targets.data.iter())
                .map(|(&p, &t)| {
                    let diff = p - t;
                    if diff.abs() < self.beta {
                        diff / self.beta
                    } else {
                        diff.signum()
                    }
                })
                .collect(),
            predictions.shape.clone(),
        )
    }
}

/// Negative Log Likelihood Loss
#[derive(Debug, Clone, Copy)]
pub struct NLLLoss;

impl LossFunction for NLLLoss {
    fn forward(&self, log_predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        for i in 0..log_predictions.data.len() {
            loss -= targets.data[i] * log_predictions.data[i];
        }
        Tensor::scalar(loss / log_predictions.data.len() as f64)
    }

    fn backward(&self, log_predictions: &Tensor, targets: &Tensor) -> Tensor {
        Tensor::new(
            log_predictions
                .data
                .iter()
                .map(|&lp| -lp.exp())
                .zip(targets.data.iter())
                .map(|(neg_exp, &t)| neg_exp * t)
                .collect(),
            log_predictions.shape.clone(),
        )
    }
}

/// Focal Loss (for imbalanced classification)
#[derive(Debug, Clone, Copy)]
pub struct FocalLoss {
    pub alpha: f64,
    pub gamma: f64,
}

impl FocalLoss {
    pub fn new(alpha: f64, gamma: f64) -> Self {
        Self { alpha, gamma }
    }
}

impl LossFunction for FocalLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let mut loss = 0.0;
        for i in 0..predictions.data.len() {
            let p = predictions.data[i].clamp(1e-10, 1.0 - 1e-10);
            let t = targets.data[i];
            let ce = -t * p.ln();
            let pt = if t == 1.0 { p } else { 1.0 - p };
            loss += self.alpha * ce * (1.0 - pt).powf(self.gamma);
        }
        Tensor::scalar(loss / predictions.data.len() as f64)
    }

    fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        // Simplified gradient
        predictions.sub(targets).unwrap()
    }
}

/// Contrastive Loss
#[derive(Debug, Clone, Copy)]
pub struct ContrastiveLoss {
    pub margin: f64,
}

impl ContrastiveLoss {
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }
}

impl LossFunction for ContrastiveLoss {
    fn forward(&self, distances: &Tensor, labels: &Tensor) -> Tensor {
        let mut loss = 0.0;
        for i in 0..distances.data.len() {
            let d = distances.data[i];
            let y = labels.data[i];
            if y == 1.0 {
                loss += d * d;
            } else {
                loss += (self.margin - d).max(0.0).powi(2);
            }
        }
        Tensor::scalar(loss / distances.data.len() as f64)
    }

    fn backward(&self, distances: &Tensor, labels: &Tensor) -> Tensor {
        Tensor::new(
            distances
                .data
                .iter()
                .zip(labels.data.iter())
                .map(|(&d, &y)| {
                    if y == 1.0 {
                        2.0 * d
                    } else if d < self.margin {
                        2.0 * (d - self.margin)
                    } else {
                        0.0
                    }
                })
                .collect(),
            distances.shape.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let pred = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let target = Tensor::vector(vec![1.5, 2.5, 3.5]);
        let loss = MSELoss.forward(&pred, &target);
        // MSE = ((0.5^2 + 0.5^2 + 0.5^2) / 3) = 0.25
        assert!((loss.data[0] - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_bce_loss() {
        let pred = Tensor::vector(vec![0.7, 0.3]);
        let target = Tensor::vector(vec![1.0, 0.0]);
        let loss = BCELoss.forward(&pred, &target);
        // BCE = -(ln(0.7) + ln(0.7)) / 2 ≈ 0.357
        assert!((loss.data[0] - 0.357).abs() < 0.01);
    }

    #[test]
    fn test_smooth_l1_loss() {
        let pred = Tensor::vector(vec![1.0, 2.0]);
        let target = Tensor::vector(vec![1.2, 1.8]);
        let loss_fn = SmoothL1Loss::new(0.5);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data[0] > 0.0);
    }
}
