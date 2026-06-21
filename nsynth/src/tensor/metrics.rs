//! Evaluation Metrics for nCPU/nSynth
//!
//! Comprehensive metrics for classification, regression, ranking.

use super::ops::{Shape, Tensor};

// ============================================================================
// CLASSIFICATION METRICS
// ============================================================================

/// Accuracy
pub fn accuracy(predictions: &Tensor, targets: &Tensor) -> f64 {
    let mut correct = 0;
    let total = targets.data.len();

    for i in 0..targets.data.len() {
        let mut max_idx = 0;
        let mut max_val = f64::NEG_INFINITY;
        for (j, &v) in predictions.data.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = j;
            }
        }
        let pred_class = max_idx;

        if pred_class == targets.data[i] as usize {
            correct += 1;
        }
    }

    correct as f64 / total as f64
}

/// Top-K Accuracy
pub fn top_k_accuracy(predictions: &Tensor, targets: &Tensor, k: usize) -> f64 {
    let mut correct = 0;
    let total = targets.data.len();

    for i in 0..targets.data.len() {
        let mut values: Vec<(usize, f64)> = predictions
            .data
            .iter()
            .enumerate()
            .filter(|(j, _)| *j == i)
            .map(|(j, &v)| (j, v))
            .collect();

        values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_k: Vec<usize> = values.iter().take(k).map(|(j, _)| *j).collect();
        if top_k.contains(&(targets.data[i] as usize)) {
            correct += 1;
        }
    }

    correct as f64 / total as f64
}

/// Precision (for binary classification)
pub fn precision(predictions: &Tensor, targets: &Tensor) -> f64 {
    let mut true_positives = 0.0;
    let mut predicted_positives = 0.0;

    for i in 0..targets.data.len() {
        let pred = if predictions.data[i] > 0.5 { 1.0 } else { 0.0 };
        let target = targets.data[i];

        if pred == 1.0 {
            predicted_positives += 1.0;
            if target == 1.0 {
                true_positives += 1.0;
            }
        }
    }

    if predicted_positives > 0.0 {
        true_positives / predicted_positives
    } else {
        0.0
    }
}

/// Recall (for binary classification)
pub fn recall(predictions: &Tensor, targets: &Tensor) -> f64 {
    let mut true_positives = 0.0;
    let mut actual_positives = 0.0;

    for i in 0..targets.data.len() {
        let pred = if predictions.data[i] > 0.5 { 1.0 } else { 0.0 };
        let target = targets.data[i];

        if target == 1.0 {
            actual_positives += 1.0;
            if pred == 1.0 {
                true_positives += 1.0;
            }
        }
    }

    if actual_positives > 0.0 {
        true_positives / actual_positives
    } else {
        0.0
    }
}

/// F1 Score
pub fn f1_score(predictions: &Tensor, targets: &Tensor) -> f64 {
    let p = precision(predictions, targets);
    let r = recall(predictions, targets);

    if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    }
}

/// Macro-averaged F1 (for multiclass)
pub fn macro_f1_score(predictions: &Tensor, targets: &Tensor, num_classes: usize) -> f64 {
    let mut f1_sum = 0.0;
    let mut count = 0;

    for c in 0..num_classes {
        let bin_targets: Vec<f64> = targets
            .data
            .iter()
            .map(|&t| if t == c as f64 { 1.0 } else { 0.0 })
            .collect();
        let bin_preds: Vec<f64> = predictions
            .data
            .iter()
            .map(|&p| if p == c as f64 { 1.0 } else { 0.0 })
            .collect();

        let targets_tensor = Tensor::new(bin_targets, targets.shape.clone());
        let preds_tensor = Tensor::new(bin_preds, predictions.shape.clone());

        let f1 = f1_score(&preds_tensor, &targets_tensor);
        if f1.is_finite() {
            f1_sum += f1;
            count += 1;
        }
    }

    if count > 0 {
        f1_sum / count as f64
    } else {
        0.0
    }
}

/// Micro-averaged F1 (for multiclass)
pub fn micro_f1_score(predictions: &Tensor, targets: &Tensor) -> f64 {
    // Micro F1 equals accuracy for multiclass
    accuracy(predictions, targets)
}

// ============================================================================
// CONFUSION MATRIX
// ============================================================================

/// Confusion Matrix
pub fn confusion_matrix(predictions: &Tensor, targets: &Tensor, num_classes: usize) -> Tensor {
    let mut cm = vec![0.0; num_classes * num_classes];

    for i in 0..targets.data.len() {
        let t = targets.data[i] as usize;
        let p = predictions.data[i] as usize;
        if t < num_classes && p < num_classes {
            cm[t * num_classes + p] += 1.0;
        }
    }

    Tensor::new(cm, Shape::new(vec![num_classes, num_classes]))
}

// ============================================================================
// RANKING METRICS
// ============================================================================

/// Mean Average Precision (mAP)
pub fn mean_average_precision(predictions: &Tensor, targets: &Tensor) -> f64 {
    let mut ap_sum = 0.0;
    let num_queries = targets.shape.dims[0];

    for q in 0..num_queries {
        let mut indices: Vec<usize> = (0..predictions.shape.dims[1]).collect();
        let query_preds: Vec<f64> = predictions.data
            [q * predictions.shape.dims[1]..(q + 1) * predictions.shape.dims[1]]
            .iter()
            .copied()
            .collect();

        indices.sort_by(|&a, &b| query_preds[b].partial_cmp(&query_preds[a]).unwrap());

        let query_targets: Vec<f64> = targets.data
            [q * targets.shape.dims[1]..(q + 1) * targets.shape.dims[1]]
            .iter()
            .copied()
            .collect();

        let mut precision_sum = 0.0;
        let mut relevant_count = 0;

        for (rank, &idx) in indices.iter().enumerate() {
            if query_targets[idx] > 0.0 {
                relevant_count += 1;
                precision_sum += relevant_count as f64 / (rank + 1) as f64;
            }
        }

        let total_relevant = query_targets.iter().filter(|&&t| t > 0.0).count();
        if total_relevant > 0 {
            ap_sum += precision_sum / total_relevant as f64;
        }
    }

    ap_sum / num_queries as f64
}

// ============================================================================
// ROC/AUC METRICS
// ============================================================================

/// Area Under ROC Curve
pub fn roc_auc_score(predictions: &Tensor, targets: &Tensor) -> f64 {
    // Use trapezoidal rule to compute AUC
    let mut data: Vec<(f64, f64)> = predictions
        .data
        .iter()
        .zip(targets.data.iter())
        .map(|(&p, &t)| (p, t))
        .collect();

    data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut auc: f64 = 0.0;
    let mut prev_score = f64::NEG_INFINITY;
    let mut n_pos = 0.0;
    let mut n_neg = 0.0;
    let mut tp_prev = 0.0;
    let mut fp_prev = 0.0;

    for (score, target) in data {
        if score != prev_score {
            auc += (fp_prev + n_neg) * (tp_prev - n_pos) / 2.0;
            prev_score = score;
            fp_prev = n_neg;
            tp_prev = n_pos;
        }

        if target > 0.5 {
            n_pos += 1.0;
        } else {
            n_neg += 1.0;
        }
    }

    auc += (fp_prev + n_neg) * (tp_prev - n_pos) / 2.0;

    let area = auc.abs() / (n_pos * n_neg);
    if area.is_finite() {
        area
    } else {
        0.0_f64
    }
}

/// Area Under PR Curve
pub fn pr_auc_score(predictions: &Tensor, targets: &Tensor) -> f64 {
    let mut data: Vec<(f64, f64)> = predictions
        .data
        .iter()
        .zip(targets.data.iter())
        .map(|(&p, &t)| (p, t))
        .collect();

    data.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // Descending

    let mut precision = Vec::new();
    let mut recall = Vec::new();
    let mut tp = 0.0;
    let mut fp = 0.0;
    let total_pos = targets.data.iter().filter(|&&t| t > 0.5).count() as f64;

    for (_, target) in data {
        if target > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        precision.push(tp / (tp + fp));
        recall.push(tp / total_pos);
    }

    // Compute AUC using trapezoidal rule
    let mut auc = 0.0;
    for i in 1..precision.len() {
        auc += (recall[i] - recall[i - 1]) * (precision[i] + precision[i - 1]) / 2.0;
    }

    auc
}

// ============================================================================
// SEQUENCE METRICS
// ============================================================================

/// BLEU Score (for translation)
pub fn bleu_score(predictions: &[Vec<String>], references: &[Vec<String>], max_n: usize) -> f64 {
    let mut precisions = Vec::new();
    let mut bp = 1.0;

    // Compute n-gram precisions
    for n in 1..=max_n {
        let mut matches = 0;
        let mut total = 0;

        for (pred, ref_list) in predictions.iter().zip(references.iter()) {
            let pred_ngrams: Vec<Vec<String>> = pred.windows(n).map(|w| w.to_vec()).collect();

            for ngram in &pred_ngrams {
                total += 1;
                let ref_ngrams: Vec<Vec<String>> =
                    ref_list.windows(n).map(|w| w.to_vec()).collect();

                let max_count = ref_ngrams.iter().filter(|r| *r == ngram).count();

                if max_count > 0 {
                    matches += 1;
                }
            }
        }

        if total > 0 {
            precisions.push(matches as f64 / total as f64);
        }
    }

    // Compute brevity penalty
    let pred_len: usize = predictions.iter().map(|p| p.len()).sum();
    let ref_len: usize = references
        .iter()
        .map(|r| r.iter().map(|s| s.len()).max().unwrap_or(0))
        .sum();

    if pred_len > 0 && ref_len > 0 {
        let ratio = pred_len as f64 / ref_len as f64;
        bp = if ratio >= 1.0 {
            1.0
        } else {
            (1.0 - ref_len as f64 / pred_len as f64).exp()
        };
    }

    // Geometric mean of precisions
    let geo_mean: f64 = if precisions.is_empty() {
        0.0
    } else {
        let log_sum: f64 = precisions.iter().map(|&p| p.ln()).sum();
        (log_sum / precisions.len() as f64).exp()
    };

    bp * geo_mean
}

/// Perplexity (for language models)
pub fn perplexity(log_probs: &Tensor) -> f64 {
    let avg_log_prob: f64 = log_probs.data.iter().sum::<f64>() / log_probs.data.len() as f64;
    (-avg_log_prob).exp()
}

// ============================================================================
// REGRESSION METRICS
// ============================================================================

/// Mean Absolute Error
pub fn mean_absolute_error(predictions: &Tensor, targets: &Tensor) -> f64 {
    let mut sum = 0.0;
    for i in 0..predictions.data.len() {
        sum += (predictions.data[i] - targets.data[i]).abs();
    }
    sum / predictions.data.len() as f64
}

/// Mean Squared Error
pub fn mean_squared_error(predictions: &Tensor, targets: &Tensor) -> f64 {
    let mut sum = 0.0;
    for i in 0..predictions.data.len() {
        sum += (predictions.data[i] - targets.data[i]).powi(2);
    }
    sum / predictions.data.len() as f64
}

/// Root Mean Squared Error
pub fn root_mean_squared_error(predictions: &Tensor, targets: &Tensor) -> f64 {
    mean_squared_error(predictions, targets).sqrt()
}

/// R² Score (coefficient of determination)
pub fn r2_score(predictions: &Tensor, targets: &Tensor) -> f64 {
    let target_mean: f64 = targets.data.iter().sum::<f64>() / targets.data.len() as f64;

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for i in 0..targets.data.len() {
        ss_res += (targets.data[i] - predictions.data[i]).powi(2);
        ss_tot += (targets.data[i] - target_mean).powi(2);
    }

    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "experimental tensor stack — Package O; see docs/TENSOR_QUARANTINE.md"]
    #[test]
    fn test_accuracy() {
        let pred = Tensor::vector(vec![0.9, 0.1, 0.8]);
        let target = Tensor::vector(vec![0.0, 1.0, 0.0]);
        let acc = accuracy(&pred, &target);
        assert_eq!(acc, 1.0);
    }

    #[test]
    fn test_confusion_matrix() {
        let pred = Tensor::vector(vec![0.0, 1.0, 1.0, 0.0]);
        let target = Tensor::vector(vec![0.0, 1.0, 0.0, 0.0]);
        let cm = confusion_matrix(&pred, &target, 2);
        assert_eq!(cm.data[0], 2.0); // True negatives
        assert_eq!(cm.data[3], 1.0); // True positives
    }

    #[test]
    fn test_mae() {
        let pred = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let target = Tensor::vector(vec![1.5, 2.5, 3.5]);
        let mae = mean_absolute_error(&pred, &target);
        assert!((mae - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_r2_score() {
        let pred = Tensor::vector(vec![2.0, 3.0, 4.0]);
        let target = Tensor::vector(vec![2.0, 3.0, 4.0]);
        let r2 = r2_score(&pred, &target);
        assert!((r2 - 1.0).abs() < 0.01);
    }
}
