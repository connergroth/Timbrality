# Calibration Head Neural Network

This document describes the Calibration Head, a specialized neural network component designed to transform heuristic confidence scores into well-calibrated, learned confidence estimates that accurately reflect prediction uncertainty in music recommendation systems.

## Overview

The Calibration Head addresses a fundamental challenge in machine learning: ensuring that model confidence scores accurately reflect true prediction reliability. In recommendation systems, confidence scores guide critical decisions about when to trust predictions, how to weight different signals, and when to fall back to alternative strategies. Poorly calibrated confidence can lead to overconfident poor recommendations or underconfident good recommendations.

## Problem Statement

### Confidence Calibration Challenge

Traditional confidence estimation approaches suffer from several limitations:

**Heuristic Confidence Limitations**

- Based on simple rules rather than learned patterns
- Cannot adapt to complex relationships between features and uncertainty
- Often miscalibrated, leading to overconfidence or underconfidence
- Limited ability to generalize across different data distributions

**Model Uncertainty vs Data Uncertainty**

- Epistemic uncertainty: Model's knowledge limitations
- Aleatoric uncertainty: Inherent data noise and ambiguity
- Standard models conflate these types, leading to poor calibration

**Context-Dependent Reliability**

- Confidence should vary based on user type, item characteristics, and context
- Static confidence scores cannot adapt to different recommendation scenarios
- Need for dynamic confidence that reflects situational uncertainty

### Expected Calibration Error (ECE)

Perfect calibration means that among all predictions with confidence p, exactly p fraction should be correct. The Expected Calibration Error measures deviation from this ideal:

```
ECE = Σ(|accuracy(bin) - confidence(bin)| × proportion(bin))
```

The Calibration Head aims to minimize ECE while maintaining prediction quality.

## Architecture

### Core Calibration Network

```python
class CalibrationHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = [64, 32],
        num_calibration_bins: int = 10,
        temperature_init: float = 1.0,
        use_temperature_scaling: bool = True
    ):
```

The Calibration Head consists of multiple components working together:

#### Input Feature Processing

**Prediction Context Features**

- Raw prediction scores from base models (CF, content-based)
- Model embeddings and intermediate representations
- Feature completeness and data quality indicators
- Uncertainty estimates from ensemble models

**Contextual Features**

- User behavior patterns and session characteristics
- Item metadata completeness and source reliability
- Temporal context (time of day, recency effects)
- Cross-signal agreement (CF vs content alignment)

```python
def _extract_calibration_features(
    self,
    prediction_score: torch.Tensor,
    model_embedding: torch.Tensor,
    context_features: torch.Tensor,
    cross_signal_agreement: torch.Tensor
) -> torch.Tensor:

    # Combine all calibration-relevant features
    calibration_features = torch.cat([
        prediction_score.unsqueeze(1),      # Raw score
        model_embedding,                    # Model representation
        context_features,                   # User/item context
        cross_signal_agreement.unsqueeze(1) # CF vs content agreement
    ], dim=1)

    return calibration_features
```

#### Neural Calibration Network

```python
def forward(
    self,
    prediction_logits: torch.Tensor,
    context_features: torch.Tensor,
    return_temperature: bool = False
) -> Dict[str, torch.Tensor]:

    # Extract features for calibration
    calibration_input = self._extract_calibration_features(
        prediction_logits, context_features
    )

    # Neural calibration network
    x = calibration_input
    for layer in self.calibration_layers:
        x = F.relu(layer(x))
        x = self.dropout(x)

    # Generate calibration parameters
    calibration_params = self.calibration_output(x)

    # Apply temperature scaling
    if self.use_temperature_scaling:
        temperature = F.softplus(calibration_params[:, 0]) + 0.1  # Ensure T > 0
        calibrated_logits = prediction_logits / temperature.unsqueeze(1)
    else:
        calibrated_logits = prediction_logits

    # Apply Platt scaling for additional calibration
    platt_a = calibration_params[:, 1]
    platt_b = calibration_params[:, 2]
    calibrated_probs = torch.sigmoid(platt_a * calibrated_logits + platt_b)

    # Uncertainty estimation
    uncertainty_estimate = F.softplus(calibration_params[:, 3])

    result = {
        'calibrated_confidence': calibrated_probs,
        'uncertainty_estimate': uncertainty_estimate,
        'calibrated_logits': calibrated_logits
    }

    if return_temperature:
        result['temperature'] = temperature

    return result
```

### Temperature Scaling Integration

Temperature scaling is a simple but effective post-hoc calibration method integrated into the neural approach:

```python
class TemperatureScaling(nn.Module):
    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Scale logits by learned temperature
        return logits / self.temperature

    def get_temperature(self) -> float:
        return float(self.temperature.item())
```

However, the Calibration Head learns context-dependent temperatures rather than a single global temperature:

```python
def _adaptive_temperature_scaling(
    self,
    logits: torch.Tensor,
    context_features: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # Learn context-dependent temperature
    temp_features = self.temperature_network(context_features)
    temperature = F.softplus(temp_features) + 0.1  # Minimum temperature

    # Apply scaling
    scaled_logits = logits / temperature.unsqueeze(1)

    return scaled_logits, temperature
```

### Platt Scaling Enhancement

Platt scaling learns a sigmoid transformation to improve calibration:

```python
def _platt_scaling(
    self,
    scaled_logits: torch.Tensor,
    context_features: torch.Tensor
) -> torch.Tensor:

    # Learn context-dependent Platt parameters
    platt_features = self.platt_network(context_features)
    platt_a = platt_features[:, 0]  # Slope parameter
    platt_b = platt_features[:, 1]  # Intercept parameter

    # Apply Platt transformation
    calibrated_probs = torch.sigmoid(
        platt_a.unsqueeze(1) * scaled_logits + platt_b.unsqueeze(1)
    )

    return calibrated_probs
```

### Uncertainty Quantification

The Calibration Head distinguishes between different types of uncertainty:

#### Epistemic Uncertainty (Model Uncertainty)

```python
def _estimate_epistemic_uncertainty(
    self,
    model_embeddings: torch.Tensor,
    ensemble_variance: torch.Tensor
) -> torch.Tensor:

    # Estimate what the model doesn't know
    epistemic_features = torch.cat([model_embeddings, ensemble_variance], dim=1)
    epistemic_uncertainty = self.epistemic_network(epistemic_features)

    return F.softplus(epistemic_uncertainty)
```

#### Aleatoric Uncertainty (Data Uncertainty)

```python
def _estimate_aleatoric_uncertainty(
    self,
    data_quality_features: torch.Tensor,
    context_noise_indicators: torch.Tensor
) -> torch.Tensor:

    # Estimate inherent data uncertainty
    aleatoric_features = torch.cat([
        data_quality_features,
        context_noise_indicators
    ], dim=1)

    aleatoric_uncertainty = self.aleatoric_network(aleatoric_features)

    return F.softplus(aleatoric_uncertainty)
```

#### Total Uncertainty Composition

```python
def _compose_total_uncertainty(
    self,
    epistemic_uncertainty: torch.Tensor,
    aleatoric_uncertainty: torch.Tensor
) -> torch.Tensor:

    # Combine uncertainties (assuming independence)
    total_uncertainty = torch.sqrt(
        epistemic_uncertainty**2 + aleatoric_uncertainty**2
    )

    return total_uncertainty
```

## Training Process

### Calibration Dataset Construction

Training requires carefully constructed datasets with ground truth reliability labels:

#### Confidence Ground Truth Generation

```python
def generate_confidence_labels(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    confidence_bins: int = 10
) -> np.ndarray:
    """
    Generate ground truth confidence labels based on prediction accuracy.
    """

    # Calculate prediction errors
    errors = np.abs(predictions - ground_truth)
    error_threshold = np.percentile(errors, 80)  # 80th percentile threshold

    # Binary reliability labels
    reliable_predictions = errors <= error_threshold

    # Bin predictions by confidence and calculate empirical accuracy
    bin_boundaries = np.linspace(0, 1, confidence_bins + 1)
    confidence_labels = np.zeros_like(predictions)

    for i in range(confidence_bins):
        bin_mask = (
            (predictions >= bin_boundaries[i]) &
            (predictions < bin_boundaries[i + 1])
        )

        if np.sum(bin_mask) > 0:
            # Empirical accuracy in this bin
            bin_accuracy = np.mean(reliable_predictions[bin_mask])
            confidence_labels[bin_mask] = bin_accuracy

    return confidence_labels, reliable_predictions
```

#### Multi-Task Training Data

```python
def prepare_calibration_training_data(
    base_model_predictions: pd.DataFrame,
    ground_truth_interactions: pd.DataFrame,
    context_features: pd.DataFrame
) -> Dict[str, torch.Tensor]:

    # Merge predictions with ground truth
    merged_data = base_model_predictions.merge(
        ground_truth_interactions, on=['user_id', 'track_id']
    ).merge(
        context_features, on=['user_id', 'track_id']
    )

    # Generate confidence labels
    cf_confidence_labels, cf_reliable = generate_confidence_labels(
        merged_data['cf_prediction'].values,
        merged_data['user_rating'].values
    )

    content_confidence_labels, content_reliable = generate_confidence_labels(
        merged_data['content_prediction'].values,
        merged_data['user_rating'].values
    )

    # Prepare training tensors
    training_data = {
        'cf_predictions': torch.FloatTensor(merged_data['cf_prediction'].values),
        'content_predictions': torch.FloatTensor(merged_data['content_prediction'].values),
        'cf_confidence_labels': torch.FloatTensor(cf_confidence_labels),
        'content_confidence_labels': torch.FloatTensor(content_confidence_labels),
        'cf_reliable_labels': torch.BoolTensor(cf_reliable),
        'content_reliable_labels': torch.BoolTensor(content_reliable),
        'context_features': torch.FloatTensor(
            merged_data[context_feature_columns].values
        )
    }

    return training_data
```

### Loss Functions

#### Calibration Loss

The primary loss combines multiple calibration objectives:

```python
def calibration_loss(
    calibrated_confidences: torch.Tensor,
    ground_truth_confidences: torch.Tensor,
    reliable_labels: torch.Tensor,
    uncertainty_estimates: torch.Tensor,
    ground_truth_uncertainties: torch.Tensor,
    loss_weights: Dict[str, float]
) -> torch.Tensor:

    # Expected Calibration Error approximation
    ece_loss = expected_calibration_error_loss(
        calibrated_confidences, reliable_labels
    )

    # Confidence regression loss
    confidence_mse_loss = F.mse_loss(
        calibrated_confidences, ground_truth_confidences
    )

    # Uncertainty estimation loss
    uncertainty_loss = F.mse_loss(
        uncertainty_estimates, ground_truth_uncertainties
    )

    # Reliability classification loss
    reliability_bce_loss = F.binary_cross_entropy(
        calibrated_confidences, reliable_labels.float()
    )

    # Combine losses
    total_loss = (
        loss_weights['ece'] * ece_loss +
        loss_weights['confidence_mse'] * confidence_mse_loss +
        loss_weights['uncertainty'] * uncertainty_loss +
        loss_weights['reliability_bce'] * reliability_bce_loss
    )

    return total_loss
```

#### Expected Calibration Error Loss

```python
def expected_calibration_error_loss(
    confidences: torch.Tensor,
    correct_predictions: torch.Tensor,
    num_bins: int = 10
) -> torch.Tensor:
    """
    Differentiable approximation of Expected Calibration Error.
    """

    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=confidences.device)
    ece = torch.tensor(0.0, device=confidences.device)

    for i in range(num_bins):
        # Soft bin assignment using temperature
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Soft membership function (differentiable)
        bin_membership = torch.sigmoid(
            10 * (confidences - bin_lower)
        ) * torch.sigmoid(
            10 * (bin_upper - confidences)
        )

        # Weighted accuracy and confidence in this bin
        bin_weight = torch.sum(bin_membership)

        if bin_weight > 1e-8:  # Avoid division by zero
            bin_accuracy = torch.sum(
                bin_membership * correct_predictions.float()
            ) / bin_weight

            bin_confidence = torch.sum(
                bin_membership * confidences
            ) / bin_weight

            # Bin contribution to ECE
            bin_ece = torch.abs(bin_accuracy - bin_confidence) * bin_weight
            ece += bin_ece

    # Normalize by total number of samples
    ece /= confidences.size(0)

    return ece
```

### Training Configuration

```python
{
    "calibration_training": {
        "learning_rate": 0.0005,
        "batch_size": 512,
        "epochs": 200,
        "optimizer": "AdamW",
        "weight_decay": 0.001,
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "patience": 10,
            "factor": 0.5
        }
    },
    "loss_weights": {
        "ece": 1.0,
        "confidence_mse": 0.5,
        "uncertainty": 0.3,
        "reliability_bce": 0.7
    },
    "calibration_parameters": {
        "num_bins": 15,
        "temperature_range": [0.1, 5.0],
        "platt_regularization": 0.01
    }
}
```

### Training Loop

```python
def train_calibration_head(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    device
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )

    best_ece = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            optimizer.zero_grad()

            # Forward pass
            calibration_output = model(
                batch['predictions'].to(device),
                batch['context_features'].to(device)
            )

            # Calculate loss
            loss = calibration_loss(
                calibrated_confidences=calibration_output['calibrated_confidence'],
                ground_truth_confidences=batch['confidence_labels'].to(device),
                reliable_labels=batch['reliable_labels'].to(device),
                uncertainty_estimates=calibration_output['uncertainty_estimate'],
                ground_truth_uncertainties=batch['uncertainty_labels'].to(device),
                loss_weights=config['loss_weights']
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        val_loss, val_ece = evaluate_calibration(model, val_dataloader, device)

        # Learning rate scheduling
        scheduler.step(val_ece)

        # Save best model
        if val_ece < best_ece:
            best_ece = val_ece
            torch.save(model.state_dict(), 'best_calibration_head.pth')

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val ECE: {val_ece:.4f}')
```

## Evaluation and Validation

### Calibration Metrics

#### Expected Calibration Error (ECE)

```python
def calculate_expected_calibration_error(
    confidences: np.ndarray,
    correct_predictions: np.ndarray,
    num_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error.
    """

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find predictions in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct_predictions[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
```

#### Maximum Calibration Error (MCE)

```python
def calculate_maximum_calibration_error(
    confidences: np.ndarray,
    correct_predictions: np.ndarray,
    num_bins: int = 10
) -> float:
    """
    Calculate Maximum Calibration Error.
    """

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    mce = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(correct_predictions[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            mce = max(mce, bin_error)

    return mce
```

#### Reliability Diagram Analysis

```python
def generate_reliability_diagram_data(
    confidences: np.ndarray,
    correct_predictions: np.ndarray,
    num_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Generate data for reliability diagram visualization.
    """

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    accuracies = []
    avg_confidences = []
    bin_counts = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.sum(in_bin) > 0:
            accuracies.append(np.mean(correct_predictions[in_bin]))
            avg_confidences.append(np.mean(confidences[in_bin]))
            bin_counts.append(np.sum(in_bin))
        else:
            accuracies.append(0.0)
            avg_confidences.append(bin_centers[i])
            bin_counts.append(0)

    return {
        'bin_centers': bin_centers,
        'accuracies': np.array(accuracies),
        'avg_confidences': np.array(avg_confidences),
        'bin_counts': np.array(bin_counts)
    }
```

### Comprehensive Evaluation Framework

```python
def evaluate_calibration_head(
    model,
    test_dataloader,
    device,
    save_plots: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of calibration head performance.
    """

    model.eval()
    all_predictions = []
    all_confidences = []
    all_ground_truth = []
    all_uncertainties = []

    with torch.no_grad():
        for batch in test_dataloader:
            outputs = model(
                batch['predictions'].to(device),
                batch['context_features'].to(device)
            )

            all_predictions.extend(batch['predictions'].cpu().numpy())
            all_confidences.extend(outputs['calibrated_confidence'].cpu().numpy())
            all_ground_truth.extend(batch['ground_truth'].cpu().numpy())
            all_uncertainties.extend(outputs['uncertainty_estimate'].cpu().numpy())

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    confidences = np.array(all_confidences)
    ground_truth = np.array(all_ground_truth)
    uncertainties = np.array(all_uncertainties)

    # Calculate correctness (binary classification)
    errors = np.abs(predictions - ground_truth)
    error_threshold = np.percentile(errors, 80)
    correct_predictions = errors <= error_threshold

    # Calibration metrics
    metrics = {}
    metrics['ece'] = calculate_expected_calibration_error(confidences, correct_predictions)
    metrics['mce'] = calculate_maximum_calibration_error(confidences, correct_predictions)

    # Confidence-accuracy correlation
    metrics['confidence_accuracy_correlation'] = np.corrcoef(
        confidences, correct_predictions.astype(float)
    )[0, 1]

    # Uncertainty-error correlation
    metrics['uncertainty_error_correlation'] = np.corrcoef(
        uncertainties, errors
    )[0, 1]

    # Brier score
    metrics['brier_score'] = np.mean((confidences - correct_predictions.astype(float))**2)

    # Area under the calibration curve
    rel_data = generate_reliability_diagram_data(confidences, correct_predictions)
    perfect_calibration = rel_data['bin_centers']
    actual_calibration = rel_data['accuracies']
    metrics['calibration_auc'] = np.trapz(actual_calibration, perfect_calibration)

    if save_plots:
        plot_reliability_diagram(rel_data)
        plot_confidence_histogram(confidences, correct_predictions)

    return metrics
```

## Production Integration

### Real-time Calibration

```python
def calibrate_prediction_confidence(
    calibration_head,
    raw_prediction: float,
    model_embedding: np.ndarray,
    context_features: np.ndarray,
    device: torch.device
) -> Dict[str, float]:
    """
    Apply learned calibration to a single prediction.
    """

    calibration_head.eval()

    # Prepare inputs
    prediction_tensor = torch.tensor([raw_prediction], device=device)
    embedding_tensor = torch.tensor(model_embedding, device=device).unsqueeze(0)
    context_tensor = torch.tensor(context_features, device=device).unsqueeze(0)

    with torch.no_grad():
        calibration_output = calibration_head(
            prediction_tensor,
            torch.cat([embedding_tensor, context_tensor], dim=1)
        )

    return {
        'calibrated_confidence': float(calibration_output['calibrated_confidence'][0]),
        'uncertainty_estimate': float(calibration_output['uncertainty_estimate'][0]),
        'raw_prediction': raw_prediction
    }
```

### Batch Calibration Pipeline

```python
def calibrate_batch_predictions(
    calibration_head,
    predictions: np.ndarray,
    embeddings: np.ndarray,
    context_features: np.ndarray,
    batch_size: int = 1000,
    device: torch.device = None
) -> Dict[str, np.ndarray]:
    """
    Calibrate a large batch of predictions efficiently.
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    calibration_head.eval()
    calibrated_confidences = []
    uncertainty_estimates = []

    num_batches = (len(predictions) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(predictions))

            # Prepare batch tensors
            batch_predictions = torch.tensor(
                predictions[start_idx:end_idx], device=device
            )
            batch_embeddings = torch.tensor(
                embeddings[start_idx:end_idx], device=device
            )
            batch_context = torch.tensor(
                context_features[start_idx:end_idx], device=device
            )

            # Calibrate batch
            batch_output = calibration_head(
                batch_predictions,
                torch.cat([batch_embeddings, batch_context], dim=1)
            )

            calibrated_confidences.extend(
                batch_output['calibrated_confidence'].cpu().numpy()
            )
            uncertainty_estimates.extend(
                batch_output['uncertainty_estimate'].cpu().numpy()
            )

    return {
        'calibrated_confidences': np.array(calibrated_confidences),
        'uncertainty_estimates': np.array(uncertainty_estimates),
        'original_predictions': predictions
    }
```

### Integration with Hybrid Model

The Calibration Head integrates seamlessly with the hybrid recommendation system:

```python
# In HybridModel.predict()
def apply_learned_calibration(
    self,
    cf_score: float,
    content_score: float,
    cf_embedding: torch.Tensor,
    content_embedding: torch.Tensor,
    context_features: Dict[str, Any]
) -> Tuple[float, float]:

    if hasattr(self, 'calibration_head') and self.calibration_head is not None:
        # Prepare context for calibration
        calibration_context = self._prepare_calibration_context(
            context_features, cf_embedding, content_embedding
        )

        # Calibrate CF confidence
        cf_calibration = self.calibration_head.calibrate_prediction_confidence(
            raw_prediction=cf_score,
            model_embedding=cf_embedding.cpu().numpy(),
            context_features=calibration_context
        )

        # Calibrate content confidence
        content_calibration = self.calibration_head.calibrate_prediction_confidence(
            raw_prediction=content_score,
            model_embedding=content_embedding.cpu().numpy(),
            context_features=calibration_context
        )

        return (
            cf_calibration['calibrated_confidence'],
            content_calibration['calibrated_confidence']
        )
    else:
        # Fallback to heuristic confidence
        return self._calculate_heuristic_confidence(cf_score, content_score, context_features)
```

## Advanced Applications

### Adaptive Recalibration

The Calibration Head supports continuous improvement based on user feedback:

```python
class AdaptiveCalibrationHead(CalibrationHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_buffer = []
        self.adaptation_frequency = 1000  # Recalibrate every N predictions

    def update_with_feedback(
        self,
        prediction: float,
        calibrated_confidence: float,
        actual_outcome: float,
        context_features: np.ndarray
    ):
        """
        Update calibration based on observed outcomes.
        """

        # Store feedback for batch adaptation
        self.adaptation_buffer.append({
            'prediction': prediction,
            'calibrated_confidence': calibrated_confidence,
            'actual_outcome': actual_outcome,
            'context_features': context_features,
            'prediction_error': abs(prediction - actual_outcome)
        })

        # Trigger adaptation if buffer is full
        if len(self.adaptation_buffer) >= self.adaptation_frequency:
            self._perform_online_adaptation()
            self.adaptation_buffer = []

    def _perform_online_adaptation(self):
        """
        Perform online recalibration using recent feedback.
        """

        # Convert buffer to training format
        adaptation_data = self._prepare_adaptation_data(self.adaptation_buffer)

        # Quick recalibration with low learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        for _ in range(5):  # Few adaptation steps
            optimizer.zero_grad()

            outputs = self(
                adaptation_data['predictions'],
                adaptation_data['context_features']
            )

            # Simplified adaptation loss
            loss = F.mse_loss(
                outputs['calibrated_confidence'],
                adaptation_data['empirical_reliability']
            )

            loss.backward()
            optimizer.step()
```

### Ensemble Calibration

Multiple calibration heads can be combined for improved robustness:

```python
class EnsembleCalibrationHead(nn.Module):
    def __init__(self, individual_heads: List[CalibrationHead]):
        super().__init__()
        self.heads = nn.ModuleList(individual_heads)
        self.ensemble_weights = nn.Parameter(
            torch.ones(len(individual_heads)) / len(individual_heads)
        )

    def forward(self, *args, **kwargs):
        # Get predictions from all heads
        head_outputs = [head(*args, **kwargs) for head in self.heads]

        # Weighted ensemble combination
        weights = F.softmax(self.ensemble_weights, dim=0)

        ensemble_confidence = torch.sum(torch.stack([
            output['calibrated_confidence'] * weight
            for output, weight in zip(head_outputs, weights)
        ]), dim=0)

        ensemble_uncertainty = torch.sqrt(torch.sum(torch.stack([
            (output['uncertainty_estimate'] * weight)**2
            for output, weight in zip(head_outputs, weights)
        ]), dim=0))

        return {
            'calibrated_confidence': ensemble_confidence,
            'uncertainty_estimate': ensemble_uncertainty,
            'individual_outputs': head_outputs
        }
```

The Calibration Head transforms heuristic confidence estimates into principled, learned confidence scores that accurately reflect prediction reliability, enabling more trustworthy and effective recommendation systems through proper uncertainty quantification and calibration.





