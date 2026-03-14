# Spectral Graffiti: Few-Shot Audio In-Painting via Amortized Inference

## Project Overview

This project addresses a unique and challenging audio restoration problem: **few-shot audio in-painting** on degraded master tapes of *The New Yardbirds'* album, *Spectral Graffiti*. Due to catastrophic physical degradation (Sticky Shed Syndrome), the master tapes have microscopic gaps (up to 100ms) where the audio signal is entirely missing. This isn't a standard noise reduction task, but a problem of inferring and reconstructing entirely absent waveforms based on surrounding context.

Our solution leverages a novel deep learning approach that performs **amortized inference**, meaning the model learns a universal reconstruction strategy during training and can then predict missing audio in a single forward pass during inference, without requiring per-sample optimization.

## 1. The Problem: Missing Audio Data

Imagine trying to hear a song where parts are completely silent. That's the challenge here. The audio data isn't just noisy; it's gone. We need to look at the surviving fragments of a 100ms audio sample and mathematically infer the underlying spectral signature of the instrument to perfectly reconstruct the missing parts.

## 2. The Solution: Amortized Inference

Audio is incredibly complex and non-stationary. A single 100ms clip could contain anything from a sustained bass note to a sudden cymbal crash. Training a model that can handle this diversity without individual tuning for each missing segment is key.

Our approach shifts the heavy computational burden to the training phase. By learning the universal physical properties of audio across 80,000 diverse samples, our model can take a fragmented audio clip and predict all missing gaps in a **single, efficient forward pass**. This eliminates slow, iterative per-sample optimization loops during inference, making it incredibly fast and scalable.

## 3. The Architecture: Masked Transformer Encoder

To capture the intricate temporal dependencies within audio, we designed a **Masked Transformer Encoder**.

### Why a Transformer?

Traditional sequential models might struggle with non-local dependencies. A Transformer Encoder is ideal because:

*   **Self-attention:** Allows each missing point to directly 'look' at *all* available context points, regardless of their position in time. This is crucial for reconstructing complex patterns.
*   **Variable context density:** It naturally handles situations where some clips have sparse context and others have abundant context.
*   **Generalization:** It learns *how to interpret context* to infer missing audio, rather than memorizing specific instrument sounds, enabling generalization across different instruments and sounds.

### Input Representation

For each of the 100 time steps in an audio sample, the model receives **two features**:

1.  `masked_voltage`: The true voltage value if the point is known (`Is_Context=1`), otherwise `0.0` (indicating a missing value).
2.  `Is_Context` (mask): A binary flag (`1` for known, `0` for missing). This explicitly tells the model which points are reliable and which need reconstruction.

This dual input allows the model to differentiate between a genuinely silent part of the audio and a missing data point.

### Architecture Summary

```
Input [batch, 100, 2] (Masked Value, Is_Context Mask)
  -> Linear Projection (maps to higher dimension: e.g., 64)
  -> + Learnable Positional Encoding (injects time-awareness)
  -> Transformer Encoder (3 layers, 4 attention heads, GELU activation)
  -> Linear Projection (maps back to a single voltage value)
  -> squeeze (removes the last dimension, resulting in [batch, 100] predictions)
```

## 4. The Loss Function: Custom Masked MSE Loss

Using a standard Mean Squared Error (MSE) across the entire 100ms sequence would be problematic. It would force the model to waste effort perfectly reconstructing context points it already knows, diluting the learning signal for the truly missing parts.

To focus the model exclusively on learning to predict the **hidden gaps**, we implemented a custom `masked_mse_loss`:

$$\mathcal{L}_{masked}=\frac{\sum_{i=1}^{N}(\hat{y}_i-y_i)^2\cdot(1-m_i)}{\sum_{i=1}^{N}(1-m_i)+\epsilon}$$

Here:
*   $\hat{y}_i$ is the predicted value.
*   $y_i$ is the true value.
*   $m_i$ is the `Is_Context` mask (1 for known, 0 for missing).

By multiplying the squared error by $(1 - m_i)$, we effectively zero out the error for known context points. The model's weights are updated *only* based on its performance in predicting the missing sections, ensuring efficient and targeted learning.

## 5. Dataset Design

The `SpectralDataset` class processes the raw CSV data. It groups the flattened audio samples by `Sample_ID` and converts each 100ms clip into:

*   `x`: The 2-feature input tensor for the Transformer (shape `[100, 2]`).
*   `y_true`: The full ground-truth voltage sequence (shape `[100]`).
*   `mask`: The `Is_Context` flags (shape `[100]`).

Each `Sample_ID` represents an independent audio clip, preventing information leakage between samples.

## 6. Training Strategy

### Hyperparameters

*   **Model Dimension (`d_model`):** 64 (balanced for expressiveness and efficiency)
*   **Attention Heads (`nhead`):** 4 (captures multiple patterns simultaneously)
*   **Transformer Layers (`num_layers`):** 3 (sufficient depth for complex signals)
*   **Optimizer:** `AdamW` (known for good generalization in Transformers)
*   **Learning Rate:** `1e-3` (standard starting point, with decay)
*   **Batch Size:** 32 (fits within GPU memory)
*   **Epochs:** 10 (observed good convergence within this range)

### Gradient Clipping

To prevent issues common with Transformer training on continuous data, gradients are clipped at a maximum norm of `1.0`.

### Preventing Memorization (Generalization)

To ensure the model learns generalized audio physics rather than just memorizing training patterns, a robust validation strategy was employed:

*   **Validation Split:** 20% of the dataset was held out from training.
*   **Early Stopping & Scheduling:** An `AdamW` optimizer with weight decay was used alongside a `ReduceLROnPlateau` scheduler. This monitors validation loss and reduces the learning rate when improvements plateau, helping to find the optimal point of generalization.

Tracking both training and validation MSE ensures that training stops when the model achieves the best balance between learning from the data and generalizing to unseen examples.

## 7. Inference and Output

After training, the model is put into evaluation (`model.eval()`) mode. A `DataLoader` processes the entire dataset (including both training and validation splits for a full reconstruction), and predictions are generated in batches using `torch.no_grad()` for efficiency.

The `Predicted_Value` for each missing point (`Is_Context == 0`) is then clipped to the original data's min/max range to ensure realistic output and saved into a `submission.csv` file. This file contains the `Sample_ID`, `Time_ms`, and the `Predicted_Value` for all in-painted audio segments.
