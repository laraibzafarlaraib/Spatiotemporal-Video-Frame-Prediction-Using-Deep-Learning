# Spatiotemporal Video Frame Prediction Using Deep Learning
### Comparative Study: ConvLSTM vs. PredRNN vs. Spatial-Temporal Vision Transformer on UCF101

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCF101-green.svg)](https://www.crcv.ucf.edu/data/UCF101.php)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-red.svg)](https://gradio.app/)

---

##  Overview

Predicting subsequent frames in real-world video requires learning both **fine spatial appearance** (shapes, textures) and **temporal motion dynamics** (direction, acceleration). 

This project explores spatiotemporal predictive learning by training and benchmarking three distinct deep neural network architectures on human action video sequences from the **UCF101** dataset:

1. **ConvLSTM2D**: Combines spatial convolutions with recurrence to maintain 2D feature representations across time.
2. **PredRNN-style Spatiotemporal RNN**: Decouples and enriches spatiotemporal memory flows for better long-term sequence coherence.
3. **CNN-Transformer**: Uses a 2D-CNN feature extractor coupled with Multi-Head Self-Attention to capture non-local temporal relationships.

The pipeline takes **10 sequential frames** ($64 \times 64 \times 3$) as input and predicts the next **10 consecutive frames** into the future. Evaluated using pixel-wise metric (**MSE**) and perceptual structure fidelity (**SSIM**), alongside an interactive **Gradio web dashboard** and **MoviePy video stitching**.

---

## Architecture Comparison

```
Input Video Sequence: 10 Frames [B, 10, 64, 64, 3]
              │
              ├───► ConvLSTM2D (3x ConvLSTM Layers + BatchNorm + Conv3D Decoder)
              │
              ├───► PredRNN (Multi-layer Spatiotemporal Memory Units + Conv3D)
              │
              └───► CNN-Transformer (TimeDistributed CNN + Multi-Head Self-Attention + Dense Decoder)
              │
              ▼
Future Predictions: 10 Frames [B, 10, 64, 64, 3]
              │
              ▼
Metrics (MSE & SSIM) + MoviePy Video Rendering + Gradio UI
```

---

## Results

Evaluated across test sequences under identical experimental conditions:

| Architecture | MSE (Lower is better) | SSIM (Higher is better) | Strengths & Observations |
| :--- | :---: | :---: | :--- |
| **PredRNN**  | **`0.0068`** | **`0.7093`** | **Best overall performer.** Retains edge clarity and spatial detail over extended timesteps without excessive blurring. |
| **ConvLSTM** | `0.0070` | `0.6913` | Strong spatiotemporal baseline with stable gradients; slight temporal smoothing across later frames. |
| **CNN-Transformer** | `0.0126` | `0.4945` | Fast attention-based convergence, but standard dense decoders suffer from spatial reconstruction loss on frame synthesis. |

> *Quantitative test comparisons computed with scikit-image `structural_similarity` (SSIM) across RGB channels and Mean Squared Error (MSE).*

---

## Tech Stack & Libraries

- **Deep Learning Framework**: TensorFlow / Keras (ConvLSTM2D, TimeDistributed, MultiHeadAttention, Conv3D)
- **Computer Vision**: OpenCV (`cv2`) for frame decimation, scaling, and normalization
- **Image Metrics**: Scikit-Image (`skimage.metrics.structural_similarity`, `mse`)
- **Video Stitching & Encoding**: MoviePy (`ImageSequenceClip` with `libx264`)
- **Interactive UI**: Gradio (`gr.Blocks`, `gr.Video`, `gr.Slider`)
- **Data & Plotting**: NumPy, Matplotlib, Pandas, Tqdm

---


## Getting Started

### 1. Prerequisites
Clone the repository and install requirements:
```bash
git clone https://github.com/YOUR_USERNAME/video-prediction-deep-learning.git
cd video-prediction-deep-learning
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the **UCF101 Action Recognition** dataset (or subsets via Kaggle):
- Target Action Classes: `WalkingWithDog`, `JumpingJack`, `Biking`, `HorseRiding`, `Skijet`
- Preprocessing standardizes all video sequences to $64 \times 64$ RGB at normalized `[0, 1]` floats.

### 3. Training & Inference
Run the notebook or launch the interactive Gradio demo:
```python
# To launch the Gradio interactive comparator:
python -c "import gradio as gr; ..." # Or execute Step 8 inside the notebook
```

---

## Key Findings & Learnings

1. **Spatial Inductive Bias Matters**: While Transformers excel at global attention, autoregressive spatial frame generation requires localized inductive biases. Convolutional recurrence (ConvLSTM & PredRNN) consistently outperformed vanilla attention on pixel synthesis.
2. **SSIM vs. MSE Discrepancy**: MSE can reward blurry average predictions because blur minimizes Euclidean distance across moving edges. SSIM proved to be a far better indicator of perceptual visual quality.
3. **Decoupled Memory**: PredRNN's dual memory mechanism better prevented information decay when projecting out multiple consecutive timesteps.

---

## Author

- **Laraib Zafar**
- [LinkedIn](https://www.linkedin.com) • [GitHub](https://github.com)
