# Handwritten Digit Classification: FNN vs. CNN on MNIST

> Comparing Fully Connected and Convolutional Neural Networks for image classification - with noise robustness testing.

---

## Overview

This project implements and compares two deep learning architectures - a **Fully Connected Neural Network (FNN)** and a **Convolutional Neural Network (CNN)** - for classifying handwritten digits using the MNIST dataset. Beyond standard evaluation, the models are stress-tested against **impulsive (salt-and-pepper) noise** to analyze real-world robustness.

---

## Results Summary

| Model | Clean Accuracy (Subset) | Noisy Accuracy (Subset) | Accuracy Drop |
|-------|------------------------|------------------------|---------------|
| FNN   | 95.15%                 | 90.30%                 | **4.85%**     |
| CNN   | 98.90%                 | 88.85%                 | **10.05%**    |
| FNN (Full Test Set) | 96.27% | - | - |
| CNN (Full Test Set) | **99.21%** | - | - |

**Key insight:** The CNN achieves higher baseline accuracy by preserving spatial structure, but the FNN proves more resilient to pixel-level noise distortions.

---

## Model Architectures

### Fully Connected Neural Network (FNN)
```
Input (784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
```
- Images flattened from 28×28 → 784-dimensional vectors
- Optimizer: Adam | Loss: Sparse Categorical Crossentropy

### Convolutional Neural Network (CNN)
```
Input (28×28×1)
  → Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
  → Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
  → Flatten
  → Dense(64, ReLU)
  → Dense(10, Softmax)
```
- Optimizer: Adam | Loss: Sparse Categorical Crossentropy

---

## Hyperparameter Tuning

Both models were tuned using **Scikit-learn's `GridSearchCV`** with 3-fold cross-validation on a 5,000-sample subset.

**FNN search space:**
- `neurons_layer1`: [64, 128]
- `neurons_layer2`: [32, 64]
- `optimizer`: [adam, sgd]
- `batch_size`: [32, 64]

**CNN search space:**
- `filters_1`: [16, 32]
- `dense_units`: [32, 64]
- `optimizer`: [adam, rmsprop]
- `batch_size`: [32, 64]

Final models were retrained for **10 epochs** on the full training set using the best found parameters.

---

## Noise Robustness Testing

A custom **impulsive noise function** randomly flips a proportion of pixels to either white (1.0) or black (0.0), simulating real-world sensor noise.

```python
def add_impulsive_noise(image, noise_factor=0.1):
    noisy_image = image.copy()
    white_pixels = np.random.rand(*image.shape) < (noise_factor / 2)
    noisy_image[white_pixels] = 1
    black_pixels = np.random.rand(*image.shape) < (noise_factor / 2)
    noisy_image[black_pixels] = 0
    return noisy_image
```

Models were evaluated at noise levels: **10%, 20%, 30%, 40%, 50%** on a 2,000-image subset.

---

## Project Structure

```
├── FNN.ipynb   # FNN implementation
├── CNN.ipynb   # CNN implementation
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| TensorFlow / Keras | Model building & training |
| Scikit-learn | GridSearchCV hyperparameter tuning |
| SciKeras | Keras–Scikit-learn wrapper |
| NumPy | Data manipulation & noise injection |


---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/your-username/mnist-fnn-vs-cnn.git
cd mnist-fnn-vs-cnn
```

**2. Install dependencies**
```bash
pip install tensorflow scikit-learn scikeras numpy matplotlib
```

**3. Run the notebooks**

Open either notebook in Jupyter or VS Code and run all cells. The MNIST dataset downloads automatically via `keras.datasets.mnist`.

```bash
jupyter notebook CNN.ipynb
```

---

## Conclusion

The CNN outperforms the FNN on clean data due to its ability to extract local spatial features like edges and curves through convolutional layers. However, because the FNN processes images as flat vectors rather than structured feature maps, it shows greater resilience to pixel-level noise - dropping only 4.85% in accuracy versus the CNN's 10.05% drop under identical noise conditions.

This tradeoff highlights an important principle: **architectural strengths in one domain can become vulnerabilities in another.**

