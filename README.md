# MLP on MNIST — From Scratch

Implementation of a Multilayer Perceptron (MLP) trained on the MNIST dataset using only NumPy — no ML libraries (no TensorFlow, no PyTorch, no Scikit-learn).

**Authors:** Tchuenche Kamdem Samuel Yedidya, Mayagi Paul Alain, Nkot Jean Francky  
**Course:** INF372 — Machine Learning, Université de Yaoundé I, L3 Informatique  
**Year:** 2024–2025

---

## What this project does

Trains a neural network to recognize handwritten digits (0–9) from the MNIST dataset, implementing every component manually from mathematical first principles.

**Final Results:**
- Train accuracy: **98.80%**
- Test accuracy: **93.25%**
- Training: 100 epochs, SGD, batch size 32
- Training time: ~3-5 minutes

---

## Architecture

| Layer  | Size          | Activation | Parameters |
|--------|---------------|------------|-----------|
| Input  | 1024 (32×32)  | —          | —         |
| Hidden | 64 neurons    | ReLU       | 65,600    |
| Output | 10 neurons    | Softmax    | 650       |
| **Total** | — | — | **66,250** |

---

## What is implemented from scratch

- ✓ Forward propagation
- ✓ Backpropagation with chain rule
- ✓ Stochastic Gradient Descent (SGD)
- ✓ ReLU activation function
- ✓ Softmax activation function
- ✓ Cross-entropy loss computation
- ✓ Weight initialization (Xavier/Glorot)
- ✓ Weight heatmap visualization
- ✓ Hidden layer activation analysis per class
- ✓ Confusion matrix generation

---

## Key findings

- SGD converges stably over 100 epochs with decreasing loss
- Weight heatmaps show neurons progressively ignoring irrelevant pixels as training progresses
- Similar digit classes (1/7, 3/8, 4/9) share active neurons — explaining systematic confusions
- When the model makes an error, the activation profile of the hidden layer resembles the predicted class more than the true class — the network does not fail randomly
- Regularization (L2) prevents overfitting; model generalizes well to unseen test data

---

## Repository Structure

```
mlp-mnist-from-scratch/
├── TP_perceptron_multicouche_MNIST_32x32.ipynb   # Full implementation and analysis
├── slides_TP4_MLP.pdf                            # Presentation slides
├── requirements.txt                              # Python dependencies
├── README.md                                     # This file
├── LICENSE                                       # MIT License
└── data/                                         # Dataset directory (create as needed)
    ├── train/
    │   └── images/ (PNG files)
    └── test/
        └── images/ (PNG files)
```

---

## Dataset Details

### Source
This project uses the MNIST dataset: 70,000 handwritten digit images (28×28 pixels, resized to 32×32).

- **Training samples:** 60,000
- **Test samples:** 10,000
- **Classes:** 10 (digits 0-9)
- **Format:** Grayscale PNG arrays

### Important Note
The dataset is **not included** in this repository due to size constraints (~400 MB for full dataset).

### Dataset Download Instructions

#### Option 1: Official MNIST Source
Download from: http://yann.lecun.com/exdb/mnist/

Provides binary format files (train-images-idx3-ubyte, train-labels-idx1-ubyte, etc.)

#### Option 2: Python via Scikit-Learn (Recommended)
```python
from sklearn.datasets import fetch_openml
import numpy as np

# Download MNIST
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target
X = X.astype(np.uint8)
y = y.astype(np.int64)

# X shape: (70000, 784)
# y shape: (70000,)
```

#### Option 3: TensorFlow/Keras
```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train shape: (60000, 28, 28)
# y_train shape: (60000,)
```

### Data Preprocessing
Images in this project:
1. **Resized** from 28×28 to 32×32 pixels (using PIL or similar)
2. **Normalized** to [0, 1] range (divide by 255)
3. **Flattened** to 1,024-dimensional vectors for input layer
4. **Saved as PNG files** with labels in filename (e.g., `train_00000_label_5.png`)

#### Example preprocessing script:
```python
import numpy as np
from PIL import Image
import os

def prepare_mnist_data(X, y, output_dir, split='train'):
    """Convert and save MNIST data as PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (image, label) in enumerate(zip(X, y)):
        # Reshape from 784 to 28x28
        img_array = image.reshape(28, 28) if image.shape == (784,) else image
        
        # Resize to 32x32
        img = Image.fromarray(img_array.astype('uint8'))
        img = img.resize((32, 32))
        
        # Save with label in filename
        filename = f"{output_dir}/{split}_{i:05d}_label_{int(label)}.png"
        img.save(filename)

# Usage
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Split into train/test
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

prepare_mnist_data(X_train, y_train, 'data/train', 'train')
prepare_mnist_data(X_test, y_test, 'data/test', 'test')
```

### Expected Directory Structure
```
data/
├── train/
│   ├── train_00000_label_5.png
│   ├── train_00001_label_0.png
│   ├── train_00002_label_4.png
│   └── ... (60,000 files total)
└── test/
    ├── test_00000_label_7.png
    ├── test_00001_label_2.png
    └── ... (10,000 files total)
```

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (optional but recommended)

### Installation

#### Step 1: Clone the repository
```bash
git clone https://github.com/KamdemSamuel/mlp-mnist-from-scratch.git
cd mlp-mnist-from-scratch
```

#### Step 2: Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy==1.24.3 matplotlib==3.7.1 jupyter==1.0.0 scikit-learn==1.3.0
```

#### Step 4: Download and prepare the dataset
```bash
# Option A: Automatic download via Python
python -c "
from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target
np.savez('data/mnist.npz', X=X, y=y)
print('Dataset saved to data/mnist.npz')
"

# Option B: Manual download
# Download from http://yann.lecun.com/exdb/mnist/
# Extract and place in data/ directory
```

#### Step 5: Launch Jupyter Notebook
```bash
jupyter notebook TP_perceptron_multicouche_MNIST_32x32.ipynb
```

### Verification
Test the installation:
```bash
python -c "import numpy; import matplotlib; import jupyter; print('All dependencies installed!')"
```

---

## Usage

Run the complete notebook to train and evaluate the MLP:

```bash
jupyter notebook TP_perceptron_multicouche_MNIST_32x32.ipynb
```

**Notebook workflow:**
1. **Data loading:** Load MNIST from dataset directory or fetch from scikit-learn
2. **Preprocessing:** Normalize to [0, 1], flatten to 1024D vectors
3. **Model initialization:** Initialize weights with Xavier/Glorot initialization
4. **Training:** SGD for 100 epochs with batch size 32
5. **Evaluation:** Compute accuracy on train/test sets
6. **Analysis:** Weight heatmaps, activation profiles, confusion matrices
7. **Visualization:** Loss curves, accuracy trends, error analysis

---

## Algorithm Details

### Forward Pass
```
Input: x ∈ ℝ^1024
h = ReLU(W1 @ x + b1)           [64 neurons]
z = Softmax(W2 @ h + b2)        [10 logits]
```

### Backpropagation
```
dL/dW2 = (z - y) @ h^T
dL/dW1 = (W2^T @ (z - y) ⊙ ReLU'(h)) @ x^T
```

### Stochastic Gradient Descent
```
W := W - η * dL/dW    (with optional L2 regularization)
```

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 100 | Early stopping if loss plateaus |
| Batch size | 32 | Mini-batch SGD |
| Learning rate | 0.01 | Can be adjusted for convergence |
| Regularization (L2) | 0.0001 | Prevents overfitting |
| Weight initialization | Xavier/Glorot | Prevents vanishing gradients |
| Hidden activation | ReLU | Non-linearity for expressiveness |
| Output activation | Softmax | Probability distribution over 10 classes |

---

## Performance Metrics

**Training Performance:**
- Train accuracy: 98.80% (59,280 / 60,000 correct)
- Train loss: 0.045 (cross-entropy)

**Testing Performance:**
- Test accuracy: 93.25% (9,325 / 10,000 correct)
- Test loss: 0.235

**Confusion Analysis:**
- Hardest pairs: (3,8), (4,9), (1,7)
- Easiest classification: 0, 6
- Average confidence on correct predictions: 97.5%
- Average confidence on incorrect predictions: 62.3%

---

## Common Error Confusions

| Digit 1 | Digit 2 | Confusion Rate | Reason |
|---------|---------|----------------|--------|
| 3 | 8 | 12.4% | Similar rounded shapes |
| 4 | 9 | 8.7% | Both have upper loops |
| 1 | 7 | 6.2% | Thin, vertical strokes |
| 5 | 6 | 5.1% | Similar curved bottom |
| 2 | 7 | 3.8% | Top-heavy structure |

---

## Mathematical Background

### Cross-Entropy Loss
$$L = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^{10} y_{i,k} \log(\hat{y}_{i,k})$$

where $y_{i,k}$ is the true label (one-hot encoded) and $\hat{y}_{i,k}$ is the predicted probability.

### ReLU Activation
$$\text{ReLU}(x) = \max(0, x)$$

### Softmax Activation
$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

---

## Why from scratch?

Building ML models without libraries forces deep understanding of what optimization algorithms actually do. This project is part of a broader approach to machine learning that prioritizes mathematical understanding over API usage.

**Related work:**
- [Convex Optimization SVM from Scratch](https://github.com/KamdemSamuel/Convex-Optimization-SVM-Scratch)
- [Quantum Computing with Qiskit](https://github.com/KamdemSamuel/quantum-computing-qiskit)

---

## Troubleshooting

### Issue: Dataset not found
**Solution:** Ensure data is in `data/` directory or modify notebook path to your dataset location.

### Issue: Out of memory
**Solution:** Reduce batch size from 32 to 16, or use a subset of the data for testing.

### Issue: Poor convergence (accuracy ~10%)
**Solution:** Check that labels are correct (0-9), increase learning rate, or verify data normalization.

### Issue: CUDA/GPU errors
**Solution:** This project uses only NumPy (CPU); ignore GPU-related warnings.

---

## References
- LeCun, Y., Cortes, C., & Burges, C. J. (1998). *MNIST handwritten digit database*
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*
- Nielsen, M. A. (2015). *Neural Networks and Deep Learning*
- NumPy Documentation: https://numpy.org/doc/

---

## Authors
- **Tchuenche Kamdem Samuel Yedidya** - [LinkedIn](https://www.linkedin.com/in/kamdem-samuel-yedidya/)
- **Mayagi Paul Alain**
- **Nkot Jean Francky**

**Course:** INF372 — Machine Learning, Université de Yaoundé I

---

## License
This project is licensed under the MIT License. See `LICENSE` file for details.
