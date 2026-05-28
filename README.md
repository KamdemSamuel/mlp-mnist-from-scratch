# MLP on MNIST — From Scratch

Implementation of a Multilayer Perceptron (MLP) trained on the MNIST dataset
using only NumPy — no ML libraries (no TensorFlow, no PyTorch, no Scikit-learn).

**Authors:** Tchuenche Kamdem Samuel Yedidya, Mayagi Paul Alain, Nkot Jean Francky  
**Course:** INF372 — Machine Learning, Université de Yaoundé I, L3 Informatique  
**Year:** 2024–2025

---

## What this project does

Trains a neural network to recognize handwritten digits (0–9) from the MNIST
dataset, implementing every component manually from mathematical first principles.

**Final results:**
- Train accuracy: **98.80%**
- Test accuracy: **93.25%**
- Training: 100 epochs, SGD, batch size 32

---

## Architecture

| Layer  | Size          | Activation |
|--------|---------------|------------|
| Input  | 1024 (32×32)  | —          |
| Hidden | 64 neurons    | ReLU       |
| Output | 10 neurons    | Softmax    |

Total parameters: 66,250

---

## What is implemented from scratch

- Forward propagation
- Backpropagation
- Stochastic Gradient Descent (SGD)
- ReLU and Softmax activation functions
- Cross-entropy loss
- Weight heatmap visualization
- Hidden layer activation analysis per class

---

## Key findings

- SGD converges stably over 100 epochs
- Weight heatmaps show neurons progressively ignoring irrelevant pixels
- Similar digit classes (1/7, 3/8, 4/9) share active neurons — explaining confusions
- When the model makes an error, the activation profile resembles the predicted
  class more than the true class — the network does not fail randomly

---

## Files
