# Neural Network from Scratch in Python

This project is a **minimalist, from-scratch implementation of a feedforward neural network in Python**. It is designed for educational purposes, helping users understand the **core building blocks of neural networks**—including forward propagation, backpropagation, and gradient descent—**without relying on high-level deep learning frameworks**.

The implementation is capable of training on datasets such as the **MNIST handwritten digits dataset** (assumed in `train.csv` format).

---

## ✨ Features
- **Multi-Layer Architecture**  
  Support for multiple hidden layers with customizable sizes.  

- **Activation Functions**  
  - **ReLU** (Rectified Linear Unit) for hidden layers  
  - **Softmax** for output layer (multi-class classification)  
  - *(Optional)* **Sigmoid** function for binary classification (commented out for reference).  

- **Backpropagation**  
  Core backpropagation algorithm for computing gradients and updating weights & biases.  

- **Cross-Entropy Loss**  
  Stable and widely used loss function for classification.  

- **Training & Evaluation**  
  - Training loop with gradient descent optimizer  
  - Accuracy evaluation on predictions  
  - Training progress visualization  

---

## ⚙️ How it Works

The project is structured into modular components:

- **`Layer` Class**  
  Represents a single neural network layer with its own weights, biases, and activation function.  

- **`Model` Class**  
  Orchestrates the network, managing multiple `Layer` objects. Handles:  
  - **Forward Propagation**: Passes data through layers to produce outputs.  
  - **Backward Propagation**: Uses the chain rule to compute gradients.  
  - **Training Loop**: Updates weights & biases with gradient descent.  

- **`Data_analysis` Class**  
  Utility class for predictions, accuracy calculation, and visualization.  

---

## 📦 Prerequisites

Make sure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib
```

1. Clone the repository
```bash
git clone https://github.com/Legendparth/Basic_Neural_network_from_scratch.git
cd Basic_Neural_network_from_scratch
```

Place your dataset at:

2. Prepare the Dataset
```bash
datasets/training/train.csv
```

- Format:
- First column → labels (digits 0–9 for MNIST)
- Remaining columns → pixel values

```bash
python neural_net.py
```

The script will:

- Train the network
- Display progress
- Plot accuracy over time
- Print the final test accuracy

```bash
.
├── datasets/
│   └── training/
│       └── train.csv
├── neural_net.py
└── README.md
```

🧑‍💻 Educational Purpose
This project is not optimized for speed or production, but rather to demystify how neural networks work internally.
It is recommended for:

- Beginners learning machine learning
- Students building projects from first principles
- Developers curious about the math behind deep learning frameworks
