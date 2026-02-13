# ACE Implementation

This repo is created to overcome the concepts of Automated Concept-based Explanations (ACE) for education purposes.

## Repository architecutre
```
src/
├── backend/
├── frontend/
├── models/
│   ├── CNN/
│   ├── KMeans/
│   ├── SLIC/
│   └── TCAV/
└── datasets/
README.md
```

### CNN

A custom Convolutional Neural Network (CNN) was trained on the Imagenette dataset for a 10-class image classification task. The input image resolution is 128 × 128 (RGB).

#### Architecture Overview

- Input: 3 × 128 × 128

- Output: 10 class logits

#### Feature extractor:

- Stacked convolutional layers with kernel sizes 7×7, 5×5, and 3×3

- Channel progression: 3 → 64 → 192 → 384 → 256 → 256

- Each convolution is followed by Batch Normalization and ReLU

- MaxPooling used for spatial downsampling

- Dropout (p = 0.1) applied after the final pooling layer

- Final feature map size: 256 × 6 × 6

#### Classifier:

- Fully connected layers: 9216 → 4096 → 4096 → 10

- ReLU activations between layers

- Dropout (p = 0.5) used for regularization

#### Training Setup

- Epochs: 80

- Batch size: 64

- Learning rate: 3 × 10⁻⁴

- Optimizer: Adam

- Loss function: Cross-Entropy Loss

- Learning rate scheduler: StepLR (step size = 30)

#### Results

Final accuracy: 85.87% after 80 epochs

## Load the CNN

To load the model you may go here:

```bash
http://193.104.57.253/download/cnn_model
```