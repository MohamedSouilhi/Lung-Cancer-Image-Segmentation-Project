# Lung Cancer Detection using U-Net with Attention Mechanism

This project implements a lung cancer detection model using a U-Net architecture with an attention mechanism. The goal is to perform semantic segmentation to identify cancerous regions in lung images.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Loss Functions](#loss-functions)
5. [Feature Extraction](#feature-extraction)
6. [Training](#training)
7. [Results](#results)
8. [Requirements](#requirements)
9. [Usage](#usage)
10. [Acknowledgments](#acknowledgments)

## Introduction
This project is designed to aid in lung cancer detection using a convolutional neural network (CNN). It utilizes a U-Net architecture enhanced with attention blocks to improve segmentation accuracy.

## Dataset
The model is trained, validated, and tested using lung image datasets with corresponding segmentation masks. Each image is resized to 256x256 for compatibility with the U-Net input requirements.

### Dataset Structure
- **Training**: `/content/data/train`
  - Images: `/images`
  - Masks: `/labels`
- **Validation**: `/content/data/val`
  - Images: `/images`
  - Masks: `/labels`
- **Testing**: `/content/data/test`
  - Images: `/images`
  - Masks: `/labels`

## Model Architecture
The U-Net model is designed for semantic segmentation and includes:
- Encoder path: Series of convolutional and max-pooling layers.
- Decoder path: Up-sampling and concatenation layers.
- Attention blocks: Enhance the feature selection process by highlighting important regions.

### Attention Block
Attention blocks are added between the encoder and decoder paths to focus on relevant features and suppress irrelevant ones.

## Loss Functions
To optimize model performance, advanced loss functions are used:
- **Dice Loss**: Helps in handling data imbalance by focusing on region overlap.
- **Focal Loss**: Addresses class imbalance by penalizing misclassified examples.

## Feature Extraction
The project includes feature extraction techniques:
- **Local Binary Pattern (LBP)**: Captures texture features.
- **Histogram of Oriented Gradients (HOG)**: Describes the shape and structure of lung regions.

## Training
The model is trained using the following configuration:
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy with Dice Loss
- Metrics: Accuracy
- Input Shape: `(256, 256, 1)`

### Callbacks
- **EarlyStopping**: Stops training if validation loss does not improve.
- **ReduceLROnPlateau**: Reduces the learning rate when a plateau is detected.
- **ModelCheckpoint**: Saves the best model during training.

## Results
Model evaluation metrics:
- **Accuracy**: Measures segmentation performance.
- **Dice Coefficient**: Quantifies overlap between predicted and true masks.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- OpenCV
- scikit-image
- Matplotlib

Install dependencies using:
```bash
pip install -r requirements.txt
