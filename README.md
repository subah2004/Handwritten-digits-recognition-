# 🖊️ Handwritten Digit Recognition - MNIST and EMNIST 🖊️

This project is a convolutional neural network (CNN) model for recognizing handwritten digits and characters. Using the MNIST and EMNIST datasets, it accurately classifies digit and letter images. The model leverages PyTorch for building and training the CNN, achieving efficient and accurate classification.

## 📜 Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
  - [MNIST](#mnist)
  - [EMNIST](#emnist)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)

## 🎯 Project Overview

Handwritten digit recognition is an essential task in computer vision, used in applications like automated document processing and postal mail sorting. This project implements a CNN model in PyTorch to recognize handwritten digits from the MNIST and EMNIST datasets, enabling users to explore deep learning with a well-structured image classification problem.

## 📚 Datasets

### MNIST
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels. It’s widely used as a benchmark in machine learning and deep learning tasks.

### EMNIST
The EMNIST dataset extends MNIST by including letters and is organized into several splits. This project focuses on the EMNIST Digits and Letters splits to handle both digit and character recognition.

## 🏗️ Model Architecture

This CNN model has the following structure:
- **Input Layer**: Grayscale 28x28 images (1 channel).
- **First Convolutional Block**:
  - Two convolutional layers with 32 filters each, kernel size of 3x3, ReLU activation, and 2x2 max pooling.
- **Second Convolutional Block**:
  - Two convolutional layers with 64 filters each, kernel size of 3x3, ReLU activation, and 2x2 max pooling.
- **Fully Connected Classifier Block**:
  - Fully connected layers with dropout to prevent overfitting.
  - Outputs a prediction over 10 classes (0-9) for MNIST or a larger set for EMNIST.

The model is designed to capture spatial hierarchies in handwritten characters, while dropout layers help reduce overfitting.

### 📝 Code Explanation

Main model components:
- **`conv_1` and `conv_2`**: Sequential layers with convolution, ReLU, and max-pooling operations to capture spatial information.
- **`classifier`**: Fully connected layers with dropout for final classification.

## ⚙️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   > Ensure `torch` and `torchvision` are installed. The `requirements.txt` file should include any additional dependencies.

## 🚀 Usage

1. **Run the Model**:
   Train the CNN on either MNIST or EMNIST by executing the main script.
   ```bash
   python main.py --dataset mnist  # or 'emnist' for the EMNIST dataset
   ```

2. **Evaluate the Model**:
   After training, the script outputs the model’s accuracy on the test dataset.

3. **Adjust Model Hyperparameters**:
   Modify hyperparameters such as learning rate, batch size, and number of epochs in the configuration file or directly in `mnist.py` or `emnist.py` to experiment with model performance.

## 📊 Results

The CNN achieves high accuracy on both MNIST and EMNIST datasets.

## 🔮 Future Improvements

Possible enhancements:
- **Experiment with Larger Architectures**: Test deeper architectures or transfer learning with pre-trained models.
- **Add More EMNIST Splits**: Extend the model to support additional EMNIST splits, such as ByClass or ByMerge.
- **Add Webcam functionality**: We need to integrate OpenCV to enable webcam functionality for real-time digit recognition.

--- 
