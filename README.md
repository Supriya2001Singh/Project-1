# # Fire and Smoke Detection System

## Overview
This project aims to develop a system for detecting the presence of fire or smoke in images using deep learning techniques. The system utilizes a Convolutional Neural Network (CNN) trained on a dataset of labeled images containing fire/smoke and non-fire/non-smoke instances.

## Key Features
- Image classification to determine whether an image contains fire/smoke or not.
- Text overlay indicating "Fire Detected" on images where fire/smoke is detected.
- Model evaluation metrics including accuracy, precision, recall, and F1-score.

## Usage
1. **Data Collection:** Gather a dataset of images containing fire/smoke and non-fire/non-smoke instances.
2. **Data Preprocessing:** Preprocess the images by resizing, normalization, and augmentation.
3. **Model Training:** Train a CNN model on the preprocessed image data.
4. **Model Evaluation:** Evaluate the trained model's performance on a test set to assess its accuracy and other metrics.
5. **Deployment:** Deploy the trained model to detect fire/smoke in new images.

## Requirements
- Python 3
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## Directory Structure
.
├── data/ # Directory for storing dataset
│ ├── train/ # Training images
│ ├── test/ # Testing images
│ └── ...
├── models/ # Directory for saving trained models
├── src/ # Source code files
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ └── detection.py
├── README.md # Project overview and instructions
└── requirements.txt # Python dependencies


## Future Work
- Real-time fire/smoke detection in video streams.
- Integration with other sensor data for multimodal detection.
- Enhancing model accuracy through advanced deep learning techniques.
- Deployment in real-world scenarios for fire prevention and management.

