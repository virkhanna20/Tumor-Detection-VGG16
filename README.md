# Brain Tumor Detection with Transfer Learning (VGG16)

This project demonstrates how to build a brain tumor detection model using transfer learning with the VGG16 convolutional neural network. 

## Overview

The model is trained on a dataset of brain MRI images, leveraging the pre-trained weights of the VGG16 model to extract relevant features. By adding a custom classification head on top of the VGG16 base, the model learns to classify images as either "Tumor" or "No Tumor."

## Key Features:

- **Transfer Learning:** Utilizes the pre-trained VGG16 model for feature extraction, significantly reducing training time and improving performance.
- **Custom Classifier Head:**  Includes GlobalAveragePooling, Dense layers, and Softmax activation to adapt VGG16 to the brain tumor detection task.
- **Regularization Techniques:**  Implements dropout and L2 regularization to prevent overfitting and enhance the model's generalization ability.
- **Early Stopping:**  Monitors validation accuracy during training and stops training when it plateaus to prevent overfitting.
- **Image Preprocessing:**  Applies VGG16-specific preprocessing steps to ensure input images are in the correct format.

## Dataset

The project uses the "Brain MRI Images for Brain Tumor Detection" dataset available on Kaggle: [https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-learn
- tqdm

You can install the necessary packages using pip:

```bash
pip install -r requirements.txt
Citation

If you use this code or dataset in your research, please cite this repository and the original dataset.
