# 🖥️ E-Waste Generation Classification using EfficientNetV2B0

## 📌 Problem Statement
Electronic waste (e-waste) is increasing rapidly due to the fast growth of technology. Improper disposal poses serious environmental and health hazards. Manual classification is time-consuming, error-prone, and inefficient on a large scale.

## 🎯 Goal
To develop an AI-based image classification model that can automatically classify e-waste into 10 categories using transfer learning with EfficientNetV2B0 and deploy the model using a Gradio interface.

## 📊 Dataset
- Source: Kaggle (10 e-waste categories like Mobile, Keyboard, PCB, etc.)
- Split:  
  - Training: 2400 images  
  - Validation: 300 images  
  - Testing: 300 images

## 🛠️ Tools & Technologies
- Python 3.x
- Jupyter Notebook
- TensorFlow & Keras
- EfficientNetV2B0 (Transfer Learning)
- Gradio (Web UI)
- PIL (Image Processing)
- Numpy, Matplotlib, Scikit-learn

## 🔄 Methodology

1. **Dataset Preprocessing**  
   - Resize: 128x128 pixels  
   - Rescale: Normalize [0–255] → [0–1]  
   - Augment: Random flip, rotation, zoom  
   - Normalize: Using `preprocess_input()` for EfficientNet

2. **Transfer Learning**  
   - Used pretrained EfficientNetV2B0  
   - Removed top layers  
   - Froze first 100 layers  
   - Added new classification head

3. **Training Configuration**  
   - Optimizer: Adam (lr = 0.0001)  
   - Loss: SparseCategoricalCrossentropy  
   - Metric: Accuracy  
   - EarlyStopping with best weight restore

## 🧠 Model Accuracy
- Achieved **96% accuracy** on test data
- Strong generalization across 10 e-waste categories

## 🚀 Output & Deployment
- Real-time predictions via **Gradio** interface
- Upload an image and instantly get the e-waste category

