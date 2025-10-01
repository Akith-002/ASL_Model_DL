# 🤟 ASL Alphabet Recognition Model

A deep learning model for American Sign Language (ASL) alphabet recognition using **MobileNetV3Large** architecture with transfer learning. This project achieves high accuracy in classifying ASL hand signs for letters A-Z and special characters.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training Process](#training-process)
- [Results](#results)
- [Model Export](#model-export)
- [Requirements](#requirements)
- [License](#license)

## 🎯 Overview

This project implements a state-of-the-art deep learning model for recognizing American Sign Language alphabet gestures. The model uses **MobileNetV3Large** as the base architecture with custom classification layers, trained in two phases:

1. **Phase 1**: Training the classifier head with frozen base model
2. **Phase 2**: Fine-tuning the entire network with reduced learning rate

The model is optimized for both accuracy and deployment, with support for:

- **Keras** format for training and evaluation
- **TensorFlow Lite** format for mobile and edge device deployment

## ✨ Features

- 🧠 **Transfer Learning**: Leverages pre-trained MobileNetV3Large on ImageNet
- 🎨 **Data Augmentation**: Random rotation, zoom, contrast, and brightness adjustments
- ⚖️ **Class Balancing**: Automatic class weight calculation for imbalanced datasets
- 📊 **Comprehensive Evaluation**: Detailed metrics, confusion matrix, and visualizations
- 📱 **Mobile-Ready**: TensorFlow Lite export for on-device inference
- 🚀 **GPU Acceleration**: Mixed precision training support for faster training
- 📈 **Learning Rate Scheduling**: Adaptive learning rate reduction on plateau

## 📊 Dataset

The model is trained on the **ASL Alphabet Dataset** which includes:

- **26 letters** (A-Z)
- **3 special characters** (space, delete, nothing)
- **Total**: 29 classes

### Data Split

- **Training**: 70% of the dataset
- **Validation**: 15% of the dataset
- **Test**: 15% of the dataset

Expected dataset structure:

```
dataset/
├── A/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── B/
├── C/
...
├── Z/
├── space/
├── del/
└── nothing/
```

## 🏗️ Model Architecture

The model consists of:

1. **Base Model**: MobileNetV3Large (pre-trained on ImageNet)

   - Input shape: 200x200x3
   - Pooling: Global Average Pooling
   - Initial state: Frozen (Phase 1)

2. **Custom Head**:

   - Dropout layer (0.2)
   - Dense layer (29 units, softmax activation)

3. **Training Configuration**:
   - **Phase 1**: Adam optimizer (lr=0.001), 15 epochs
   - **Phase 2**: Adam optimizer (lr=0.00002), 15 epochs
   - Loss: Categorical Crossentropy
   - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## 🚀 Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- CUDA-compatible GPU (optional, but recommended)

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Notebook

1. **Open the notebook**:

   ```bash
   jupyter notebook asl-model.ipynb
   ```

2. **Update dataset path** in the notebook to point to your ASL dataset location

3. **Run all cells** to:
   - Load and prepare the dataset
   - Train the model
   - Evaluate performance
   - Export models

### Using the Trained Model

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('models/model.keras')

# Load and preprocess image
img = Image.open('test_image.jpg').resize((200, 200))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Load class names
with open('models/training_set_labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"Predicted: {class_names[predicted_class]}")
print(f"Confidence: {predictions[0][predicted_class]:.2%}")
```

## 🎓 Training Process

The training follows a two-phase approach:

### Phase 1: Transfer Learning (15 epochs)

- Base model layers are frozen
- Only the classification head is trained
- Higher learning rate (0.001)
- Class weights applied for imbalanced data

### Phase 2: Fine-tuning (15 epochs)

- All layers are unfrozen
- Entire network is fine-tuned
- Lower learning rate (0.00002)
- Learning rate reduction on plateau

### Data Augmentation

Applied during training to improve generalization:

- Random rotation (±10%)
- Random zoom (±10%)
- Random contrast (±20%)
- Random brightness (±20%)
- Rescaling to [0, 1]

## 📈 Results

The model achieves high accuracy on the test set with robust performance across all ASL alphabet classes.

### Training Outputs

- `best_model_phase1.keras`: Best model from Phase 1
- `best_model_final.keras`: Final best model after Phase 2
- `training_results.png`: Visualization of training metrics
- `training_history.json`: Complete training history
- `model_metadata.json`: Model information and metadata

### Visualization

Training plots include:

- Training vs Validation Accuracy
- Training vs Validation Loss
- Final metrics summary

## 📦 Model Export

The notebook automatically exports models in multiple formats:

### 1. Keras Format (`.keras`)

- Full model with architecture and weights
- Use for continued training or Python inference
- Location: `models/model.keras`

### 2. TensorFlow Lite Format (`.tflite`)

- Optimized for mobile and edge devices
- Smaller file size with quantization
- Location: `models/model.tflite`

### 3. Supporting Files

- `training_set_labels.txt`: Class names mapping
- `model_metadata.json`: Model configuration and metrics
- `training_history.json`: Complete training logs

## 📋 Requirements

```
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tensorflow>=2.8.0
pillow>=8.0.0
```

## 🔧 Configuration

Key hyperparameters that can be adjusted:

```python
BATCH_SIZE = 64          # Batch size for training
IMG_SIZE = (200, 200)    # Input image dimensions
EPOCHS_PHASE1 = 15       # Training epochs for Phase 1
EPOCHS_PHASE2 = 15       # Training epochs for Phase 2
LEARNING_RATE_1 = 0.001  # Phase 1 learning rate
LEARNING_RATE_2 = 0.00002 # Phase 2 learning rate
```

## 🎯 Use Cases

- **Mobile Applications**: Real-time ASL recognition on smartphones
- **Educational Tools**: Interactive ASL learning applications
- **Accessibility Solutions**: Communication aids for deaf and hard-of-hearing individuals
- **Research**: Baseline for gesture recognition research

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- ASL Alphabet Dataset on Kaggle
- TensorFlow and Keras teams
- MobileNetV3 architecture by Google Research

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for the deaf and hard-of-hearing community**
