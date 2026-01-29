# Intel Image Classification – End-to-End ML Pipeline

## Project Overview
This repository demonstrates a complete **image classification pipeline** using TensorFlow and transfer learning.  
It is structured for professional, production-like workflows, including data preprocessing, model training, checkpointing, and evaluation.

The goal is to classify natural scene images into six categories: **buildings, forest, glacier, mountain, sea, and street**.

---

## Dataset
**Intel Image Classification dataset (Natural Scenes)**

- 6 classes, 3,600+ images
- Training: 600 images (100 per class)  
- Validation: 3,000 images
- Challenges: varied lighting, similar textures, and real-world image noise

> The dataset folder structure follows best practices for reproducibility:
>
> ```
> data/raw/train/<class_name>/
> data/raw/val/<class_name>/
> ```

---

## ML Workflow

1. **Data Loading and Preprocessing**
   - Images resized to 224x224
   - Normalization with `tf.keras.layers.Rescaling`
   - Batched for efficient training

2. **Model Architecture**
   - Transfer learning using **MobileNetV2** (pretrained on ImageNet)
   - Global Average Pooling + Dense output layer
   - Frozen backbone for initial training

3. **Training**
   - Optimizer: Adam
   - Loss: Sparse Categorical Crossentropy
   - Early stopping with best model checkpointing
   - Best model saved in **Keras native format**: `checkpoints/best_model.keras`

4. **Evaluation**
   - Validation accuracy, precision, recall, F1-score per class
   - Confusion matrix visualization
   - Automated script for reproducibility

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Train the model
python -m src.training.train
3. Evaluate the model
python -m src.evaluation.evaluate
Confusion matrix will be saved as confusion_matrix.png in the project root

Classification metrics printed in terminal

Repository Structure
image-classification-intel/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   └── raw/
│       ├── train/
│       └── val/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
└── checkpoints/
    └── best_model.keras
Environment
Python 3.10 (managed via pyenv recommended)

TensorFlow 2.x

Tested on Windows / Linux

Results
Validation accuracy typically above 90% after 10 epochs

Confusion matrix demonstrates strong class separation

Modular, reproducible pipeline for further experimentation


