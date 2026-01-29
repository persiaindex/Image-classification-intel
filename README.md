# Intel Image Classification – End-to-End ML Pipeline

## Problem
Classify natural scene images into six categories using a convolutional neural network.

## Dataset
Intel Image Classification dataset (natural scenes).
Challenges include varied lighting conditions and visual similarity between classes.

## Approach
- Transfer learning using MobileNetV2
- Image normalization and batching
- Training with frozen backbone for stability
- Evaluation using accuracy and validation metrics

## Results
Achieved stable validation accuracy with limited training epochs.
Further fine-tuning can improve performance.

## How to Run
```bash
pip install -r requirements.txt
bash scripts/train.sh
