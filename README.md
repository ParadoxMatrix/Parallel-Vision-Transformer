# Parallel-Vision-Transformer

# Deepfake Detection using Parallel Vision Transformers (PViT) 🤖🧠

This project implements a powerful **Parallel Vision Transformer (PViT)** model to detect deepfake images. Deepfake content, generated using GANs and similar technologies, poses a significant threat to digital trust and cybersecurity. Our research uses transformer-based architectures to accurately classify real and fake face images with impressive performance.

---

## 📌 Project Overview

- **Model**: Parallel Vision Transformer (PViT)
- **Dataset**: 140,000 real & fake face images
  - 70k real faces from Flickr
  - 70k fake faces from NVIDIA StyleGAN
- **Accuracy**: Achieved **91.92% test accuracy**
- **Platform**: Trained on **Google Colab** using **NVIDIA A100 GPU**

---

## ✨ Key Features

- Utilizes **parallel attention and feedforward layers** to process multiple feature maps simultaneously.
- Performs better than traditional CNN, RNN, and standard ViT models.
- Designed for high robustness in detecting subtle deepfake artifacts.
- Strong real-world applications in **cybersecurity**, **forensics**, **media integrity**, and **identity verification**.

---

## 🧠 Model Architecture Summary

- Input images resized to 224x224 and divided into patches.
- Feature vectors generated using fully connected layers.
- Transformers run **in parallel**, enabling simultaneous multi-perspective learning.
- Uses Adam optimizer with ReduceLROnPlateau.
- Trained over 50 epochs with early stopping to prevent overfitting.

---

## ⚙️ Technical Stack

- **Python**
- **TensorFlow**
- Google Colab (with A100 GPU)
- Numpy, Matplotlib, OpenCV

---

## 📊 Performance Metrics

| Metric              | Value    |
|---------------------|----------|
| Test Accuracy       | 91.92%   |
| Validation Accuracy | 91.70%   |
| Precision           | 95.20%   |
| Recall              | 88.21%   |
| F1 Score            | 91.57%   |
| AUC                 | 97.90%   |

> 🏆 *Outperforms CNN, RNN, and standard ViTs on the same dataset.*

---

## 🔬 Comparison with Other Models

| Model | Accuracy | AUC |
|-------|----------|-----|
| CNN   | 88.54%   | 95.21% |
| RNN   | 86.75%   | 93.50% |
| ViT   | 89.91%   | 96.35% |
| **PViT (Ours)** | **91.92%** | **97.90%** |

---

## 🚧 Limitations & Future Scope

- Requires large datasets and high compute resources.
- Future goals:
  - Fine-tuning for **real-time detection**
  - Expansion to **audio-visual deepfakes**
  - **Quantum-inspired models** and **self-supervised training**

---

## 🛡️ Ethical Note

While detecting deepfakes is essential for online safety, ethical considerations are vital in deploying such models. This project is focused on **defense**, not surveillance or misuse.

---

## 📁 File Structure

