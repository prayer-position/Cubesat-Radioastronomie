# Cubesat GMSK Transceiver & AI Denoising

This project implements a complete Digital Signal Processing (DSP) pipeline for **Gaussian Minimum Shift Keying (GMSK)** modulation and demodulation, designed for Cubesat and Radioastronomy applications. It features an AI-driven denoising layer using a Convolutional Auto-Encoder (CAE).

## 🚀 Features
* **Full DSP Chain:** Implementation of GMSK modulation, AWGN channel simulation, and differential demodulation.
* **Spectral Analysis:** Power Spectral Density (PSD) estimation using Welch's method to analyze bandwidth efficiency (BT products).
* **AI Denoising:** A Deep Learning pipeline to clean IQ samples under low SNR conditions.
* **Automation:** Complete setup scripts for virtual environments and dependency management.

## 🛠️ Project Structure
* `src/`: Core logic (Modulation, Demodulation, Channel).
* `Data/`: Scripts for generating synthetic IQ datasets for AI training.
* `notebooks/`: Interactive Jupyter Notebooks for BER analysis and system demos.
* `ai/`: Neural Network architectures and training loops.

## 📦 Setup & Installation
Ensure you have Python 3.8+ installed.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Cubesat-Radioastronomie.git](https://github.com/YOUR_USERNAME/Cubesat-Radioastronomie.git)
   cd Cubesat-Radioastronomie
