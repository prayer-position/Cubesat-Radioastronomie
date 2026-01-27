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
* `weights/`: Neural Network architectures and training loops.

## 📦 Setup & Installation
Ensure you have Python 3.8+ installed.

1. **Clone the repository:**
   	```bash
  	git clone git@github.com:prayer-position/Cubesat-Radioastronomie.git
   	cd Cubesat-Radioastronomie
	```

2. **Setup environment**
   	```bash
   	bash setup.sh 
	```
If on Windows/git bash use : 
	```
	source venv_cubesat/Scripts/activate
	```
If on Mac/Linux use : 
	``` 
	source venv_cubesat/bin/activate
	```

3. **Path fixing**
	```bash
	pip install -e .
	```

