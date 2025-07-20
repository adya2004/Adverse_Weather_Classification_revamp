# Adverse Weather Classification System

A comprehensive deep learning solution for weather condition classification designed to enhance autonomous vehicle navigation and safety in diverse weather conditions.

## Overview

This project implements an intelligent weather classification system using advanced computer vision and deep learning techniques. The system classifies weather conditions into 7 categories: **Cloudy**, **Rainy**, **Snow**, **Sand/Dusty**, **Shine/Sunny**, **Sunrise**, and **Fog**.

## Project Structure

### 🔬 Preprocessing & Training (`preprocessing_and_training/`)

#### Data Processing Pipeline
- **`combine_data.py`**: Automated data organization script that consolidates weather datasets from multiple sources and restructures them into class-specific folders
- **`data-preprocessing-and-modelling.ipynb`**: Main preprocessing pipeline implementing:
  - Image resizing and normalization (224×224×3)
  - Data augmentation (rotation, cropping, horizontal flipping)
  - Train/validation/test split (80/20 division with further 20% validation split)
  - Label encoding for 7 weather classes

#### Advanced Data Processing
- **`advanced-data-processing.ipynb`**: Implements sophisticated preprocessing techniques:
  - **PCA Dimensionality Reduction**: Color channel-wise PCA with 100 components achieving ~90% variance retention
  - **Image Reconstruction Quality Analysis**: Using VIF (Visual Information Fidelity) metrics
  - **Convolution Filtering**: Custom kernels for blur, sharpening, edge detection (Sobel, Laplacian)
  - **Feature Extraction**: ResNet-50 based feature extraction with t-SNE visualization

#### Model Architecture & Training
- **Transfer Learning Implementation**:
  - **EfficientNet-B0**: Achieved 91.5% test accuracy
  - **ResNet-50**: Achieved 91.9% test accuracy
- **Training Configuration**: 50 epochs with Adam optimizer
- **Performance Evaluation**: Confusion matrix analysis and per-class accuracy metrics

#### Visualization & Analysis
- **`Basic Processing.ipynb`**: Color channel analysis and basic image preprocessing
- **`visualisation.ipynb`**: Comprehensive data visualization and exploratory data analysis

### 🖥️ Backend (`backend/`)

FastAPI-based REST API server featuring:
- TensorFlow model serving with GPU optimization
- Image preprocessing pipeline
- Real-time weather classification endpoint
- CORS-enabled for web integration
- Model: Keras-based trained model (`my_model.keras`)

### 🌐 Frontend (`frontend/`)

React-based web application with:
- Modern UI built with Tailwind CSS
- Drag-and-drop image upload
- Real-time classification results
- Responsive design optimized for various devices
- Integration with backend API

## Live Demo

🌐 **Access the live application**: [https://adverse-weather-classification-reva.vercel.app/](https://adverse-weather-classification-reva.vercel.app/)

## Key Features

- **Multi-class Weather Classification**: 7 distinct weather conditions
- **High Accuracy**: 91%+ accuracy on test datasets
- **Real-time Processing**: Fast inference with optimized models
- **Web-based Interface**: User-friendly interface for image upload and classification
- **Advanced Preprocessing**: PCA-based dimensionality reduction and feature extraction
- **GPU Acceleration**: Optimized for both CPU and GPU inference

## Technical Stack

- **Deep Learning**: TensorFlow, Keras, Transfer Learning (EfficientNet, ResNet)
- **Backend**: FastAPI, Python
- **Frontend**: React, Tailwind CSS, Vite
- **Data Processing**: OpenCV, PIL, scikit-learn, NumPy
- **Visualization**: Matplotlib, seaborn

## Applications

Designed specifically for autonomous vehicle systems to:
- Adapt driving behavior based on weather conditions
- Enhance safety through real-time weather awareness
- Optimize navigation routes considering weather impacts
