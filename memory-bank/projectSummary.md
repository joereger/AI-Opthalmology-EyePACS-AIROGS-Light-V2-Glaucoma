# Project Summary: AI Ophthalmology - Glaucoma Detection Pipeline

## Description of the Project

This project is a modular, Python-based command-line application designed to automate the screening of retinal fundus images for glaucoma. Specifically, it uses deep learning to classify images into two categories: **Referable Glaucoma (RG)** and **Non-Referable Glaucoma (NRG)**.

The system is built as a multi-phase pipeline that handles the entire workflow:
1.  **Ingestion & Conformation**: processing raw data into a standard format.
2.  **Training**: fine-tuning a MobileNetV3 model.
3.  **Evaluation**: calculating performance metrics (Accuracy, AUC, etc.).
4.  **Prediction**: running inference on new images.
5.  **Explainability**: generating visual explanations (Grad-CAM, Activation Maximization) to help clinicians understand the model's decisions.

The application is structured to be reproducible, with clear tracking of experiments via `run_id`s, metadata generation, and organized output directories.

## Technology Context

*   **Core Language**: Python 3.12
*   **Deep Learning Framework**: PyTorch (`torch`, `torchvision`)
*   **CLI Framework**: `click` (for command-line orchestration)
*   **Data Handling**: `pandas`, `numpy`, `Pillow`, `tqdm`
*   **Image Processing**: `opencv-python`, `matplotlib`
*   **Configuration**: YAML (`PyYAML`)
*   **Environment**: Developed on macOS. Uses a Python virtual environment.
*   **Key Libraries**: `scikit-learn` (metrics), `torchvision` (models/transforms).
