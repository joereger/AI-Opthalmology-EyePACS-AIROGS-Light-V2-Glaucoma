# Product Context: AI Ophthalmology - Glaucoma Detection Pipeline

## Problem Solved

Glaucoma is a leading cause of irreversible blindness. Early detection through retinal image screening can significantly improve patient outcomes. However, manual review by specialists is time-consuming and resource-intensive. This project aims to automate the initial screening process by identifying images likely showing referable glaucoma (RG), allowing specialists to focus on critical cases.

## How it Works

The system analyzes retinal fundus images (JPEG format) using a pre-trained MobileNetV3 deep learning model. It classifies images into two categories: Non-Referable Glaucoma (NRG) and Referable Glaucoma (RG).

The pipeline includes steps for:
1.  Validating and conforming input data.
2.  Training the model on labeled data.
3.  Evaluating model performance using standard metrics (Accuracy, Precision, Recall, AUC).
4.  Predicting the class for new, unseen images.
5.  Generating visual explanations (Grad-CAM, Activation Maximization) to understand *why* the model made a specific prediction, aiding interpretability.

## User Experience Goals

-   **Clinicians/Researchers:** Provide a reliable, automated tool for initial glaucoma screening from retinal images. Offer insights into model decisions through visualizations.
-   **Developers:** Offer a modular, extensible pipeline structure that is easy to understand, run, and potentially integrate into larger systems. Ensure reproducibility through clear configuration and metadata.

*(Disclaimer: This tool is experimental and not intended for clinical use.)*
