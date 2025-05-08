# Project Brief: AI Ophthalmology - Glaucoma Detection Pipeline

## Core Goal

To develop a modular, Python-based command-line application capable of detecting referable glaucoma (RG) vs. non-referable glaucoma (NRG) from retinal fundus images using a deep learning model (MobileNetV3).

## Key Requirements

-   Implement a multi-phase pipeline for data processing, model training, evaluation, prediction, and visualization.
-   Utilize the "Glaucoma Dataset - EyePACS-AIROGS Light V2".
-   Provide explainability features (Grad-CAM, Activation Maximization) to understand model decisions.
-   Maintain clear code structure and documentation.
-   Enable execution of individual pipeline phases via a CLI.
-   Store model artifacts, results, and visualizations in a structured `data/` directory, organized by `run_id`.
-   Generate model metadata alongside trained models for reproducibility and consistency.
