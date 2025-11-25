# Project Components

This file lists the high-level components of the system. Each entry links to a detailed component file.

*   [CLI & Orchestration](componentCLI.md) - The central command-line interface that orchestrates the execution of pipeline phases and manages global configuration.
*   [Core System](componentCore.md) - The foundational libraries used across the system, including the Model architecture, Data Handling classes, and shared Utility functions.
*   [Phase 01: Ingest](componentPhase01Ingest.md) - Responsible for ingesting raw data.
*   [Phase 02: Conform](componentPhase02Conform.md) - Standardizes raw data into a consistent `ImageFolder` structure.
*   [Phase 03: Train](componentPhase03Train.md) - Handles the training of the MobileNetV3 deep learning model.
*   [Phase 04: Evaluate](componentPhase04Evaluate.md) - Evaluates trained models using standard metrics (Accuracy, AUC, etc.).
*   [Phase 05: Predict](componentPhase05Predict.md) - Runs inference on new, unseen images.
*   [Phase 06: GradCAM](componentPhase06GradCAM.md) - Generates Grad-CAM visual explanations for model predictions.
*   [Phase 07: Activation Max](componentPhase07ActivationMax.md) - Generates Activation Maximization visualizations to understand what the model has learned.
