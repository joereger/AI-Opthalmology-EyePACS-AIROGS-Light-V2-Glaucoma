# Component: Phase 04 - Evaluate

## Description

This phase assesses the performance of a trained model using the validation dataset. It calculates standard classification metrics to quantify how well the model distinguishes between Referable Glaucoma (RG) and Non-Referable Glaucoma (NRG).

## Key Files

*   `src/pipeline/phase04_evaluate/main.py`
    *   **Role**: Loads the trained model and validation data, runs inference, and computes metrics.
*   `src/pipeline/phase04_evaluate/config.yaml`
    *   **Role**: Configuration for evaluation (batch size, etc.).

## Technology and Architecture

*   **Metrics**: Uses `scikit-learn` to calculate Accuracy, Precision, Recall, F1-Score, and AUC (Area Under the ROC Curve).
*   **Model Loading**: dynamically loads the model architecture and weights based on the `run_id` and metadata.

## TODO

*   No specific TODOs.

## Progress

*   **Stable**: Evaluation logic is functioning.

## Key Lessons and Insights

*   Separating evaluation from training allows us to re-evaluate old models or evaluate on new datasets without re-triggering the expensive training process.
