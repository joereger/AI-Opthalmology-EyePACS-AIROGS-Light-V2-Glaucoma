# Component: Phase 05 - Predict

## Description

This phase is designed for running inference on new, unseen images ("wild" data). It loads a specific trained model (identified by `run_id`) and applies it to images located in a designated input directory.

It outputs the model's prediction (RG vs. NRG) and the confidence score for each image.

## Key Files

*   `src/pipeline/phase05_predict/main.py`
    *   **Role**: Orchestrates the loading of images, application of transforms, model inference, and results saving.
*   `src/pipeline/phase05_predict/config.yaml`
    *   **Role**: Configures input/output directories and batch processing parameters.

## Technology and Architecture

*   **Inference Mode**: Sets the PyTorch model to `eval()` mode to disable dropout/batchnorm updates.
*   **Metadata Usage**: Critically relies on `model_metadata.json` to know the correct normalization stats to apply to the new images.

## TODO

*   No specific TODOs.

## Progress

*   **Stable**: Prediction logic is functioning.

## Key Lessons and Insights

*   Handling "wild" data requires robust error handling for corrupt images or non-image files in the input directory.
