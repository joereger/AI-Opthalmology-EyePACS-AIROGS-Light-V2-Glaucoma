# Component: Phase 03 - Train

## Description

This phase handles the fine-tuning of the deep learning model (MobileNetV3). It loads the conformed dataset, initializes the model, and executes the training loop (forward pass, loss calculation, backward pass, optimizer step).

It is also responsible for saving the best model weights and crucially, the model metadata (architecture name, normalization statistics, class mappings) required for inference.

## Key Files

*   `src/pipeline/phase03_train_model/main.py`
    *   **Role**: Contains the training loop, validation logic, and checkpoint saving mechanism.
*   `src/pipeline/phase03_train_model/config.yaml`
    *   **Role**: Defines training hyperparameters (learning rate, batch size, num_epochs, etc.).

## Technology and Architecture

*   **Training Loop**: Standard PyTorch loop.
*   **Loss Function**: typically CrossEntropyLoss for classification.
*   **Optimizer**: Adam or SGD (configurable).
*   **Device Management**: Automatically selects CUDA, MPS (Apple Silicon), or CPU.

## TODO

*   No specific TODOs.

## Progress

*   **Stable**: The training logic is established.

## Key Lessons and Insights

*   Saving metadata (`model_metadata.json`) at the end of training is a pattern we adopted to ensure that future phases don't have to guess the normalization stats or class indices used during training.
