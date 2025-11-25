# Component: Core System

## Description

The Core System component encompasses the foundational libraries and shared logic used throughout the pipeline. This includes the definition of the deep learning model architectures, data handling utilities (datasets, transforms), and generic helper functions (file I/O, path management).

Centralizing these elements ensures consistency across phases—for example, ensuring the training phase and prediction phase use the exact same image preprocessing transforms and model definition.

## Key Files

*   `src/models/base_model.py`
    *   **Role**: Defines the abstract base class for models, enforcing a standard interface.
*   `src/models/mobilenetv3/model.py`
    *   **Role**: The concrete implementation of the MobileNetV3 model. Handles loading pre-trained weights, modifying the classifier head, and generating model metadata.
*   `src/data_handling/datasets.py`
    *   **Role**: Custom Dataset classes (e.g., `ImageFolderWithPaths`) that extend PyTorch's standard utilities to provide additional functionality like returning file paths during loading.
*   `src/data_handling/transforms.py`
    *   **Role**: Defines the standard image transformations (resize, normalize, augmentation) used for training and inference.
*   `src/utils/file_utils.py`
    *   **Role**: Helper functions for file system operations, such as determining the next `run_id` or creating directories.

## Technology and Architecture

*   **PyTorch**: The `torch.nn.Module` system is the foundation for `src/models`.
*   **torchvision**: Used for the base MobileNetV3 architecture and image transforms.
*   **Inheritance**: `datasets.py` typically inherits from `torchvision.datasets.ImageFolder` to reuse existing robust logic while adding custom behavior.

## TODO

*   **Refinement**: Consider adding fallback logic for `get_transforms` in `utils.py` (mentioned in active context).

## Progress

*   **2025-05-08**: Added `shutil` import to utilities.
*   **2025-05-08**: Refactored `src/data_handling/datasets.py` to include `ImageFolderWithPaths` class, enabling phases to track which image file corresponds to a specific tensor.
*   **2025-05-08**: Updated `src/models/mobilenetv3/model.py` (indirectly involved in Phase 07 debugging regarding metadata).

## Key Lessons and Insights

*   Saving model metadata (normalization stats, class names) alongside the model weights is critical for reproducibility and ensuring that inference/visualization phases treat the data exactly as the training phase did.
