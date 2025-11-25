# Component: Phase 06 - GradCAM

## Description

This phase provides explainability for the model's predictions using Gradient-weighted Class Activation Mapping (Grad-CAM). It identifies which regions of an input image were most influential in the model's decision to classify it as RG or NRG.

The output typically includes the original image overlaid with a heatmap, highlighting "hot" zones of high importance.

## Key Files

*   `src/pipeline/phase06_visualize_gradcam/main.py`
    *   **Role**: Orchestrates the Grad-CAM generation process for a batch of images.
*   `src/pipeline/phase06_visualize_gradcam/utils.py`
    *   **Role**: Implements the Grad-CAM logic (registering hooks to capture gradients, computing the heatmap).
*   `src/pipeline/phase06_visualize_gradcam/config.yaml`
    *   **Role**: Specifies the target layer of the model to analyze (usually the last convolutional layer).

## Technology and Architecture

*   **Hooks**: Uses PyTorch hooks (`register_forward_hook`, `register_backward_hook`) to access intermediate feature maps and gradients without modifying the model definition.
*   **OpenCV**: Used to resize the generated heatmap to match the original image size and apply colormaps.
*   **HTML Reporting**: Generates a static HTML report to easily view and compare Grad-CAM results.

## TODO

*   No specific TODOs.

## Progress

*   **Stable**: Logic is functional.

## Key Lessons and Insights

*   Choosing the correct target layer (`features[-1]`) is crucial for getting meaningful semantic heatmaps. Too early, and it's just edges; too late, and spatial information is lost.
