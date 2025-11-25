# Component: Phase 07 - Activation Maximization

## Description

This phase implements Activation Maximization, a technique to visualize what a specific filter (or unit) in the model is "looking for." It synthesizes an input image that maximally activates a specific target neuron, effectively "dreaming" the ideal input for that feature.

It also identifies real images from the validation set that highly activate the same units, providing a "Synthetic vs. Real" comparison.

## Key Files

*   `src/pipeline/phase07_visualize_activation_maximization/main.py`
    *   **Role**: Orchestrator. Loads model, calculates channel importance, triggers optimization, and generates reports.
*   `src/pipeline/phase07_visualize_activation_maximization/utils.py`
    *   **Role**: Contains the core logic: `calculate_importance_scores`, `find_real_examples`, and the (currently placeholder) `perform_activation_maximization`.
*   `src/pipeline/phase07_visualize_activation_maximization/html_report_generator.py`
    *   **Role**: Dedicated module for generating the tabbed HTML report.
*   `src/pipeline/phase07_visualize_activation_maximization/config.yaml`
    *   **Role**: Parameters for optimization (learning rate, iterations, regularization).

## Technology and Architecture

*   **Gradient Ascent**: Unlike training (gradient descent), this process freezes the model weights and updates the *input pixels* to maximize the activation of a target unit.
*   **Regularization**: Essential to produce recognizable images; typically involves total variation (TV) loss or Gaussian smoothing to suppress high-frequency noise.

## TODO

*   **Implement Optimization**: The `perform_activation_maximization` function in `utils.py` is currently a placeholder. It needs the actual gradient ascent loop with regularization.
*   **Mean Image Init**: Implement initialization with a mean image.

## Progress

*   **2025-05-08**: Scaffolding complete.
*   **2025-05-08**: Implemented `calculate_importance_scores` (finding which filters care most about the 'RG' class).
*   **2025-05-08**: Implemented `find_real_examples` (finding top activating real images).
*   **2025-05-08**: Refactored HTML generation to support tabbed layout.
*   **2025-05-08**: Fixed integration bugs (run_id paths, config loading).

## Key Lessons and Insights

*   Activation Maximization is highly sensitive to hyperparameters (learning rate, regularization). Without regularization, the output is often adversarial noise rather than a meaningful image.
