# Component: Phase 02 - Conform

## Description

This phase is critical for data standardization. It takes the ingested data and restructures it into the standard PyTorch `ImageFolder` format (`train/CLASS/img.jpg`, `validation/CLASS/img.jpg`, etc.).

This ensures that all subsequent phases (training, evaluation, prediction) can use standard data loading techniques without needing to know the idiosyncrasies of the original raw data structure.

## Key Files

*   `src/pipeline/phase02_conformed_to_imagefolder/main.py`
    *   **Role**: The main script that iterates through raw data and copies/moves it into the structured format under `data/02_conformed_to_imagefolder/`.

## Technology and Architecture

*   **Structure**: Creates `train` and `validation` subdirectories, and within those, class-specific directories (e.g., `RG`, `NRG`).
*   **Run Isolation**: Creates a copy of the dataset for the specific `run_id`, ensuring that if data changes later, the run's input remains preserved (storage permitting).

## TODO

*   No specific TODOs.

## Progress

*   **Stable**: No recent changes reported.

## Key Lessons and Insights

*   Investing time in a "Conform" phase simplifies every downstream phase, as they can all assume a guaranteed directory structure.
