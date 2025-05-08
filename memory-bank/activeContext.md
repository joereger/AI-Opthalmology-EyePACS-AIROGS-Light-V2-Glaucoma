# Active Context: AI Ophthalmology - Glaucoma Detection Pipeline

## Current Work Focus (as of 2025-05-08 ~11:15 AM)

-   Initializing the Memory Bank documentation system as requested.
-   Previously completed tasks involved:
    -   Scaffolding Phase 07 (Activation Maximization), including `main.py`, `utils.py`, `config.yaml`, and `html_report_generator.py`.
    -   Integrating Phase 07 into the CLI (`cli.py`).
    -   Debugging numerous runtime errors (ImportErrors, NameErrors, TypeErrors, incorrect logic) encountered while trying to run the initial Phase 07 scaffolding.
    -   Refactoring Phase 07 code to correctly handle `run_id`, dynamic data paths, model metadata usage, and report layout based on Phase 06 examples.
    -   Updating `README.md` with a results summary and visualization screenshots.

## Recent Changes / Decisions

-   **Memory Bank:** Adopted the Memory Bank structure for context management.
-   **Phase 07 Output:** Corrected logic to place Phase 07 output within the corresponding `run_id` directory (e.g., `data/07.../run_1/`) instead of creating new sequential run directories.
-   **Phase 07 Config:** Removed hardcoded `data_dir` and `image_size`; added `model_artifact_subdir`.
-   **Phase 07 Data Loading:** Refactored `utils.py` to dynamically determine validation data path based on `run_id` and global config; corrected positive class filtering logic to use index instead of name.
-   **Phase 07 Dependencies:** Added `tqdm` to `requirements.txt`. Added `ImageFolderWithPaths` class to `datasets.py`. Added `shutil` import to `utils.py`.
-   **Phase 07 Layer Parsing:** Updated `get_target_layer` in `utils.py` to handle nested indexing (e.g., `features[-1][0]`).
-   **Phase 07 Report Layout:** Refactored HTML generation into `html_report_generator.py` and implemented a tabbed layout for examples, similar to Phase 06.
-   **README Update:** Added Grad-CAM example near top, added Results Summary, added Visualizations section with screenshots.

## Next Steps

-   Complete the initialization of the Memory Bank by creating `progress.md`.
-   Await further instructions or tasks from the user. The core logic for Phase 07's `perform_activation_maximization` and detailed report generation in `html_report_generator.py` remains largely as placeholders and requires implementation.

## Important Patterns / Preferences

-   Pipeline phases should operate within the context of a specific `run_id`. Output for a phase analyzing `run_X` should generally go into a subdirectory related to `run_X` within that phase's output folder (e.g., `data/07.../run_X/`).
-   Avoid hardcoding paths or parameters that can be derived from configuration or model metadata.
-   Leverage existing patterns from other phases (e.g., Phase 06's HTML report structure) for consistency.
-   Ensure imports are correct and necessary dependencies are listed in `requirements.txt`.
-   Verify code against actual file contents to avoid errors based on stale context.

## Learnings / Insights

-   Incremental code generation without immediate verification of referenced files/functions can lead to significant integration errors. Re-reading files and confirming interfaces is crucial.
-   Careful attention to variable scope and how configuration (global vs. phase-specific) is passed between modules is necessary.
-   Understanding the established data flow and run ID conventions across pipeline phases is essential for adding new phases correctly.
