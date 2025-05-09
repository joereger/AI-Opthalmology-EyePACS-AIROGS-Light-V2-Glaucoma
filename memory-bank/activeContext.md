# Active Context: AI Ophthalmology - Glaucoma Detection Pipeline

## Current Work Focus (as of 2025-05-08 ~11:28 AM)

-   Updating Memory Bank files (`progress.md`, `activeContext.md`) after re-reading Phase 07 code and confirming its current implementation status.
-   Previously completed tasks involved:
    -   Initializing the Memory Bank system (creating core files).
    -   Scaffolding Phase 07 (Activation Maximization).
    -   Integrating Phase 07 into the CLI.
    -   Debugging runtime errors in Phase 07 setup/integration code.
    -   Refactoring Phase 07 for correct `run_id` handling, dynamic paths, metadata usage, and HTML report layout/structure.
    -   Updating `README.md`.

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

-   Await further instructions or tasks from the user. 
-   The primary remaining task for Phase 07 is implementing the core optimization logic within the `perform_activation_maximization` function in `utils.py`.

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
