# Technical Context: AI Ophthalmology - Glaucoma Detection Pipeline

## Core Technologies

-   **Language:** Python 3.12 (as indicated by environment)
-   **Deep Learning Framework:** PyTorch (`torch`, `torchvision`)
-   **CLI Framework:** `click`
-   **Configuration:** YAML (`PyYAML`)
-   **Data Handling:** `pandas`, `numpy`, `Pillow` (PIL Fork)
-   **Image Processing:** `opencv-python` (cv2), `matplotlib` (for colormaps)
-   **Utilities:** `tqdm` (progress bars)
-   **Evaluation Metrics:** `scikit-learn`

## Development Setup

-   **Environment:** Assumed to be macOS based on user's file paths. A Python virtual environment (`AIOpthalEyepacsGlaucVenv`) is used.
-   **Dependencies:** Managed via `requirements.txt` and installed using `pip`.
-   **Logging:** Configured via `logger_config.yaml` using Python's standard `logging` module. Setup is handled centrally in `src/cli.py`.

## Technical Constraints & Assumptions

-   **Input Data Format:** Assumes raw data follows the `ImageFolder` structure (`train/CLASS/img.jpg`, `validation/CLASS/img.jpg`, etc.) and images are JPEGs. Phase 02 conforms data to this structure under `data/02_conformed_to_imagefolder/`.
-   **Model Architecture:** Currently hardcoded to use MobileNetV3 Large via `torchvision.models`. The `src/models/mobilenetv3/model.py` wraps this and adds metadata capabilities.
-   **Compute:** Training and inference can leverage CUDA GPUs (NVIDIA) or MPS (Apple Silicon) if available, falling back to CPU. Device selection logic exists in phase main scripts.
-   **Run Identification:** Pipeline runs are identified by `run_id` (e.g., `run_1`), which is used to organize related data across different phase output directories. The CLI determines the latest or next `run_id` using helper functions in `src/utils/file_utils.py`.
-   **Prediction Input:** Phase 05 looks for input images in specific directories (`data/05_predict/{run_id}/{batch_id}/inference_input/` or `data/05_predict/inference_input_default_dataset/`).

## Tool Usage Patterns

-   **CLI Execution:** The primary interaction method is via `python main.py <command> [options]`.
-   **Configuration Files:** `config.yaml` (global) and phase-specific `config.yaml` files control parameters like learning rates, paths, batch sizes, etc.
-   **Virtual Environment:** Standard Python practice for dependency management.
