# AI Ophthalmology System Prompt - Glaucoma Detection Pipeline

## 1. Project Goal

Develop a modular, Python-based command-line application for **identifying Glaucoma** from retinal images. This version uses the **"Glaucoma Dataset - EyePACS-AIROGS Light V2"** dataset (expected at `data/01_raw/eyepac-light-v2-512-jpg/`, assumed to be pre-structured in `ImageFolder` format with `NRG`/`RG` class subdirectories within `train/`, `validation/`, and `test/` folders) and processes JPEG images directly through a multi-phase pipeline. The pipeline relies on the **directory structure** of the raw data (`ImageFolder` convention) to determine class labels and data splits. 

## 2. Core Requirements

*   **Input Handling:** Ingest the AIROGS dataset located at `data/01_raw/eyepac-light-v2-512-jpg/`, assuming it follows the `ImageFolder` convention (e.g., `train/NRG/`, `train/RG/`, `validation/NRG/`, etc.).
*   **Data Management:**
    *   Organize data for a specific run (`run_id`).
    *   Validate the raw data's `ImageFolder` structure.
    *   Create a run-specific copy of the data in `data/02_conformed_to_imagefolder/{run_id}/` maintaining the `ImageFolder` structure (`train/`, `validation/`, `test/` directories with `NRG`/`RG` subdirectories). This structure will contain **copies** of the original images.
*   **Modeling (MobileNetV3):**
    *   Implement `torchvision.models.mobilenet_v3_large` (pre-trained, `weights=MobileNet_V3_Large_Weights.DEFAULT`).
    *   Modify the final classifier layer for 2 output classes (`model.classifier[-1] = torch.nn.Linear(1280, 2)`).
*   **Training:** Utilize supervised learning based on the `ImageFolder` structure. Implement the specific optimizer (`Adam`, lr=0.001), scheduler (`StepLR`, step_size=3, gamma=0.1), and loss function (`CrossEntropyLoss`) derived from the reference notebook.
*   **Evaluation:** Implement and report relevant performance metrics (Accuracy, Precision, Recall, F1, AUC) for the 2-class Glaucoma detection task (NRG/RG).
*   **Prediction:** Apply the trained MobileNetV3 model to new, unseen JPEG images, predicting NRG/RG classes.
*   **Interface:** Provide a Command Line Interface (CLI) to execute each stage of the pipeline individually.
*   **Abstraction & Workflow:** Use the directory-based structure, with phases handling JPEGs and the `ImageFolder` convention.
*   **Reproducibility:** Ensure experiments (runs) are self-contained using the defined structure.
*   **Documentation & Version Control:** Maintain `README.md`. Ensure `.gitignore` excludes the `data/` directory (except potentially a placeholder file like `.gitkeep`).
*   **Technology Stack:** Python, **PyTorch (`torch`, `torchvision`)**, `click`/`argparse`, `pyyaml` (for logging config), `scikit-learn` (metrics), `pandas` (CSV handling), `Pillow` (image loading).

## 3. System Architecture & Workflow (JPEG Pipeline)

The system follows a sequential multi-phase pipeline for direct JPEG handling:

1.  **Phase 01: Raw Input Ingestion:** Validates the existence of the main raw dataset directory.
2.  **Phase 02: Conform to ImageFolder Format:** Validates the raw data's `ImageFolder` structure and creates a run-specific **copy** of the dataset in the standard format.
3.  **Phase 03: Model Training:** Trains MobileNetV3 using the copied `ImageFolder` data from Phase 02.
4.  **Phase 04: Model Evaluation:** Evaluates the trained model on the test set copy from Phase 02.
5.  **Phase 05: Prediction:** Uses the trained model to predict on new JPEGs.

*(Note: Phase numbering corresponds to the directory structure, e.g., `src/pipeline/01_ingest/`, `src/pipeline/02_conformed_to_imagefolder/`)*.

## 4. Detailed Phase Descriptions (Glaucoma Focus)

*   **Phase 01: Ingest (`src/pipeline/01_ingest/main.py`)**
    *   **Purpose:** Validate the presence of the raw AIROGS dataset directory.
    *   **Inputs:** Expected path `data/01_raw/eyepac-light-v2-512-jpg/`.
    *   **Outputs:** Confirmation log messages.
    *   **Key Tasks:** Check if the specified directory exists. Log confirmation or errors. (Detailed validation of the internal structure is deferred to Phase 02).
*   **Phase 02: Conform to ImageFolder Format (`src/pipeline/02_conformed_to_imagefolder/main.py`)**
    *   **Purpose:** Validate the raw data's `ImageFolder` structure and create a run-specific **copy** of the dataset suitable for `torchvision.datasets.ImageFolder`.
    *   **Inputs:** `data/01_raw/eyepac-light-v2-512-jpg/` (base path containing pre-structured `train/`, `validation/`, `test/` folders with `NRG`/`RG` subfolders), `run_id`.
    *   **Outputs:** `data/02_conformed_to_imagefolder/{run_id}/` containing `train/`, `validation/`, and `test/` subdirectories. Each of these contains `NRG/` and `RG/` subdirectories holding **copies** of the corresponding original JPEG image files from the raw data directory.
    *   **Key Tasks:** Validate that the input directory contains `train`, `validation`, and `test` subdirectories. Validate that each of these contains `NRG` and `RG` subdirectories. Perform a basic check that these class directories contain image files (e.g., `.jpg`). Create the necessary output directory structure under `data/02_conformed_to_imagefolder/{run_id}/`. Iterate through the validated raw structure (`train/NRG`, `train/RG`, etc.) and **copy** each image file to the corresponding location in the output directory structure. Log progress and any errors (e.g., missing directories, non-image files found).
*   **Phase 03: Train (`src/pipeline/03_train/main.py`)**
    *   **Purpose:** Train the MobileNetV3 model for Glaucoma detection using the **copied** `ImageFolder` from Phase 02.
    *   **Inputs:** `data/02_conformed_to_imagefolder/{run_id}/` (ImageFolder structure), merged configuration from global `config.yaml` and `src/pipeline/03_train/config.yaml` (phase-specific settings override global), `run_id`.
    *   **Outputs:** Trained model artifacts (best model based on validation) in `models/mobilenetv3/{run_id}/`, training logs in `logs/training/mobilenetv3/{run_id}/`.
    *   **Key Tasks:** Load data using `torchvision.datasets.ImageFolder` from `data/02_conformed_to_imagefolder/{run_id}/train` and `data/02_conformed_to_imagefolder/{run_id}/validation` via `DataLoader` (using batch size from merged config). Apply transforms defined in merged configuration (e.g., Resize, AddNoise, ColorJitter, Flips for train; Resize, Normalize for val/test). Define the MobileNetV3 model (`mobilenet_v3_large`, pre-trained) with modified classifier (`out_features=2`). Implement training loop using `CrossEntropyLoss`, `Adam` optimizer (using LR from merged config), `StepLR` scheduler (using step/gamma from merged config). Save best model checkpoint (based on validation accuracy) and logs. (Note: Final evaluation on the test set using the saved best model is performed separately in Phase 04).
*   **Phase 04: Evaluate (`src/pipeline/04_evaluate/main.py`)**
    *   **Purpose:** Evaluate the **best saved** trained MobileNetV3 model (artifact from Phase 03) using the **copied** test set from Phase 02.
    *   **Inputs:** `models/mobilenetv3/{run_id}/` (path to trained model), `data/02_conformed_to_imagefolder/{run_id}/test/` (ImageFolder structure), `run_id`.
    *   **Outputs:** Evaluation metrics/results saved in `results/mobilenetv3/{run_id}/`.
    *   **Key Tasks:** Load test data using `ImageFolder` and `DataLoader` (with validation transforms) from `data/02_conformed_to_imagefolder/{run_id}/test/`. Load the specified trained model. Perform inference. Calculate metrics (Accuracy, Precision, Recall, F1, AUC for NRG/RG classes). Save results. Log.
*   **Phase 05: Predict (`src/pipeline/05_predict/main.py`)**
    *   **Purpose:** Apply a trained MobileNetV3 model to new JPEG images.
    *   **Inputs:** `models/mobilenetv3/{run_id}/` (path to trained model), `data/04_inference_input/{batch_id}/` (containing JPEGs), `run_id`, `batch_id`.
    *   **Outputs:** Predictions saved in `data/05_inference_output/{batch_id}/`.
    *   **Key Tasks:** Load the specified trained model. Load input JPEGs (applying validation transforms). Perform inference. Save predictions (e.g., CSV mapping image filename to predicted class NRG/RG and probabilities). Log.

## 5. Proposed Directory Structure (Glaucoma Focus, No DICOM)

```
/AI-Opthalmology-EyePACS-AIROGS-Light-V2-Glaucoma/
├── .gitignore           # Should ignore data/*, logs/*, results/*, models/* etc.
├── README.md
├── requirements.txt       # Python dependencies
├── main.py              # CLI entry point
├── config.yaml          # Global configuration (e.g., base paths, shared settings)
├── logger_config.yaml   # Logging configuration
├── data/                  # Root data directory (IGNORED BY GIT)
│   ├── .gitkeep           # Placeholder to keep dir in git repo
│   ├── 01_raw/eyepac-light-v2-512-jpg/ # ASSUMED LOCATION of dataset (pre-structured JPEGs in ImageFolder format)
│   │   ├── train/ # Contains JPEGs, structure IS the source of truth
│   │   │   ├── NRG/ # Example subdirectory
│   │   │   └── RG/  # Example subdirectory
│   │   ├── validation/
│   │   │   ├── NRG/
│   │   │   └── RG/
│   │   ├── test/
│   │   │   ├── NRG/
│   │   │   └── RG/
│   ├── 02_conformed_to_imagefolder/{run_id}/ # ImageFolder structure (populated with COPIES from raw data)
│   │   ├── train/
│   │   │   ├── NRG/ # Contains COPIED images labeled NRG for training
│   │   │   └── RG/
│   │   ├── validation/
│   │   │   ├── NRG/
│   │   │   └── RG/
│   │   └── test/
│   │       ├── NRG/
│   │       └── RG/
│   ├── 04_inference_input/{batch_id}/ # Input JPEGs for prediction
│   └── 05_inference_output/{batch_id}/ # Output predictions
├── models/                # Trained model artifacts (IGNORED BY GIT)
│   └── mobilenetv3/{run_id}/
├── results/               # Evaluation results (IGNORED BY GIT)
│   └── mobilenetv3/{run_id}/
├── logs/                  # Log files (IGNORED BY GIT)
│   ├── debug.log
│   └── training/
│       └── mobilenetv3/{run_id}/
└── src/                   # Source code (CHECKED INTO GIT)
    ├── __init__.py
    ├── cli.py             # CLI implementation
    ├── config_loader.py   # Handles loading global and phase-specific config.yaml files
    ├── pipeline/          # Core processing phase modules
    │   ├── __init__.py
    │   ├── 01_ingest/
    │   │   ├── __init__.py
    │   │   └── main.py    # May not need config
    │   ├── 02_conformed_to_imagefolder/ # Renamed Phase 02
    │   │   ├── __init__.py
    │   │   └── main.py    # May not need config
    │   ├── 03_train/
    │   │   ├── __init__.py
    │   │   ├── main.py
    │   │   └── config.yaml # Phase-specific training config (overrides global)
    │   ├── 04_evaluate/
    │   │   ├── __init__.py
    │   │   ├── main.py
    │   │   └── config.yaml # Phase-specific evaluation config (overrides global)
    │   └── 05_predict/
    │       ├── __init__.py
    │       ├── main.py
    │       └── config.yaml # Phase-specific prediction config (overrides global)
    ├── models/            # Model architecture definition modules
    │   ├── __init__.py
    │   ├── base_model.py
    │   └── mobilenetv3/
    │       ├── __init__.py
    │       └── model.py   # Function to get/modify mobilenet
    ├── data_handling/     # Data loading, datasets, transforms
    │   ├── __init__.py
    │   ├── datasets.py    # Setup for ImageFolder, transforms
    │   └── transforms.py  # Custom transforms like AddNoise
    └── utils/             # Shared utilities
        ├── __init__.py
        ├── logger.py      # Central logging setup
        └── file_utils.py  # General file/directory operations
```

## 6. Initial File Creation TODO List (Revised)

*   `.gitignore`: Create/update to ignore `data/`, `logs/`, `results/`, `models/`, `*.pth`, `*.pt`, `__pycache__/`, `*.pyc`, `AIOpthalEyepacsGlaucVenv/`, `.DS_Store`. Add an exception `!data/.gitkeep` if using a placeholder file to keep the `data` directory in the repository.
*   `README.md`: Create a README file that describes:
    *   **Purpose:** What the system does (Glaucoma detection pipeline).
    *   **Disclaimer:** State clearly that this is an experimental implementation for learning purposes and should not be used for clinical diagnosis or decisions.
    *   **Workflow:** Briefly explain the multi-phase pipeline (Ingest validation, Structure, Train, Evaluate, Predict).
    *   **Data Flow:** Describe where data is expected (`data/01_raw/...`), where structured data goes (`data/02_structured/...`), and where outputs like models, results, and logs are stored. Mention that these directories (except potentially `data/.gitkeep`) are ignored by git.
    *   **Code Structure:** Briefly outline the organization under the `src/` directory (pipeline phases, models, data handling, utils).
    *   **Setup & Usage:** Basic instructions on setting up the virtual environment (`AIOpthalEyepacsGlaucVenv`), installing requirements (`requirements.txt`), and running the pipeline phases via the CLI (`main.py`). (Do *not* mention the reference notebook here).
*   `requirements.txt` (ensure `torch`, `torchvision`, `pandas`, `Pillow`, `pyyaml`, `scikit-learn`, `click`/`argparse` are included)
*   `main.py`
*   `config.yaml` # Global configuration file
*   `logger_config.yaml`
*   `data/.gitkeep` (empty file, optional placeholder)
*   `src/__init__.py`
*   `src/cli.py`
*   `src/config_loader.py` # Handles loading/merging global and phase-specific configs
*   `src/pipeline/__init__.py`
*   `src/pipeline/01_ingest/__init__.py`, `main.py`
*   `src/pipeline/02_conformed_to_imagefolder/__init__.py`, `main.py` # Renamed Phase 02
*   `src/pipeline/03_train/__init__.py`, `main.py`, `config.yaml` # Phase-specific config
*   `src/pipeline/04_evaluate/__init__.py`, `main.py`, `config.yaml` # Phase-specific config
*   `src/pipeline/05_predict/__init__.py`, `main.py`, `config.yaml` # Phase-specific config
*   `src/models/__init__.py`
*   `src/models/base_model.py`
*   `src/models/mobilenetv3/__init__.py`, `model.py`
*   `src/data_handling/__init__.py`
*   `src/data_handling/datasets.py` (Focus on ImageFolder setup)
*   `src/data_handling/transforms.py` (Implement AddNoise here)
*   `src/utils/__init__.py`
*   `src/utils/logger.py`
*   `src/utils/file_utils.py`
*   Create top-level directories (if not done by git): `data/`, `models/`, `results/`, `logs/`.

## 7. Implementation Guidelines (Glaucoma Focus)

*   **Dataset:** Input data is located at **`data/01_raw/eyepac-light-v2-512-jpg/`** and is assumed to follow the `ImageFolder` convention (e.g., `train/NRG/`, `train/RG/`, etc.). The **directory structure itself** defines the split (`train`/`validation`/`test`) and class (`NRG`/`RG`). Phase 02 validates this structure and **copies** the files into `data/02_conformed_to_imagefolder/{run_id}/`. Subsequent phases (Train, Evaluate) consume data from this copied directory.
*   **Model (MobileNetV3):**
    *   Use `torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)`.
    *   Modify the classifier's last layer: `model.classifier[-1] = torch.nn.Linear(1280, 2)`.
    *   Loss Function: `torch.nn.CrossEntropyLoss()`.
    *   Optimizer: `torch.optim.Adam(model.parameters(), lr=0.001)`.
    *   Scheduler: `torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)`.
*   **Image Handling & Transforms:**
    *   Resize images to 512x512.
    *   Normalize using ImageNet `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`.
    *   Implement custom `AddNoise` transform in `src/data_handling/transforms.py`, taking `noise_level` from the merged configuration.
    *   Apply training transforms: `Resize(512)`, `AddNoise` (using configured level), `ColorJitter` (using configured parameters), `RandomHorizontalFlip()`, `RandomVerticalFlip()`, `ToTensor()`, `Normalize(...)`. Parameters for transforms like `AddNoise` and `ColorJitter` should be read from the merged configuration.
    *   Apply validation/test transforms: `Resize(512)`, `ToTensor()`, `Normalize(...)`.
    *   Use `torchvision.datasets.ImageFolder` in `src/data_handling/datasets.py` or the training script to load data from the **copied structure created by Phase 02** (`data/02_conformed_to_imagefolder/{run_id}/`).
*   **Libraries:** Ensure `requirements.txt` includes `torch`, `torchvision`, `pandas`, `Pillow`, `pyyaml`, `scikit-learn`, `click` (or `argparse`).
*   **Logging:** Use the configured logger (`src/utils/logger.py` & `logger_config.yaml`) as previously defined.
*   **Modularity:** Maintain the CLI structure and independent phase execution.
*   **Configuration:** Use a hierarchical configuration approach. Load settings from the global `config.yaml` first, then load the phase-specific `config.yaml` (e.g., `src/pipeline/03_train/config.yaml`) if it exists, allowing phase-specific settings to override global ones. The `src/config_loader.py` utility will handle this merging logic. Parameters like learning rate, epochs, batch size, optimizer settings, scheduler settings, and transform parameters should be managed this way. CLI arguments can potentially override specific merged config values (e.g., for `run_id`).
*   **Run/Batch ID Generation:** The `run_id` for training/evaluation and `batch_id` for prediction should be determined automatically by the CLI. When initiating a run, the application should scan the relevant output directories (e.g., `data/02_structured/`, `models/mobilenetv3/` for `run_id`; `data/04_inference_input/`, `data/05_inference_output/` for `batch_id`), find the highest existing numerical ID (e.g., `run_99`), and use the next sequential number (e.g., `run_100`) for the current operation. If no previous runs exist, start with `run_1` or `batch_1`.
*   **Code Quality:** Maintain type hinting, docstrings, etc.
*   **Error Handling:** Implement robust error handling and logging, ensuring pipeline phases fail gracefully if prerequisites are not met (e.g., Phase 02 requires valid raw data structure, Phase 03 requires successful Phase 02 output).
*   **Data Loading (`src/data_handling/datasets.py`):** Focus on setting up `ImageFolder` (using data from Phase 02's output directory `data/02_conformed_to_imagefolder/{run_id}/`) and applying the correct `torchvision.transforms.Compose` pipelines for train and validation/test sets.

## 8. Implementation Priorities and Limitations

The following points provide guidance on implementation priorities and deliberate limitations for this project:

*   **Error Handling:** While robust error handling should be incorporated into the code implementation, extensive error handling documentation is not required in this PROMPT document. Implement practical error handling within the code itself.

*   **Training Process:**
    *   **DO NOT** implement checkpointing and training resumption functionality.
    *   **DO NOT** implement complex early stopping mechanisms.
    *   Save only the best model based on validation accuracy (as in the reference notebook).

*   **Hyperparameter Management:**
    *   **DO NOT** build a dedicated hyperparameter tuning framework.
    *   Use a simple configuration system that naturally captures hyperparameters as they appear in the code.
    *   **DO NOT** add structure solely for the purpose of hyperparameter experimentation or comparison.

*   **Evaluation:**
    *   Keep evaluation focused on basic metrics for binary glaucoma classification.
    *   **DO NOT** implement complex visualization of results or model interpretability features.

*   **Performance:**
    *   **DO NOT** focus on performance optimization.
    *   The system will run on a local MacBook without GPU acceleration requirements.
    *   **DO NOT** implement mixed precision training or other performance optimizations.

*   **Deployment:**
    *   **DO NOT** implement deployment-related features such as model export to ONNX.
    *   **DO NOT** add containerization or production serving capabilities.
    *   This is a local development implementation without production deployment considerations.

These limitations are intentional to maintain focus on the core functionality of the glaucoma detection pipeline. Future extensions could address these areas if needed.
