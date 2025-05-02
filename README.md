# AI Ophthalmology - Glaucoma Detection Pipeline

## Purpose

This project implements a modular, Python-based command-line application for identifying Glaucoma from retinal images using the "Glaucoma Dataset - EyePACS-AIROGS Light V2". It utilizes a MobileNetV3 model and follows a defined multi-phase pipeline.

## Disclaimer

**This is an experimental implementation for learning and demonstration purposes only. It should NOT be used for clinical diagnosis, patient management, or any medical decision-making.**

## Workflow

The system operates through a sequential multi-phase pipeline:

1.  **Phase 01: Ingest:** Validates the presence of the raw dataset directory.
2.  **Phase 02: Conform to ImageFolder:** Validates the raw data's `ImageFolder` structure and creates a run-specific **copy** of the dataset in the standard format.
3.  **Phase 03: Train Model:** Trains the MobileNetV3 model using the copied `ImageFolder` data and saves both the model weights and metadata.
4.  **Phase 04: Evaluate:** Evaluates the trained model on the copied test set.
5.  **Phase 05: Predict:** Uses a trained model to predict Glaucoma presence (NRG/RG) on new JPEG images.

Each phase can be executed independently via the command-line interface.

## Data Flow

*   **Raw Data:** Expected at `data/01_raw/eyepac-light-v2-512-jpg/`. This directory should contain `train/`, `validation/`, and `test/` subdirectories, each with `NRG/` (Non-Referable Glaucoma) and `RG/` (Referable Glaucoma) subdirectories containing the JPEG images. This structure is the source of truth for labels and splits.
*   **Conformed Data:** During Phase 02, a run-specific copy of the raw data is created in `data/02_conformed_to_imagefolder/{run_id}/`, maintaining the `ImageFolder` structure. This copied data is used for training and evaluation.
*   **Trained Models:** Models are saved in `data/03_train_model/{run_id}/mobilenetv3/` along with model metadata.
*   **Inference Input:** New images for prediction should be placed in `data/04_inference_input/{batch_id}/`.
*   **Outputs:**
    *   Evaluation results are saved in `results/mobilenetv3/{run_id}/`.
    *   Prediction outputs are saved in `data/05_inference_output/{batch_id}/`.
    *   Logs are stored in `logs/`.

**Note:** The `data/`, `models/`, `results/`, and `logs/` directories are ignored by Git (except for `data/.gitkeep`). Ensure the raw data is obtained and placed correctly.

## Code Structure

The source code is organized under the `src/` directory:

*   `main.py`: Main CLI entry point.
*   `cli.py`: Implements the command-line interface using `click`.
*   `config_loader.py`: Handles loading and merging configuration files.
*   `pipeline/`: Contains modules for each pipeline phase (01_ingest, 02_conformed_to_imagefolder, 03_train, 04_evaluate, 05_predict).
*   `models/`: Defines the model architecture (MobileNetV3 modification).
*   `data_handling/`: Includes modules for dataset loading (`ImageFolder`) and image transformations.
*   `utils/`: Contains shared utilities for logging, file operations, etc.
*   `config.yaml`: Global configuration settings.
*   `logger_config.yaml`: Logging configuration.

## Model Metadata

Each trained model is accompanied by a `model_metadata.json` file that contains critical information about the model:

* **Model Architecture**: Architecture type (e.g., "MobileNetV3") and variant (e.g., "Large")
* **Model Properties**: Number of classes, input size requirements, and normalization parameters
* **Training Details**: Training date, optimizer, learning rate, scheduler settings, and other hyperparameters

This metadata serves several purposes:
1. **Documentation**: Self-documenting models with all information needed for reproducibility
2. **Consistency**: Ensures evaluation and prediction phases use the same parameters as training
3. **Extensibility**: Makes it easier to add new model architectures in the future
4. **Portability**: All information needed to use the model travels with it

The metadata is generated directly from the model class through its `get_metadata()` method, allowing each model type to provide its specific characteristics.

## Setup & Usage

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd AI-Opthalmology-EyePACS-AIROGS-Light-V2-Glaucoma
    ```
2.  **Set up Python Environment:** It's recommended to use a virtual environment.
    ```bash
    python3 -m venv AIOpthalEyepacsGlaucVenv
    source AIOpthalEyepacsGlaucVenv/bin/activate
    # On Windows use: .\AIOpthalEyepacsGlaucVenv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Obtain Data:** Download the "Glaucoma Dataset - EyePACS-AIROGS Light V2" and place its contents (the `eyepac-light-v2-512-jpg` folder containing `train`, `validation`, `test` subdirectories) into the `data/01_raw/` directory. The final path should be `data/01_raw/eyepac-light-v2-512-jpg/`.
5.  **Run Pipeline Phases:** Use the main CLI script. A `run_id` will be generated automatically if not provided.
    ```bash
    # Example: Run the full pipeline sequentially
    python main.py ingest
    python main.py conform # Creates data/02_conformed_to_imagefolder/run_1/
    python main.py train --run-id run_1 # Trains using run_1 data, saves model to data/03_train_model/run_1/mobilenetv3/
    python main.py evaluate --run-id run_1 # Evaluates model run_1 using test data
    
    # Example: Run prediction on new images (assuming images are in data/04_inference_input/batch_1/)
    # Place your new JPEG images in data/04_inference_input/batch_1/
    python main.py predict --run-id run_1 --batch-id batch_1 # Uses model run_1
    ```
    Use `python main.py --help` to see all available commands and options.
