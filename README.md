# AI Ophthalmology - Glaucoma Detection Pipeline

## Purpose

This project implements a modular, Python-based command-line application for identifying Glaucoma from retinal images using the "Glaucoma Dataset - EyePACS-AIROGS Light V2". It utilizes a MobileNetV3 model and follows a defined multi-phase pipeline.

## Disclaimer

**This is an experimental implementation for learning and demonstration purposes only. It should NOT be used for clinical diagnosis, patient management, or any medical decision-making.**

## Explainable AI with Grad-CAM

This pipeline includes Grad-CAM visualizations (Phase 06) to show which parts of the retina image the model focuses on for its diagnosis. Below is an example comparing the original image with the Grad-CAM overlay highlighting influential regions for an (incorrect) prediction:

<table>
  <tr>
    <td align="center">Original Image</td>
    <td align="center">Grad-CAM Overlay</td>
  </tr>
  <tr>
    <td valign="top"><img src="src/assets/grad-cam-zoom-original.png" alt="Original Fundus Image Example" width="400"></td>
    <td valign="top"><img src="src/assets/grad-cam-zoom-overlay.png" alt="Grad-CAM Overlay Example" width="400"></td>
  </tr>
</table>

--- 

## Workflow

The system operates through a sequential multi-phase pipeline:

1.  **Phase 01: Ingest:** Validates the presence of the raw dataset directory.
2.  **Phase 02: Conform to ImageFolder:** Validates the raw data's `ImageFolder` structure and creates a run-specific **copy** of the dataset in the standard format.
3.  **Phase 03: Train Model:** Trains the MobileNetV3 model using the copied `ImageFolder` data and saves both the model weights and metadata.
4.  **Phase 04: Evaluate:** Evaluates the trained model on the copied test set.
5.  **Phase 05: Predict:** Uses a trained model to predict Glaucoma presence (NRG/RG) on new JPEG images.
6.  **Phase 06: Visualize GradCAM:** Generates visual explanations showing which regions of retinal images most influenced the model's decisions.

Each phase can be executed independently via the command-line interface.

## Data Flow

*   **Raw Data:** Expected at `data/01_raw/eyepac-light-v2-512-jpg/`. This directory should contain `train/`, `validation/`, and `test/` subdirectories, each with `NRG/` (Non-Referable Glaucoma) and `RG/` (Referable Glaucoma) subdirectories containing the JPEG images. This structure is the source of truth for labels and splits.
*   **Conformed Data:** During Phase 02, a run-specific copy of the raw data is created in `data/02_conformed_to_imagefolder/{run_id}/`, maintaining the `ImageFolder` structure. This copied data is used for training and evaluation.
*   **Trained Models:** Models are saved in `data/03_train_model/{run_id}/mobilenetv3/` along with model metadata.
*   **Evaluation Results:** Metrics and test predictions are stored in `data/04_evaluate/{run_id}/`.
*   **Prediction:**
    *   Default input images should be placed in `data/05_predict/inference_input_default_dataset/`.
    *   Custom batch inputs can be placed in `data/05_predict/{run_id}/{batch_id}/inference_input/`.
    *   Prediction outputs are saved in `data/05_predict/{run_id}/{batch_id}/inference_output/`.
    *   If no custom batch input exists, the system will use the default dataset.
*   **GradCAM Visualizations:** Grad-CAM heatmaps showing which regions of the images influenced the model's decisions are saved in `data/06_visualize_gradcam/{run_id}/`. This includes:
    *   Original images, heatmaps, and overlays saved as PNG files
    *   Interactive HTML reports with filtering options
    *   For evaluation results: separate directories for correct and incorrect predictions
*   **Logs:** Logs are stored in `logs/`.

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

## Results Summary

The trained MobileNetV3 model demonstrates strong performance in detecting referable glaucoma (RG) versus non-referable glaucoma (NRG) on the test set. It achieved an overall accuracy of **93.6%**, with high sensitivity (Recall) of **93.8%** (correctly identifying actual RG cases) and high precision of **93.5%** (correctly identifying predicted RG cases). The Area Under the ROC Curve (AUC) was **0.981**, indicating excellent discriminative ability between the two classes. These metrics suggest performance comparable to or exceeding that of general ophthalmologists and approaching specialist levels for this specific task and dataset, making it a promising candidate for further investigation as a screening aid (subject to rigorous clinical validation). 
*(Note: Metrics based on `data/04_evaluate/explanation_of_metrics.md`)*

## Visualizations

This pipeline generates several types of visualizations to help understand model behavior:

### Grad-CAM Report (Phase 06)

Phase 06 generates Grad-CAM visualizations highlighting image regions influencing the model's decision. An interactive HTML report allows filtering by class and prediction correctness.

![Grad-CAM Report Screenshot](src/assets/grad-cam-report.png)

### Activation Maximization Report (Phase 07)

Phase 07 generates activation maximization visualizations, showing synthetic patterns that maximally activate specific model features, along with real image examples exhibiting high or low activation for that feature. An interactive HTML report with tabs allows comparison.

![Activation Maximization Report Screenshot](src/assets/activation-maximization-report.png)

*Example Activation Maximization Detail:*

![Activation Maximization Zoom](src/assets/activation-maximization-zoom.png)

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
    python main.py evaluate --run-id run_1 # Evaluates model run_1 using test data, saves results to data/04_evaluate/run_1/
    
    # Example: Run prediction on new images
    # Place your new JPEG images in data/05_predict/inference_input_default_dataset/
    python main.py predict --run-id run_1 # Uses model run_1, saves results to data/05_predict/run_1/inference_output/
    
    # Example: Generate GradCAM visualizations
    python main.py gradcam --run-id run_1  # Visualize evaluation results
    python main.py gradcam --run-id run_1 --filter incorrect  # Only show incorrect predictions
    python main.py gradcam --run-id run_1 --source predict  # Visualize prediction results
    ```
    Use `python main.py --help` to see all available commands and options.
