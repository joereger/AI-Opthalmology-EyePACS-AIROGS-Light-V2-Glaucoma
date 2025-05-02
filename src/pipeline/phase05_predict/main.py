import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from PIL import Image
from pathlib import Path
import json
import pandas as pd
from typing import Dict, Any, Optional, List

# Project-specific imports
from src.config_loader import load_config
from src.utils.file_utils import ensure_dir_exists, get_next_batch_id
from src.data_handling.transforms import get_transforms # Need the 'test' transform
from src.models.mobilenetv3.model import get_model

logger = logging.getLogger(__name__)

# Define phase-specific config path
PHASE_CONFIG_PATH = 'src/pipeline/05_predict/config.yaml'
ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg'] # From Phase 02

# Simple Dataset for inference on a folder of images
class InferenceDataset(Dataset):
    def __init__(self, image_dir: Path, transform: Optional[Compose] = None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = sorted([
            p for p in self.image_dir.iterdir() 
            if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        ])
        if not self.image_paths:
             logger.warning(f"No valid image files found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, str(img_path) # Return image tensor and its path
        except Exception as e:
            logger.error(f"Error loading or transforming image {img_path}: {e}", exc_info=True)
            # Return None or raise error? Returning None allows skipping bad images.
            # Need careful handling in the prediction loop.
            # For simplicity, let's return a dummy tensor and path, and log error.
            # This avoids issues with DataLoader collation if None is returned.
            dummy_tensor = torch.zeros((3, 224, 224)) # Placeholder size, adjust if needed
            return dummy_tensor, f"ERROR_LOADING_{img_path.name}"


def predict_on_batch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str] # e.g., ['NRG', 'RG']
) -> pd.DataFrame:
    """
    Performs inference on the images provided by the dataloader.

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader for the inference dataset.
        device: The device to run inference on ('cuda' or 'cpu').
        class_names: List of class names corresponding to model output indices.

    Returns:
        pd.DataFrame: DataFrame with predictions (image_path, predicted_label, probability_NRG, probability_RG).
    """
    model.eval()
    logger.info("Starting inference on the input batch...")

    results = []
    pos_label_index = class_names.index('RG') if 'RG' in class_names else 1 # Assume RG is class 1

    with torch.no_grad():
        for inputs, paths in dataloader:
            # Handle potential errors during loading (dummy tensors/paths)
            valid_indices = [i for i, p in enumerate(paths) if not p.startswith("ERROR_LOADING_")]
            if not valid_indices:
                 logger.warning("Skipping batch due to loading errors for all images.")
                 continue # Skip entire batch if all images failed to load

            # Filter inputs and paths to only include valid ones
            inputs = inputs[valid_indices].to(device)
            paths = [paths[i] for i in valid_indices]

            if inputs.nelement() == 0: # Double check if filtering resulted in empty tensor
                 logger.warning("Skipping batch as no valid images remained after filtering.")
                 continue

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Process results for the valid images in the batch
            probs_cpu = probabilities.cpu().numpy()
            preds_cpu = preds.cpu().numpy()

            for i, img_path_str in enumerate(paths):
                pred_idx = preds_cpu[i]
                pred_label = class_names[pred_idx]
                prob_nrg = probs_cpu[i][0] # Assumes class 0 is NRG
                prob_rg = probs_cpu[i][1]  # Assumes class 1 is RG
                
                results.append({
                    'image_path': Path(img_path_str).name, # Store only filename
                    'predicted_label': pred_label,
                    f'probability_{class_names[0]}': prob_nrg,
                    f'probability_{class_names[1]}': prob_rg
                })

    logger.info(f"Inference complete. Processed {len(results)} images.")
    
    if not results:
        logger.warning("No results generated. Check input data and logs.")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['image_path', 'predicted_label', f'probability_{class_names[0]}', f'probability_{class_names[1]}'])

    return pd.DataFrame(results)


def run_phase(config: Dict[str, Any], run_id: str, batch_id: Optional[str] = None) -> bool:
    """
    Executes Phase 05: Prediction.

    Args:
        config (Dict[str, Any]): The merged configuration dictionary.
        run_id (str): The run ID of the trained model to use for prediction.
        batch_id (Optional[str]): The batch ID for input/output directories. If None, the next available ID is determined.

    Returns:
        bool: True if the phase completes successfully, False otherwise.
    """
    logger.info(f"--- Starting Phase 05: Prediction using model from run_id='{run_id}' ---")
    phase_success = False
    try:
        # --- Determine Batch ID ---
        if batch_id is None:
            batch_id = get_next_batch_id(config, run_id)
            logger.info(f"No batch_id provided, determined next batch_id: {batch_id}")
        else:
            logger.info(f"Using provided batch_id: {batch_id}")

        # --- Configuration & Paths ---
        num_classes = config.get('num_classes', 2)
        model_name = config.get('model_name', 'mobilenetv3')
        
        # Use data/03_train_model directory structure for model loading
        model_base_dir = Path(config.get('train_model_dir', 'data/03_train_model'))
        model_path = model_base_dir / run_id / model_name / 'best_model.pth'
        
        # New directory structure for prediction
        predict_dir = Path(config.get('predict_dir', 'data/05_predict'))
        
        # Custom batch input directory - if it exists
        custom_input_dir = predict_dir / run_id / batch_id / 'inference_input'
        
        # Default input dataset location (fallback)
        predict_default_input_dir = Path(config.get('predict_default_input_dir', 
                                            'data/05_predict/inference_input_default_dataset'))
        
        # Output directory structure
        inference_output_dir = predict_dir / run_id / batch_id / 'inference_output'
        predictions_save_path = inference_output_dir / 'predictions.csv'
        
        # Create output directory
        ensure_dir_exists(inference_output_dir)
        
        # Determine which input directory to use
        if custom_input_dir.exists() and any(custom_input_dir.iterdir()):
            # Use custom batch input if it exists and has files
            inference_input_dir = custom_input_dir
            logger.info(f"Using custom input directory: {inference_input_dir}")
        elif predict_default_input_dir.exists() and any(predict_default_input_dir.iterdir()):
            # If no custom input, use default dataset
            inference_input_dir = predict_default_input_dir
            logger.info(f"No custom input found. Using default dataset: {inference_input_dir}")
        else:
            # No input data found
            logger.error(f"No input data found. Please put images in either: \n"
                        f"1. Custom batch input: {custom_input_dir}\n"
                        f"2. Default dataset: {predict_default_input_dir}")
            return False
        
        # Use batch size from config, default to 1 for inference if not specified
        # Larger batch sizes can speed up inference if resources allow
        batch_size = config.get('batch_size', 1) 
        num_workers = config.get('num_workers', 1)
        pin_memory = config.get('pin_memory', True)

        ensure_dir_exists(inference_output_dir)

        # --- Device Setup ---
        # Check for CUDA (NVIDIA GPU) first, then MPS (Apple Silicon GPU), then fall back to CPU
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon GPU via MPS backend")
        else:
            device = torch.device("cpu")
            logger.warning("No GPU acceleration available. Prediction will run on CPU (slow).")
        
        logger.info(f"Using device: {device}")

        # --- Input Data Validation ---
        if not inference_input_dir.is_dir():
            logger.error(f"Inference input directory not found: {inference_input_dir}")
            logger.error("Please place JPEG images for prediction in this directory.")
            return False

        # --- Model Loading with Metadata ---
        logger.info(f"Loading model from: {model_path}")
        if not model_path.is_file():
            logger.error(f"Model file not found at {model_path}. Cannot perform prediction.")
            return False
        
        # Check for model metadata
        metadata_path = model_path.parent / 'model_metadata.json'
        model_metadata = None
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    model_metadata = json.load(f)
                logger.info(f"Loaded model metadata from {metadata_path}")
                
                # Use metadata to configure model
                if model_metadata:
                    # Log the model details from metadata
                    logger.info(f"Model metadata: Architecture={model_metadata.get('architecture')}, "
                                f"Variant={model_metadata.get('variant')}, "
                                f"Classes={model_metadata.get('num_classes')}, "
                                f"Training date={model_metadata.get('training_date')}")
                    
                    # Use num_classes from metadata to override config
                    if 'num_classes' in model_metadata:
                        num_classes = model_metadata['num_classes']
                        logger.info(f"Using num_classes={num_classes} from metadata")
                        
                    # Use normalization from metadata if specified
                    if 'normalization' in model_metadata:
                        logger.info(f"Using normalization parameters from metadata: "
                                   f"mean={model_metadata['normalization'].get('mean')}, "
                                   f"std={model_metadata['normalization'].get('std')}")
            except Exception as e:
                logger.warning(f"Could not load model metadata from {metadata_path}: {e}. "
                              f"Will use configuration values instead.")
        else:
            logger.warning(f"No model metadata found at {metadata_path}. "
                          f"Using configuration values for model initialization.")
        
        # Create model architecture (using metadata values where available)
        logger.info(f"Initializing model architecture with num_classes={num_classes}")
        model = get_model(num_classes=num_classes, pretrained=False)
        
        # Load weights
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model weights from {model_path}: {e}", exc_info=True)
            return False
        
        model = model.to(device)

        # --- Data Loading & Transforms ---
        logger.info("Loading inference data...")
        # Get the 'test' transforms (usually resize, totensor, normalize)
        transforms_dict = get_transforms(config)
        inference_transform = transforms_dict.get('test') 
        if inference_transform is None:
             logger.error("Could not get 'test' transforms from configuration.")
             return False

        inference_dataset = InferenceDataset(image_dir=inference_input_dir, transform=inference_transform)
        if len(inference_dataset) == 0:
             logger.error(f"No valid images found in input directory: {inference_input_dir}")
             # Still create an empty output file for consistency? Or return False?
             # Let's return False as no prediction happened.
             return False

        inference_dataloader = DataLoader(
            inference_dataset,
            batch_size=batch_size,
            shuffle=False, # Do not shuffle inference data
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # Assuming class order is NRG, RG based on ImageFolder convention
        # TODO: Make class names configurable or load from training run if possible
        class_names = ['NRG', 'RG'] 
        logger.info(f"Loaded {len(inference_dataset)} images for inference.")

        # --- Prediction ---
        predictions_df = predict_on_batch(model, inference_dataloader, device, class_names)

        # --- Save Results ---
        if not predictions_df.empty:
            logger.info(f"Saving predictions to: {predictions_save_path}")
            try:
                predictions_df.to_csv(predictions_save_path, index=False)
                logger.info("Predictions saved successfully.")
                phase_success = True
            except Exception as e:
                logger.error(f"Failed to save predictions CSV: {e}", exc_info=True)
                phase_success = False # Saving is critical here
        else:
             logger.warning("Prediction resulted in an empty DataFrame. No output file saved.")
             # Consider this a failure if images were expected? Or success if 0 valid images?
             # Let's call it success if the process ran, but warn heavily.
             phase_success = True 


    except Exception as e:
        logger.error(f"An unexpected error occurred during Phase 05: {e}", exc_info=True)
        phase_success = False

    if phase_success:
        logger.info("--- Phase 05 Completed Successfully ---")
    else:
        logger.error("--- Phase 05 Failed ---")
        
    return phase_success


# Example of how this might be called from the CLI (in cli.py)
# if __name__ == '__main__':
#     from src.utils.logger import setup_logging
#     setup_logging() 
#
#     # Load merged config (Global + Phase-specific)
#     cfg = load_config(phase_config_path=PHASE_CONFIG_PATH) 
#
#     if cfg:
#         # Assumes Phase 03 ran successfully for 'run_1'
#         test_run_id = "run_1" 
#         # Assumes images exist in data/04_inference_input/batch_1/
#         test_batch_id = "batch_1" # Or None to auto-determine
#
#         # Ensure model exists
#         model_p = Path(cfg['model_dir']) / test_run_id / 'best_model.pth'
#         if not model_p.exists():
#              print(f"ERROR: Model file {model_p} not found. Cannot run Phase 05 example.")
#         else:
#             # Ensure input dir exists and maybe add a dummy image
#             input_dir = Path(cfg['inference_input_dir']) / test_batch_id
#             ensure_dir_exists(input_dir)
#             dummy_image_path = input_dir / "test_image.jpg"
#             if not dummy_image_path.exists():
#                  try:
#                      Image.new('RGB', (100, 100), color = 'blue').save(dummy_image_path)
#                      print(f"Created dummy input image: {dummy_image_path}")
#                  except Exception as e:
#                      print(f"Error creating dummy input image: {e}")
#
#             print(f"\nRunning Phase 05 for run_id='{test_run_id}', batch_id='{test_batch_id}'...")
#             success = run_phase(cfg, run_id=test_run_id, batch_id=test_batch_id)
#             print(f"Phase 05 finished: {'Success' if success else 'Failed'}")
#             # Check data/05_inference_output/batch_1/ for predictions.csv
#
#             # Clean up dummy image and output (optional)
#             # if dummy_image_path.exists(): os.remove(dummy_image_path)
#             # output_csv = Path(cfg['inference_output_dir']) / test_batch_id / 'predictions.csv'
#             # if output_csv.exists(): os.remove(output_csv)
#
#     else:
#         print("Failed to load configuration. Cannot run Phase 05.")
