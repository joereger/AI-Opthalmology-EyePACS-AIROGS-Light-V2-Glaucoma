import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from typing import Dict, Any, Optional, List, Tuple

# Project-specific imports
from src.config_loader import load_config
from src.utils.file_utils import ensure_dir_exists
from src.data_handling.datasets import get_datasets, get_dataloaders
from src.models.mobilenetv3.model import get_model # Assuming get_model can load state_dict

logger = logging.getLogger(__name__)

# Define phase-specific config path
PHASE_CONFIG_PATH = 'src/pipeline/04_evaluate/config.yaml'

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str] # e.g., ['NRG', 'RG']
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluates the model on the provided dataloader (test set).

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader for the test set.
        device: The device to run evaluation on ('cuda' or 'cpu').
        class_names: List of class names corresponding to dataset indices.

    Returns:
        A tuple containing:
        - Dict[str, Any]: Dictionary of calculated metrics (accuracy, precision, recall, f1, auc, confusion_matrix).
        - pd.DataFrame: DataFrame with predictions (image_path, predicted_label, probability_NRG, probability_RG, true_label, true_label_idx, predicted_label_idx).
    """
    model.eval()  # Set model to evaluation mode
    logger.info("Starting model evaluation on the test set...")

    all_preds = []
    all_labels = []
    all_probs = [] # Store probabilities for AUC calculation
    all_image_paths = [] # Store image paths for output
    all_indices = []  # Store indices for mapping to dataset samples later

    with torch.no_grad():
        batch_idx = 0
        for inputs, labels in dataloader:
            # Track batch indices for later image path mapping
            batch_size = inputs.size(0)
            indices = list(range(batch_idx * dataloader.batch_size, 
                                batch_idx * dataloader.batch_size + batch_size))
            all_indices.extend(indices)
            
            # For now, just use placeholder names - we'll replace them after inference
            for i in range(batch_size):
                all_image_paths.append(f"placeholder_{indices[i]}.jpg")
            
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_idx += 1

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Calculate probabilities (apply softmax)
            probabilities = torch.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    logger.info("Inference complete. Calculating metrics...")

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    # Calculate precision, recall, f1 for each class (binary case: pos_label=1, i.e., 'RG')
    # average='binary' assumes the positive class is 1 ('RG' based on typical ImageFolder order)
    # Check class_to_idx if unsure. Let's assume 'RG' is class 1.
    pos_label_index = class_names.index('RG') if 'RG' in class_names else 1 # Default to 1 if RG not found
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', pos_label=pos_label_index, zero_division=0
    )
    
    # Calculate AUC using probabilities of the positive class ('RG')
    # all_probs is a list of [prob_NRG, prob_RG] arrays
    probs_pos_class = [p[pos_label_index] for p in all_probs]
    try:
        auc = roc_auc_score(all_labels, probs_pos_class)
    except ValueError as e:
        logger.warning(f"Could not calculate AUC: {e}. This might happen if only one class is present in the test set.")
        auc = None # Or 0.0 or float('nan')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    logger.info(f"Confusion Matrix:\n{cm_df}")


    metrics = {
        'accuracy': accuracy,
        f'precision_{class_names[pos_label_index]}': precision,
        f'recall_{class_names[pos_label_index]}': recall,
        f'f1_score_{class_names[pos_label_index]}': f1,
        'auc': auc,
        'confusion_matrix': cm_df.to_dict() # Store as dict for JSON compatibility
    }
    
    logger.info(f"Metrics calculated: {metrics}")
    
    # Now replace the placeholder image paths with the actual file names
    real_image_paths = []
    if hasattr(dataloader.dataset, 'samples'):
        for idx in all_indices:
            if idx < len(dataloader.dataset.samples):
                img_path, _ = dataloader.dataset.samples[idx]
                real_image_paths.append(Path(img_path).name)
            else:
                # Fallback in case of any index errors
                real_image_paths.append(f"unknown_image_{idx}.jpg")
    else:
        # If we couldn't get the samples, keep the placeholders
        real_image_paths = all_image_paths
    
    # Now use these real image paths in the DataFrame
    # Create predictions DataFrame with standardized format (matching Phase 05)
    idx_to_class = {v: k for k, v in dataloader.dataset.class_to_idx.items()}
    
    # Order columns to match standardized format across phases:
    # 1. image_path, 2. predicted_label, 3-4. probabilities, 5. true_label, 6-7. label indices
    predictions_df = pd.DataFrame({
        'image_path': real_image_paths,
        'predicted_label': [idx_to_class[pred] for pred in all_preds],
        f'probability_{class_names[0]}': [p[0] for p in all_probs], # Assumes class 0 is NRG
        f'probability_{class_names[1]}': [p[1] for p in all_probs],  # Assumes class 1 is RG
        'true_label': [idx_to_class[label] for label in all_labels],
        'true_label_idx': all_labels,
        'predicted_label_idx': all_preds
    })


    return metrics, predictions_df


def run_phase(config: Dict[str, Any], run_id: str) -> bool:
    """
    Executes Phase 04: Model Evaluation.

    Args:
        config (Dict[str, Any]): The merged configuration dictionary.
        run_id (str): The specific run ID of the model to evaluate.

    Returns:
        bool: True if the phase completes successfully, False otherwise.
    """
    logger.info(f"--- Starting Phase 04: Model Evaluation for run_id='{run_id}' ---")
    phase_success = False
    try:
        # --- Configuration ---
        num_classes = config.get('num_classes', 2)
        model_name = config.get('model_name', 'mobilenetv3')
        
        # Use data/03_train_model directory structure for model loading
        model_base_dir = Path(config.get('train_model_dir', 'data/03_train_model'))
        model_path = model_base_dir / run_id / model_name / 'best_model.pth'
        
        # New output path in data/04_evaluate
        evaluate_dir = Path(config.get('evaluate_dir', 'data/04_evaluate'))
        results_run_dir = evaluate_dir / run_id
        metrics_save_path = results_run_dir / 'evaluation_metrics.json'
        predictions_save_path = results_run_dir / 'predictions.csv'  # Standardized filename across phases

        ensure_dir_exists(results_run_dir)

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
            logger.warning("No GPU acceleration available. Evaluation will run on CPU (slow).")
        
        logger.info(f"Using device: {device}")

        # --- Data Loading (Test Set Only) ---
        logger.info("Loading test dataset and dataloader...")
        # Need the full config for get_datasets/get_dataloaders
        datasets = get_datasets(config, run_id) 
        if not datasets or 'test' not in datasets:
             logger.error(f"Could not load test dataset for run_id '{run_id}'. Check Phase 02 output.")
             return False
             
        # Create a dictionary containing only the test dataset for get_dataloaders
        test_dataset_dict = {'test': datasets['test']}
        dataloaders = get_dataloaders(test_dataset_dict, config)
        if not dataloaders or 'test' not in dataloaders:
             logger.error("Failed to create test dataloader.")
             return False
        
        test_dataloader = dataloaders['test']
        class_names = datasets['test'].classes # Get class names from dataset
        logger.info(f"Test dataset loaded: {len(datasets['test'])} samples. Classes: {class_names}")


        # --- Model Loading with Metadata ---
        logger.info(f"Loading model from: {model_path}")
        if not model_path.is_file():
            logger.error(f"Model file not found at {model_path}. Ensure Phase 03 ran successfully and saved the model.")
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

        # --- Evaluation ---
        metrics, predictions_df = evaluate_model(model, test_dataloader, device, class_names)

        # --- Save Results ---
        logger.info(f"Saving evaluation metrics to: {metrics_save_path}")
        try:
            with open(metrics_save_path, 'w') as f:
                # Convert numpy types if necessary for JSON serialization (esp. in confusion matrix)
                 json.dump(metrics, f, indent=4) 
            logger.info("Metrics saved successfully.")
        except TypeError as e:
             logger.error(f"TypeError saving metrics to JSON (possibly due to numpy types): {e}. Trying conversion.")
             # Attempt basic conversion for common numpy types
             def convert_types(obj):
                 if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                     np.int16, np.int32, np.int64, np.uint8,
                                     np.uint16, np.uint32, np.uint64)):
                     return int(obj)
                 elif isinstance(obj, (np.float_, np.float16, np.float32, 
                                      np.float64)):
                     return float(obj)
                 elif isinstance(obj, (np.ndarray,)): # Handle arrays if needed
                     return obj.tolist()
                 elif isinstance(obj, dict):
                      return {k: convert_types(v) for k, v in obj.items()}
                 elif isinstance(obj, list):
                      return [convert_types(i) for i in obj]
                 return obj # Default case
             try:
                 import numpy as np # Import numpy only if needed for conversion
                 converted_metrics = convert_types(metrics)
                 with open(metrics_save_path, 'w') as f:
                     json.dump(converted_metrics, f, indent=4)
                 logger.info("Metrics saved successfully after type conversion.")
             except Exception as conv_e:
                 logger.error(f"Failed to save metrics even after conversion: {conv_e}", exc_info=True)
                 # Decide if this is fatal
        except Exception as e:
            logger.error(f"Failed to save evaluation metrics: {e}", exc_info=True)
            # Decide if this is fatal

        logger.info(f"Saving test predictions to: {predictions_save_path}")
        try:
            predictions_df.to_csv(predictions_save_path, index=False)
            logger.info("Predictions saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save predictions CSV: {e}", exc_info=True)
            # Decide if this is fatal

        phase_success = True # Consider success if evaluation ran, even if saving failed?

    except Exception as e:
        logger.error(f"An unexpected error occurred during Phase 04: {e}", exc_info=True)
        phase_success = False

    if phase_success:
        logger.info("--- Phase 04 Completed Successfully ---")
    else:
        logger.error("--- Phase 04 Failed ---")
        
    return phase_success


# Example of how this might be called from the CLI (in cli.py)
# if __name__ == '__main__':
#     import numpy as np # Import numpy here if running the example conversion
#     from src.utils.logger import setup_logging
#     setup_logging() 
#
#     # Load merged config (Global + Phase-specific)
#     cfg = load_config(phase_config_path=PHASE_CONFIG_PATH) 
#
#     if cfg:
#         # Assumes Phase 02 & 03 ran successfully for 'run_1'
#         test_run_id = "run_1" 
#         print(f"\nRunning Phase 04 for run_id='{test_run_id}'...")
#         
#         # Ensure the model file exists for the test
#         model_p = Path(cfg['model_dir']) / test_run_id / 'best_model.pth'
#         if not model_p.exists():
#              print(f"ERROR: Model file {model_p} not found. Cannot run Phase 04 example.")
#              # Create a dummy model file for testing structure? Requires saving a dummy state_dict
#         else:
#             success = run_phase(cfg, run_id=test_run_id)
#             print(f"Phase 04 finished: {'Success' if success else 'Failed'}")
#             # Check results/mobilenetv3/run_1/ for output files
#     else:
#         print("Failed to load configuration. Cannot run Phase 04.")
