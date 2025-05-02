import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import time
import copy
import json
import datetime
from typing import Dict, Any, Optional

# Project-specific imports
from src.config_loader import load_config
from src.utils.file_utils import ensure_dir_exists, get_next_run_id
from src.data_handling.datasets import get_datasets, get_dataloaders
from src.models.mobilenetv3.model import get_model

logger = logging.getLogger(__name__)

# Define phase-specific config path relative to this file's location
# This assumes the script is run from the project root or paths are handled correctly by the caller (cli.py)
PHASE_CONFIG_PATH = 'src/pipeline/03_train/config.yaml' 

def train_model_phase(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    dataloaders: Dict[str, DataLoader],
    dataset_sizes: Dict[str, int],
    device: torch.device,
    num_epochs: int,
    model_save_path: Path
) -> nn.Module:
    """
    Trains and validates the model, saving the best version based on validation accuracy.

    Args:
        model: The PyTorch model to train.
        criterion: The loss function.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        dataloaders: Dictionary containing 'train' and 'validation' DataLoaders.
        dataset_sizes: Dictionary containing 'train' and 'validation' dataset sizes.
        device: The device to run training on ('cuda' or 'cpu').
        num_epochs: The number of epochs to train for.
        model_save_path: Path to save the best model weights.

    Returns:
        The model with the best validation weights loaded.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Ensure parent directory for saving model exists
    ensure_dir_exists(model_save_path.parent)

    logger.info(f"Starting training for {num_epochs} epochs on device: {device}")

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Epoch {epoch+1}/{num_epochs}') # Also print to console for visibility
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
                logger.debug("Set model to training mode")
            else:
                model.eval()   # Set model to evaluate mode
                logger.debug("Set model to evaluation mode")

            running_loss = 0.0
            running_corrects = 0
            batch_count = 0

            # Iterate over data.
            # Use tqdm for progress bar if desired, but basic logging is more robust for CLI
            total_batches = len(dataloaders[phase])
            log_interval = 10  # Log every 10th batch exactly
            
            # Start time tracking for batch timing
            batch_start_time = time.time()
            phase_start_time = time.time()

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch_count += 1
                
                # Log every 10th batch (log_interval=10)
                if batch_count % log_interval == 0:
                    batch_time = time.time() - batch_start_time
                    elapsed_time = time.time() - phase_start_time
                    percent_complete = (batch_count / total_batches) * 100
                    
                    # Reset timer for next batch
                    batch_start_time = time.time()
                    
                    # Log progress with timing details
                    logger.info(f'Epoch {epoch+1}/{num_epochs} [{phase}] Batch {batch_count}/{total_batches} ({percent_complete:.1f}%) Loss: {loss.item():.4f} Batch time: {batch_time:.2f}s Elapsed: {elapsed_time:.1f}s')


            if phase == 'train':
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0] # Get current learning rate
                logger.info(f"Scheduler stepped. Current LR: {current_lr:.7f}")


            epoch_loss = running_loss / dataset_sizes[phase]
            # Use float() instead of double() for MPS compatibility (Apple Silicon GPU)
            # MPS doesn't support float64 dtype
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            logger.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}') # Also print summary

            # deep copy the model if validation accuracy improves
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                logger.info(f"*** New best validation accuracy: {best_acc:.4f}. Saving model state. ***")
                # Save the best model weights immediately
                try:
                    torch.save(best_model_wts, model_save_path)
                    logger.info(f"Best model weights saved to {model_save_path}")
                except Exception as e:
                    logger.error(f"Error saving best model weights to {model_save_path}: {e}")


        print() # Newline after each epoch's train/val summary

    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best validation Acc: {best_acc:4f}')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc: {best_acc:4f}')


    # load best model weights before returning
    model.load_state_dict(best_model_wts)
    return model


def run_phase(config: Dict[str, Any], run_id: Optional[str] = None) -> bool:
    """
    Executes Phase 03: Model Training.

    Args:
        config (Dict[str, Any]): The merged configuration dictionary.
        run_id (Optional[str]): The specific run ID for which to train. If None, the next available ID is determined.

    Returns:
        bool: True if the phase completes successfully, False otherwise.
    """
    logger.info("--- Starting Phase 03: Model Training ---")
    phase_success = False
    try:
        # Determine Run ID
        if run_id is None:
            run_id = get_next_run_id(config) # Assumes Phase 02 created the dir
            logger.info(f"No run_id provided, using determined run_id: {run_id}")
        else:
             logger.info(f"Using provided run_id: {run_id}")

        # --- Configuration ---
        num_epochs = config.get('num_epochs', 6)
        lr = config.get('learning_rate', 0.001)
        scheduler_step = config.get('scheduler_step_size', 3)
        scheduler_gamma = config.get('scheduler_gamma', 0.1)
        num_classes = config.get('num_classes', 2)
        model_name = config.get('model_name', 'mobilenetv3')
        
        # Use data/03_train_model directory structure for model storage
        model_base_dir = Path('data/03_train_model')
        model_save_path = model_base_dir / run_id / model_name / 'best_model.pth'
        
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
            logger.warning("No GPU acceleration available. Training will run on CPU (slow).")
        
        logger.info(f"Using device: {device}")

        # --- Data Loading ---
        logger.info("Loading datasets and dataloaders...")
        datasets = get_datasets(config, run_id)
        if not datasets or 'train' not in datasets or 'validation' not in datasets:
             logger.error(f"Could not load train/validation datasets for run_id '{run_id}'. Check Phase 02 output.")
             return False
        dataloaders = get_dataloaders(datasets, config)
        if not dataloaders:
             logger.error("Failed to create dataloaders.")
             return False
        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'validation']}
        logger.info(f"Dataset sizes: Train={dataset_sizes['train']}, Validation={dataset_sizes['validation']}")

        # --- Model Initialization ---
        logger.info("Initializing model...")
        model = get_model(num_classes=num_classes, pretrained=True)
        model = model.to(device)
        logger.info(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")


        # --- Loss, Optimizer, Scheduler ---
        criterion = nn.CrossEntropyLoss()
        # TODO: Add support for other optimizers from config if needed
        optimizer = optim.Adam(model.parameters(), lr=lr) 
        # TODO: Add support for other schedulers from config if needed
        scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        logger.info(f"Optimizer: Adam (lr={lr}), Scheduler: StepLR (step={scheduler_step}, gamma={scheduler_gamma}), Loss: CrossEntropyLoss")


        # --- Training ---
        trained_model = train_model_phase(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=device,
            num_epochs=num_epochs,
            model_save_path=model_save_path
        )
        
        # Final check if model file was created
        if model_save_path.is_file():
             logger.info(f"Best model successfully saved to {model_save_path}")
             
             # Create and save model metadata
             metadata_path = model_save_path.parent / 'model_metadata.json'
             # Get PyTorch version
             pytorch_version = torch.__version__
             
             # Create training parameters dictionary
             training_params = {
                 "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "pytorch_version": pytorch_version,
                 "optimizer": "Adam",
                 "learning_rate": lr,
                 "scheduler": "StepLR",
                 "scheduler_step_size": scheduler_step,
                 "scheduler_gamma": scheduler_gamma,
                 "epochs": num_epochs,
                 "image_size": config.get('image_size', 512)
             }
             
             # Get model metadata using the model's get_metadata method
             model_metadata = trained_model.get_metadata(training_params)
             
             try:
                 with open(metadata_path, 'w') as f:
                     json.dump(model_metadata, f, indent=4)
                 logger.info(f"Model metadata saved to {metadata_path}")
             except Exception as e:
                 logger.error(f"Error saving model metadata to {metadata_path}: {e}")
             
             phase_success = True
        else:
             # This case might happen if validation accuracy never improved beyond initial state
             logger.warning(f"Best model file was not found at {model_save_path}. This might happen if validation accuracy did not improve.")
             # Let's consider this a success if training completed without errors, 
             # but log a strong warning. Phase 4 will fail if the file is missing.
             phase_success = True # Or set to False if a saved model is strictly required


    except Exception as e:
        logger.error(f"An unexpected error occurred during Phase 03: {e}", exc_info=True)
        phase_success = False

    if phase_success:
        logger.info("--- Phase 03 Completed Successfully ---")
    else:
        logger.error("--- Phase 03 Failed ---")
        
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
#         # Assumes Phase 02 ran successfully for 'run_1' (or whichever run_id is determined/passed)
#         # You might want to pass run_id explicitly from CLI args
#         test_run_id = "run_1" # Or None to auto-determine
#         print(f"\nRunning Phase 03 for run_id='{test_run_id}'...")
#         success = run_phase(cfg, run_id=test_run_id)
#         print(f"Phase 03 finished: {'Success' if success else 'Failed'}")
#     else:
#         print("Failed to load configuration. Cannot run Phase 03.")
