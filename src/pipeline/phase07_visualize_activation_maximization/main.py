import logging
import argparse
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Project-specific imports
from src.config_loader import load_config
from src.utils.file_utils import ensure_dir_exists, get_latest_run_id, _get_next_id # Use helper for output dir
from src.utils.logger import setup_logging # Corrected import location
from src.models.mobilenetv3.model import get_model # Assuming MobileNetV3 is used
# Import functions from utils and the new report generator
from src.pipeline.phase07_visualize_activation_maximization.utils import (
    perform_activation_maximization, 
    find_real_examples, 
    # generate_html_report, # Moved
    calculate_importance_scores
) 
from src.pipeline.phase07_visualize_activation_maximization.html_report_generator import generate_html_report # Added import

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        logger.info("CUDA device found, using GPU.")
        return torch.device("cuda:0")
    # Check for MPS (Apple Silicon) support
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS device found, using Apple Silicon GPU.")
        return torch.device("mps")
    else:
        logger.info("No GPU found, using CPU.")
        return torch.device("cpu")

def load_model_and_metadata(model_path: Path, metadata_path: Path, config: Dict) -> Tuple[torch.nn.Module, Dict]:
    """
    Load the trained model and its metadata.
    
    Args:
        model_path: Path to the model file (.pth)
        metadata_path: Path to the model metadata file (.json)
        config: Configuration dictionary for the current phase
        
    Returns:
        Tuple of (loaded_model, metadata)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded model metadata from {metadata_path}")
        except Exception as e:
            logger.warning(f"Could not load or parse model metadata from {metadata_path}: {e}. Proceeding without it.")
    else:
        logger.warning(f"Model metadata file not found at {metadata_path}. Proceeding without it.")

    # Determine number of classes - prioritize metadata, then config, then default
    num_classes = metadata.get('num_classes', config.get('num_classes', 2)) # Default to 2 if not found
    logger.info(f"Determined number of classes: {num_classes}")

    # Load model structure (assuming get_model factory)
    # We set pretrained=False because we are loading trained weights
    model = get_model(num_classes=num_classes, pretrained=False) 
    
    # Load trained weights
    device = get_device()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval() # Set model to evaluation mode
        logger.info(f"Successfully loaded model weights from {model_path} to device '{device}'")
    except Exception as e:
        logger.error(f"Failed to load model weights from {model_path}: {e}", exc_info=True)
        raise
        
    return model, metadata

# Removed log_level and log_path parameters as logging is handled by CLI entry point
# Added global_config parameter
def run_phase(config_path: str, run_id: Optional[str] = None, global_config: Optional[Dict] = None) -> bool:
    """
    Executes Phase 07: Visualize Activation Maximization.
    
    Args:
        config_path: Path to the configuration file for this phase.
        run_id: Specific run ID from Phase 03 (train_model) to use. If None, finds the latest.
        log_level: Logging level (e.g., 'INFO', 'DEBUG').
        log_path: Optional path to save the log file.
        
    Returns:
        bool: Success status
    """
    # --- 1. Setup: Config, Logging, Directories ---
    try:
        config = load_config(config_path)
        
        # Determine the source run directory (from Phase 03)
        # TODO: Make the source phase configurable? Defaulting to 03_train_model for now.
        source_phase_dir = Path(config.get('train_model_dir', 'data/03_train_model')) 
        if run_id:
            source_run_dir = source_phase_dir / run_id
            if not source_run_dir.is_dir():
                 raise FileNotFoundError(f"Specified run directory not found: {source_run_dir}")
            logger.info(f"Using specified source run directory: {source_run_dir}")
            source_run_id_str = run_id # Keep track of the string ID
        else:
            # Use get_latest_run_id which likely needs the config to find base dirs
            source_run_id_str = get_latest_run_id(config) 
            if not source_run_id_str:
                # Check specific dir mentioned in config if get_latest_run_id needs refinement
                raise FileNotFoundError(f"Could not find latest run ID using config. Check {source_phase_dir}")
            source_run_dir = source_phase_dir / source_run_id_str
            if not source_run_dir.is_dir():
                 raise FileNotFoundError(f"Constructed latest run directory not found: {source_run_dir}")
            logger.info(f"Using latest found source run directory: {source_run_dir}")

        # Create the output directory for this phase's run, using the source run ID
        output_dir_base = Path(config['output_dir_base'])
        # Use the run_id determined from the source phase (e.g., 'run_1')
        current_run_output_dir = output_dir_base / source_run_id_str 
        ensure_dir_exists(current_run_output_dir) # Create the directory if it doesn't exist
        
        # Logging is handled by the main CLI entry point
        logger.info(f"--- Starting Phase 07: Visualize Activation Maximization ---")
        logger.info(f"Using configuration: {config_path}")
        logger.info(f"Source model run directory: {source_run_dir}")
        logger.info(f"Output directory for this run: {current_run_output_dir}")
        # Removed redundant logging setup and log file message

        # Save the config used for this run
        with open(current_run_output_dir / 'config_used.json', 'w') as f:
            json.dump(config, f, indent=4)

    except Exception as e:
        logging.error(f"Initialization failed: {e}", exc_info=True)
        return False

    # --- 2. Load Model ---
    try:
        # Get model artifact subdirectory from config
        model_artifact_subdir = config.get('model_artifact_subdir', 'mobilenetv3') # Default if not in config
        model_path = source_run_dir / model_artifact_subdir / 'best_model.pth'
        metadata_path = source_run_dir / model_artifact_subdir / 'model_metadata.json'
        
        model, metadata = load_model_and_metadata(model_path, metadata_path, config)
        
        # Add loaded metadata to config for downstream use (e.g., normalization values)
        config['loaded_metadata'] = metadata 
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return False

    # --- 3. Identify Targets & Calculate Importance ---
    try:
        logger.info("Identifying target units and calculating importance scores...")
        # Pass the source run ID and global config to find correct validation data
        if global_config is None:
             # Attempt to load global config if not passed (e.g., direct run)
             logger.warning("Global config not passed to run_phase. Attempting to load.")
             try:
                 global_config = load_config()
             except Exception:
                 logger.error("Failed to load global config within run_phase. Data path resolution might fail.")
                 global_config = {} # Use empty dict to avoid immediate crash

        target_units = calculate_importance_scores(
            model, 
            config, # Phase-specific config
            source_run_id_str, # Pass the run ID string
            global_config,     # Pass the global config for base paths
            get_device()
        ) 
        if not target_units:
             logger.warning("No target units identified based on configuration. Check 'targets' in config.")
             # Decide if this is an error or just means no work to do
             # return True # Or False depending on desired behavior
        logger.info(f"Identified {len(target_units)} target units for visualization.")
        
    except Exception as e:
        logger.error(f"Failed during target identification/importance calculation: {e}", exc_info=True)
        return False

    # --- 4. Perform Activation Maximization ---
    visualization_results = []
    try:
        logger.info("Starting activation maximization for identified units...")
        viz_output_dir = current_run_output_dir / "visualizations"
        ensure_dir_exists(viz_output_dir)
        
        for unit_info in target_units:
            logger.debug(f"Processing unit: {unit_info['layer_name']} - Unit {unit_info['unit_index']}")
            # Placeholder: This function needs implementation in utils.py
            viz_path = perform_activation_maximization(
                model, 
                unit_info, 
                config['optimization'], 
                config, # Pass the phase-specific config dict (contains loaded_metadata)
                viz_output_dir, 
                get_device()
            )
            if viz_path:
                 unit_info['visualization_path'] = str(viz_path) # Store path relative to CWD
                 visualization_results.append(unit_info)
            else:
                 logger.warning(f"Failed to generate visualization for unit: {unit_info}")

        logger.info(f"Generated {len(visualization_results)} synthetic visualizations.")

    except Exception as e:
        logger.error(f"Failed during activation maximization: {e}", exc_info=True)
        # Continue to reporting if desired, or return False
        return False 

    # --- 5. Find Real Examples (Optional) ---
    if config.get('find_real_examples', {}).get('enabled', False):
        try:
            logger.info("Finding real image examples...")
            examples_high_dir = current_run_output_dir / "real_examples_high"
            examples_low_dir = current_run_output_dir / "real_examples_low"
            ensure_dir_exists(examples_high_dir)
            ensure_dir_exists(examples_low_dir)
            
            # Placeholder: This function needs implementation in utils.py
            find_real_examples(
                model, 
                visualization_results, 
                config, # Phase-specific config
                source_run_id_str, # Pass the run ID string
                global_config,     # Pass the global config for base paths
                examples_high_dir, 
                examples_low_dir, 
                get_device()
            )
            logger.info("Completed finding real examples.")
            
        except Exception as e:
            logger.error(f"Failed while finding real examples: {e}", exc_info=True)
            # Decide if this is fatal or just skip this part
            # return False 

    # --- 6. Generate HTML Report (Optional) ---
    if config.get('reporting', {}).get('enabled', False) and visualization_results:
        try:
            logger.info("Generating HTML report...")
            report_path = current_run_output_dir / config['reporting']['report_filename']
            
            # Placeholder: This function needs implementation in utils.py
            generate_html_report(
                visualization_results, 
                report_path, 
                config['reporting'],
                config.get('find_real_examples', {}).get('enabled', False) # Pass flag if examples were generated
            )
            logger.info(f"HTML report generated at: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}", exc_info=True)
            # Report generation failure might not be critical
            # return False 

    logger.info(f"--- Phase 07: Visualize Activation Maximization finished successfully ---")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 07: Visualize Activation Maximization - Generate visualizations of learned features.")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/pipeline/phase07_visualize_activation_maximization/config.yaml",
        help="Path to the configuration file for this phase."
    )
    parser.add_argument(
        "--run-id", 
        type=str, 
        default=None,
        help="Optional: Specify the run ID from the training phase (e.g., 'run_1') to use. If omitted, the latest run will be used."
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level."
    )
    parser.add_argument(
        "--log-path", 
        type=str, 
        default=None,
        help="Optional path to save the log file. If None, it defaults to a file in the run's output directory."
    )
    
    args = parser.parse_args()

    # Need click context to access global config easily, or load it again here.
    # For simplicity, assuming run_phase is called from cli.py which has context.
    # If run directly, would need to handle global config loading here.
    
    # This direct call won't work correctly without the click context's global config.
    # The execution should happen via the cli.py entry point.
    # If direct execution is needed, refactor to load global config here too.
    logger.warning("Running main.py directly might not work as expected without Click context for global config.")
    logger.warning("Please run using 'python main.py activation-maximization ...'")
    
    # Attempting direct run anyway for basic testing, but data path logic might fail
    try:
        # Manually load global config if run directly
        import click
        ctx = click.Context(click.Command('dummy'))
        # Load global config for direct run attempt
        global_cfg_direct = {}
        try:
             global_cfg_direct = load_config() # Load global config
        except Exception:
             logger.error("Failed to load global config when running directly.")

        success = run_phase(
            config_path=args.config, 
            run_id=args.run_id,
            global_config=global_cfg_direct # Pass loaded global config
        )
        if success:
            logger.info("Phase 07 completed successfully.")
            exit(0)
        else:
            logger.error("Phase 07 encountered errors.")
            exit(1)
            
    except Exception as e:
        logger.critical(f"Phase 07 failed with critical error: {e}", exc_info=True)
        exit(1)
