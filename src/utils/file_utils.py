import os
import shutil
import logging
from pathlib import Path
import re
from typing import Union, Optional

logger = logging.getLogger(__name__)

def ensure_dir_exists(path: Union[str, Path]):
    """Creates a directory if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        logger.info(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        logger.error(f"Path exists but is not a directory: {path}")
        raise NotADirectoryError(f"Path exists but is not a directory: {path}")

def copy_file(src: Union[str, Path], dst: Union[str, Path]):
    """Copies a single file from src to dst, creating destination directory if needed."""
    src_path = Path(src)
    dst_path = Path(dst)
    try:
        ensure_dir_exists(dst_path.parent)
        shutil.copy2(src_path, dst_path) # copy2 preserves metadata
        logger.debug(f"Copied file from {src_path} to {dst_path}")
    except Exception as e:
        logger.error(f"Error copying file from {src_path} to {dst_path}: {e}")
        raise

def _get_next_id(base_dir: Union[str, Path], prefix: str) -> str:
    """Helper function to find the next sequential ID in a directory."""
    base_path = Path(base_dir)
    ensure_dir_exists(base_path) # Ensure base directory exists

    max_id = 0
    pattern = re.compile(rf"^{prefix}(\d+)$") # Regex to match prefix followed by digits

    for item in base_path.iterdir():
        if item.is_dir(): # Check only directories for run/batch IDs
            match = pattern.match(item.name)
            if match:
                current_id = int(match.group(1))
                if current_id > max_id:
                    max_id = current_id

    next_id_num = max_id + 1
    next_id_str = f"{prefix}{next_id_num}"
    logger.info(f"Determined next ID in {base_dir} with prefix '{prefix}': {next_id_str}")
    return next_id_str

def get_latest_run_id(
    config: dict, # Pass the loaded config dictionary
    run_id_prefix: str = 'run_',
    default: Optional[str] = None
) -> Optional[str]:
    """
    Finds the highest existing run ID based on directories
    in both the conformed data and trained model directories.

    Args:
        config (dict): The loaded configuration dictionary containing base paths.
        run_id_prefix (str): The prefix for run ID directories (e.g., 'run_').
        default (Optional[str]): Default value to return if no runs exist.

    Returns:
        Optional[str]: The highest existing run ID (e.g., 'run_3'), or default if none exists.
    """
    conformed_dir = Path(config.get('conformed_data_dir', 'data/02_conformed_to_imagefolder'))
    model_dir = Path('data/03_train_model')  # Hard-coded for consistency

    max_id = 0
    found_any = False

    # Check conformed data directory
    if conformed_dir.exists():
        pattern = re.compile(rf"^{run_id_prefix}(\d+)$")
        for item in conformed_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    found_any = True
                    current_id = int(match.group(1))
                    if current_id > max_id:
                        max_id = current_id

    # Check model directory
    if model_dir.exists():
        pattern = re.compile(rf"^{run_id_prefix}(\d+)$")
        for item in model_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    found_any = True
                    current_id = int(match.group(1))
                    if current_id > max_id:
                        max_id = current_id

    if not found_any:
        logger.info(f"No existing run IDs found in '{conformed_dir}' or '{model_dir}'")
        return default
    
    latest_id_str = f"{run_id_prefix}{max_id}"
    logger.info(f"Found latest run ID based on '{conformed_dir}' and '{model_dir}': {latest_id_str}")
    return latest_id_str

def get_next_run_id(
    config: dict, # Pass the loaded config dictionary
    run_id_prefix: str = 'run_'
) -> str:
    """
    Determines the next sequential run ID based on existing directories
    in both the conformed data and trained model directories.

    Args:
        config (dict): The loaded configuration dictionary containing base paths.
        run_id_prefix (str): The prefix for run ID directories (e.g., 'run_').

    Returns:
        str: The next sequential run ID (e.g., 'run_1', 'run_2').
    """
    conformed_dir = Path(config.get('conformed_data_dir', 'data/02_conformed_to_imagefolder'))
    model_dir = Path('data/03_train_model')  # Hard-coded for consistency

    max_id = 0

    # Check conformed data directory
    if conformed_dir.exists():
        pattern = re.compile(rf"^{run_id_prefix}(\d+)$")
        for item in conformed_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    current_id = int(match.group(1))
                    if current_id > max_id:
                        max_id = current_id

    # Check model directory
    if model_dir.exists():
        pattern = re.compile(rf"^{run_id_prefix}(\d+)$")
        for item in model_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    current_id = int(match.group(1))
                    if current_id > max_id:
                        max_id = current_id

    next_id_num = max_id + 1
    next_id_str = f"{run_id_prefix}{next_id_num}"
    logger.info(f"Determined next run ID based on '{conformed_dir}' and '{model_dir}': {next_id_str}")
    return next_id_str


def get_next_batch_id(
    config: dict, # Pass the loaded config dictionary
    run_id: str,  # The run ID to associate the batch with
    batch_id_prefix: str = 'batch_'
) -> str:
    """
    Determines the next sequential batch ID based on existing batch directories
    for a specific run.

    Args:
        config (dict): The loaded configuration dictionary containing base paths.
        run_id (str): The run ID to associate the batch with.
        batch_id_prefix (str): The prefix for batch ID directories (e.g., 'batch_').

    Returns:
        str: The next sequential batch ID (e.g., 'batch_1', 'batch_2').
    """
    predict_dir = Path(config.get('predict_dir', 'data/05_predict'))
    run_batches_dir = predict_dir / run_id
    
    # If the run directory doesn't exist yet, first batch will be batch_1
    if not run_batches_dir.exists():
        return f"{batch_id_prefix}1"
    
    # Find the highest existing batch number
    max_id = 0
    pattern = re.compile(rf"^{batch_id_prefix}(\d+)$")
    
    for item in run_batches_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                current_id = int(match.group(1))
                if current_id > max_id:
                    max_id = current_id

    next_id_num = max_id + 1
    next_id_str = f"{batch_id_prefix}{next_id_num}"
    logger.info(f"Determined next batch ID for run '{run_id}': {next_id_str}")
    return next_id_str


# Example usage (optional)
# if __name__ == '__main__':
#     from utils.logger import setup_logging
#     setup_logging() # Setup logging first
#
#     # Example: Ensure a directory exists
#     ensure_dir_exists("temp_test_dir/subdir")
#
#     # Example: Copy a file
#     with open("temp_test_file.txt", "w") as f:
#         f.write("Test content")
#     copy_file("temp_test_file.txt", "temp_test_dir/subdir/copied_file.txt")
#
#     # Example: Get next run ID (assuming config.yaml exists and defines paths)
#     # Need to load config first
#     try:
#         from config_loader import load_config
#         cfg = load_config()
#         if cfg:
#             # Create dummy dirs to test ID generation
#             Path(cfg['conformed_data_dir']).mkdir(parents=True, exist_ok=True)
#             Path(cfg['model_dir']).mkdir(parents=True, exist_ok=True)
#             Path(cfg['conformed_data_dir'], 'run_1').mkdir(exist_ok=True)
#             Path(cfg['model_dir'], 'run_1').mkdir(exist_ok=True)
#             Path(cfg['model_dir'], 'run_2').mkdir(exist_ok=True)
#
#             next_run = get_next_run_id(cfg)
#             print(f"Next Run ID: {next_run}") # Should be run_3
#
#             # Example: Get next batch ID
#             Path(cfg['inference_input_dir']).mkdir(parents=True, exist_ok=True)
#             Path(cfg['inference_input_dir'], 'batch_1').mkdir(exist_ok=True)
#             next_batch = get_next_batch_id(cfg)
#             print(f"Next Batch ID: {next_batch}") # Should be batch_2
#         else:
#             print("Could not load config for testing.")
#
#     except ImportError:
#         print("Could not import config_loader for testing.")
#     except Exception as e:
#         print(f"An error occurred during testing: {e}")
#
#     # Clean up test files/dirs
#     # import time; time.sleep(1) # Allow logs to flush
#     # if Path("temp_test_file.txt").exists(): os.remove("temp_test_file.txt")
#     # if Path("temp_test_dir").exists(): shutil.rmtree("temp_test_dir")
#     # # Clean up dummy run/batch dirs if config was loaded
#     # if cfg:
#     #     if Path(cfg['conformed_data_dir'], 'run_1').exists(): shutil.rmtree(Path(cfg['conformed_data_dir'], 'run_1'))
#     #     if Path(cfg['model_dir'], 'run_1').exists(): shutil.rmtree(Path(cfg['model_dir'], 'run_1'))
#     #     if Path(cfg['model_dir'], 'run_2').exists(): shutil.rmtree(Path(cfg['model_dir'], 'run_2'))
#     #     if Path(cfg['inference_input_dir'], 'batch_1').exists(): shutil.rmtree(Path(cfg['inference_input_dir'], 'batch_1'))
