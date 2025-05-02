import logging
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

# Assuming utils and config_loader are in src directory
from src.config_loader import load_config
from src.utils.file_utils import ensure_dir_exists, copy_file, get_next_run_id

logger = logging.getLogger(__name__)

EXPECTED_SPLITS = ['train', 'validation', 'test']
EXPECTED_CLASSES = ['NRG', 'RG'] # Non-Referable Glaucoma, Referable Glaucoma
ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg'] # Case-insensitive check will be used

def validate_raw_structure(raw_data_path: Path) -> bool:
    """Checks if the raw data directory has the expected ImageFolder structure."""
    if not raw_data_path.is_dir():
        logger.error(f"Raw data path is not a directory: {raw_data_path}")
        return False

    all_valid = True
    for split in EXPECTED_SPLITS:
        split_path = raw_data_path / split
        if not split_path.is_dir():
            logger.error(f"Missing expected split directory: {split_path}")
            all_valid = False
            continue # Skip class checks if split dir is missing

        for class_label in EXPECTED_CLASSES:
            class_path = split_path / class_label
            if not class_path.is_dir():
                logger.error(f"Missing expected class directory: {class_path}")
                all_valid = False
                continue # Skip file check if class dir is missing

            # Basic check for image files (at least one)
            has_images = any(
                f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS 
                for f in class_path.iterdir()
            )
            if not has_images:
                logger.warning(f"Class directory seems empty or contains no recognized images ({ALLOWED_IMAGE_EXTENSIONS}): {class_path}")
                # Not necessarily fatal, but worth a warning. Could make it fatal if needed.
                # all_valid = False 

    if not all_valid:
        logger.error("Raw data directory structure validation failed.")
    else:
        logger.info("Raw data directory structure validation passed.")
    return all_valid


def copy_data(raw_data_path: Path, conformed_run_path: Path):
    """Copies data from raw structure to the conformed run-specific structure."""
    logger.info(f"Starting data copy from {raw_data_path} to {conformed_run_path}")
    ensure_dir_exists(conformed_run_path)
    
    copied_count = 0
    skipped_count = 0
    error_count = 0

    for split in EXPECTED_SPLITS:
        raw_split_path = raw_data_path / split
        conformed_split_path = conformed_run_path / split
        if not raw_split_path.is_dir(): continue # Skip if source split dir doesn't exist (already logged in validation)

        for class_label in EXPECTED_CLASSES:
            raw_class_path = raw_split_path / class_label
            conformed_class_path = conformed_split_path / class_label
            if not raw_class_path.is_dir(): continue # Skip if source class dir doesn't exist

            ensure_dir_exists(conformed_class_path) # Create destination class dir

            for item in raw_class_path.iterdir():
                if item.is_file() and item.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                    dest_file = conformed_class_path / item.name
                    try:
                        copy_file(item, dest_file)
                        copied_count += 1
                        if copied_count % 1000 == 0: # Log progress periodically
                             logger.info(f"Copied {copied_count} images so far...")
                    except Exception as e:
                        logger.error(f"Failed to copy {item} to {dest_file}: {e}")
                        error_count += 1
                elif item.is_file():
                    logger.warning(f"Skipping non-image file: {item}")
                    skipped_count += 1
    
    logger.info(f"Data copy finished. Copied: {copied_count}, Skipped: {skipped_count}, Errors: {error_count}")
    if error_count > 0:
        raise RuntimeError(f"Encountered {error_count} errors during data copy.")


def run_phase(config: Dict[str, Any], run_id: Optional[str] = None) -> bool:
    """
    Executes Phase 02: Conform Raw Data to ImageFolder Format.
    Validates the raw data structure and copies images to a run-specific directory.

    Args:
        config (Dict[str, Any]): The merged configuration dictionary.
        run_id (Optional[str]): The specific run ID. If None, the next available ID is determined.

    Returns:
        bool: True if the phase completes successfully, False otherwise.
    """
    logger.info("--- Starting Phase 02: Conform Raw Data to ImageFolder Format ---")

    try:
        # Determine Run ID
        if run_id is None:
            run_id = get_next_run_id(config)
            logger.info(f"No run_id provided, determined next run_id: {run_id}")
        else:
             logger.info(f"Using provided run_id: {run_id}")

        # Get paths from config
        raw_data_path_str = config.get('raw_data_dir')
        conformed_base_path_str = config.get('conformed_data_dir')

        if not raw_data_path_str or not conformed_base_path_str:
            logger.error("Configuration missing 'raw_data_dir' or 'conformed_data_dir'.")
            return False

        raw_data_path = Path(raw_data_path_str)
        conformed_run_path = Path(conformed_base_path_str) / run_id

        # Check if target directory already exists
        if conformed_run_path.exists():
            logger.warning(f"Conformed data directory for run '{run_id}' already exists: {conformed_run_path}")
            # Decide on behavior: overwrite, skip, or fail?
            # For now, let's skip to avoid accidental overwrites. Add CLI flag later if needed.
            logger.warning("Skipping Phase 02 as target directory exists.")
            logger.info("--- Phase 02 Skipped (Directory Exists) ---")
            return True # Consider existing directory as success for subsequent phases

        # 1. Validate Raw Structure
        if not validate_raw_structure(raw_data_path):
            logger.error("--- Phase 02 Failed (Raw Data Validation) ---")
            return False

        # 2. Copy Data
        copy_data(raw_data_path, conformed_run_path)

        logger.info(f"Successfully created conformed data for run '{run_id}' at: {conformed_run_path}")
        logger.info("--- Phase 02 Completed Successfully ---")
        return True

    except Exception as e:
        logger.error(f"An unexpected error occurred during Phase 02: {e}", exc_info=True)
        logger.error("--- Phase 02 Failed ---")
        # Optional: Clean up partially created directory?
        # if 'conformed_run_path' in locals() and conformed_run_path.exists():
        #     logger.info(f"Cleaning up partially created directory: {conformed_run_path}")
        #     shutil.rmtree(conformed_run_path)
        return False

# Example of how this might be called from the CLI (in cli.py)
# if __name__ == '__main__':
#     from src.utils.logger import setup_logging
#     setup_logging() 
#
#     cfg = load_config() 
#
#     if cfg:
#         # Example 1: Determine run_id automatically
#         # success1 = run_phase(cfg) 
#         # print(f"Phase 02 (auto run_id) finished: {'Success' if success1 else 'Failed'}")
#
#         # Example 2: Specify run_id (e.g., 'run_test')
#         # Make sure raw data exists at cfg['raw_data_dir'] for this to work
#         test_id = "run_test_conform"
#         print(f"\nRunning Phase 02 with specified run_id='{test_id}'...")
#         success2 = run_phase(cfg, run_id=test_id)
#         print(f"Phase 02 (run_id='{test_id}') finished: {'Success' if success2 else 'Failed'}")
#
#         # Clean up test run directory (optional)
#         # test_dir = Path(cfg['conformed_data_dir']) / test_id
#         # if test_dir.exists():
#         #     print(f"Cleaning up test directory: {test_dir}")
#         #     shutil.rmtree(test_dir)
#
#     else:
#         print("Failed to load configuration. Cannot run Phase 02.")
