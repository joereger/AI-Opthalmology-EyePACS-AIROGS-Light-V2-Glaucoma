import logging
from pathlib import Path
from typing import Dict, Any

# Assume config_loader is in src directory, adjust if structure changes
from src.config_loader import load_config 

logger = logging.getLogger(__name__)

def run_phase(config: Dict[str, Any]) -> bool:
    """
    Executes Phase 01: Raw Input Ingestion Validation.
    Checks if the raw data directory specified in the config exists.

    Args:
        config (Dict[str, Any]): The merged configuration dictionary.

    Returns:
        bool: True if the raw data directory exists, False otherwise.
    """
    logger.info("--- Starting Phase 01: Raw Input Ingestion Validation ---")
    
    try:
        raw_data_path_str = config.get('raw_data_dir')
        if not raw_data_path_str:
            logger.error("Configuration key 'raw_data_dir' is missing.")
            return False
            
        raw_data_path = Path(raw_data_path_str)
        logger.info(f"Checking for raw data directory at: {raw_data_path.resolve()}")

        if raw_data_path.is_dir():
            logger.info(f"Successfully found raw data directory: {raw_data_path}")
            logger.info("--- Phase 01 Completed Successfully ---")
            return True
        else:
            logger.error(f"Raw data directory not found or is not a directory: {raw_data_path}")
            logger.error("--- Phase 01 Failed ---")
            return False
            
    except Exception as e:
        logger.error(f"An unexpected error occurred during Phase 01: {e}", exc_info=True)
        logger.error("--- Phase 01 Failed ---")
        return False

# Example of how this might be called from the CLI (in cli.py)
# if __name__ == '__main__':
#     # Setup logging first (assuming setup_logging is called elsewhere)
#     from src.utils.logger import setup_logging
#     setup_logging() 
#
#     # Load configuration
#     # In a real scenario, cli.py would handle loading config and passing it
#     cfg = load_config() 
#
#     if cfg:
#         success = run_phase(cfg)
#         if success:
#             print("Phase 01 finished successfully.")
#         else:
#             print("Phase 01 failed. Check logs for details.")
#             # Potentially exit the main script if phases depend on each other
#             # import sys
#             # sys.exit(1) 
#     else:
#         print("Failed to load configuration. Cannot run Phase 01.")
