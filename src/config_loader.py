import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_GLOBAL_CONFIG_PATH = 'config.yaml'

def load_config(
    global_config_path: str = DEFAULT_GLOBAL_CONFIG_PATH,
    phase_config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Loads the global configuration and merges it with an optional phase-specific configuration.

    Args:
        global_config_path (str): Path to the global configuration YAML file.
        phase_config_path (Optional[str]): Path to the phase-specific configuration YAML file.

    Returns:
        Dict[str, Any]: The merged configuration dictionary. Returns an empty dict if global fails.
    """
    global_path = Path(global_config_path)
    merged_config = {}

    # Load global config
    if global_path.is_file():
        try:
            with open(global_path, 'rt') as f:
                merged_config = yaml.safe_load(f.read())
                logger.debug(f"Loaded global config from {global_config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing global config file {global_config_path}: {e}")
            return {} # Return empty dict or raise error? Returning empty for now.
        except Exception as e:
            logger.error(f"Error reading global config file {global_config_path}: {e}")
            return {}
    else:
        logger.error(f"Global config file not found at {global_config_path}")
        return {}

    # Load and merge phase-specific config if provided
    if phase_config_path:
        phase_path = Path(phase_config_path)
        if phase_path.is_file():
            try:
                with open(phase_path, 'rt') as f:
                    phase_config = yaml.safe_load(f.read())
                if phase_config: # Ensure phase config is not empty
                    merged_config.update(phase_config)
                    logger.debug(f"Loaded and merged phase-specific config from {phase_config_path}")
                else:
                    logger.warning(f"Phase-specific config file {phase_config_path} is empty.")
            except yaml.YAMLError as e:
                logger.error(f"Error parsing phase-specific config file {phase_config_path}: {e}")
                # Continue with only global config if phase-specific fails
            except Exception as e:
                logger.error(f"Error reading phase-specific config file {phase_config_path}: {e}")
                # Continue with only global config
        else:
            logger.warning(f"Phase-specific config file not found at {phase_config_path}. Using global config only.")

    logger.info("Configuration loaded successfully.")
    return merged_config

# Example usage (optional)
# if __name__ == '__main__':
#     from utils.logger import setup_logging
#     setup_logging() # Setup logging first
#
#     # Example 1: Load only global config
#     config1 = load_config()
#     print("Global Config Only:")
#     print(config1)
#
#     # Example 2: Load global and a (potentially non-existent) phase config
#     # Assuming src/pipeline/03_train/config.yaml exists and has overrides
#     phase_path = 'src/pipeline/03_train/config.yaml'
#     config2 = load_config(phase_config_path=phase_path)
#     print(f"\nMerged Config (Global + {phase_path}):")
#     print(config2)
#
#     # Example 3: Non-existent global config
#     config3 = load_config(global_config_path='non_existent_config.yaml')
#     print("\nNon-existent Global Config:")
#     print(config3)
