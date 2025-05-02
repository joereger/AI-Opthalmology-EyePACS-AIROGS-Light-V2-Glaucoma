import logging
import logging.config
import yaml
import os
from pathlib import Path

DEFAULT_LOG_CONFIG_PATH = 'logger_config.yaml'
DEFAULT_LEVEL = logging.INFO

def setup_logging(config_path=DEFAULT_LOG_CONFIG_PATH, default_level=DEFAULT_LEVEL):
    """
    Set up logging configuration from a YAML file.

    Args:
        config_path (str): Path to the logging configuration YAML file.
        default_level (int): Default logging level if config file is not found.
    """
    path = Path(config_path)
    if path.is_file():
        try:
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())
            
            # Ensure log directories exist if specified in handlers
            for handler_name, handler_config in config.get('handlers', {}).items():
                if 'filename' in handler_config:
                    log_dir = Path(handler_config['filename']).parent
                    log_dir.mkdir(parents=True, exist_ok=True)
                    
            logging.config.dictConfig(config)
            logging.info(f"Logging configured successfully from {config_path}")
        except Exception as e:
            logging.basicConfig(level=default_level)
            logging.warning(f"Failed to load logging config from {config_path}. Using basicConfig. Error: {e}")
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging config file not found at {config_path}. Using basicConfig.")

# Example usage (optional, could be called from main.py or cli.py)
# if __name__ == '__main__':
#     setup_logging()
#     # Example logging messages
#     logging.debug("This is a debug message.")
#     logging.info("This is an info message.")
#     logging.warning("This is a warning message.")
#     logging.error("This is an error message.")
#     logging.critical("This is a critical message.")
