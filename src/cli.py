import click
import logging
import importlib
from typing import Optional
import sys
# Removed sys/os imports and path manipulation from here

# Import core utilities (fast imports only)
from src.config_loader import load_config
from src.utils.logger import setup_logging, DEFAULT_LOG_CONFIG_PATH
from src.utils.file_utils import get_latest_run_id

# We'll lazily import phase modules when needed
# This prevents slow imports (like pandas in evaluate) from blocking startup
def import_phase(phase_name):
    """Dynamically import a phase module only when needed"""
    print(f"Loading phase module: {phase_name}...")
    try:
        return importlib.import_module(f"src.pipeline.{phase_name}.main")
    except ImportError as e:
        print(f"Error importing phase module {phase_name}: {e}", file=sys.stderr)
        raise

# Define phase-specific config paths relative to src
# (Adjust if structure differs or use absolute paths if necessary)
PHASE_CONFIGS = {
    'train': 'src/pipeline/phase03_train_model/config.yaml',
    'evaluate': 'src/pipeline/phase04_evaluate/config.yaml',
    'predict': 'src/pipeline/phase05_predict/config.yaml',
}

# Setup root logger temporarily for CLI setup messages
# Proper setup happens in main_cli based on config file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--log-config', default=DEFAULT_LOG_CONFIG_PATH, help='Path to logging configuration file.')
@click.pass_context
def main_cli(ctx, log_config):
    """AI Ophthalmology Glaucoma Detection Pipeline CLI."""
    # Setup logging as the first step
    try:
        setup_logging(config_path=log_config)
        logger.info(f"Logging setup from {log_config}")
    except Exception as e:
        # Fallback to basic config if setup fails
        logging.basicConfig(level=logging.WARNING)
        logger.error(f"Failed to setup logging from {log_config}: {e}. Using basic config.", exc_info=True)
        
    # Ensure context object exists
    ctx.ensure_object(dict)
    # Load global config and pass to context
    # Phase-specific configs will be loaded within each command if needed
    try:
        global_cfg = load_config() # Load only global config initially
        if not global_cfg:
             logger.error("Failed to load global configuration (config.yaml). Cannot proceed.")
             ctx.fail("Global configuration loading failed.") # Exit CLI
        ctx.obj['CONFIG'] = global_cfg
        logger.info("Global configuration loaded.")
    except Exception as e:
        logger.error(f"Critical error loading global configuration: {e}", exc_info=True)
        ctx.fail("Global configuration loading failed.")


@main_cli.command()
@click.pass_context
def ingest(ctx):
    """Phase 01: Validate raw data directory existence."""
    logger.info("Executing Phase 01: Ingest")
    config = ctx.obj['CONFIG']
    
    # Dynamically import the ingest module only when needed
    ingest_module = import_phase("phase01_ingest")
    
    success = ingest_module.run_phase(config)
    if not success:
        ctx.fail("Phase 01: Ingest failed. Check logs.")
    logger.info("Phase 01: Ingest completed successfully.")


@main_cli.command()
@click.option('--run-id', default=None, help='Specify a run ID. If None, the next sequential ID is used.')
@click.pass_context
def conform(ctx, run_id: Optional[str]):
    """Phase 02: Validate raw structure and copy data to run directory."""
    logger.info("Executing Phase 02: Conform to ImageFolder")
    config = ctx.obj['CONFIG']
    
    # Dynamically import the conform module only when needed
    conform_module = import_phase("phase02_conformed_to_imagefolder")
    
    success = conform_module.run_phase(config, run_id=run_id)
    if not success:
        ctx.fail("Phase 02: Conform failed. Check logs.")
    logger.info("Phase 02: Conform completed successfully.")


@main_cli.command()
@click.option('--run-id', default=None, help='Specify the run ID for training. If None, the latest existing run ID is used.')
@click.pass_context
def train(ctx, run_id: Optional[str]):
    """Phase 03: Train the model."""
    logger.info("Executing Phase 03: Train")
    
    # Load global config merged with phase-specific config
    try:
        config = load_config(phase_config_path=PHASE_CONFIGS['train'])
        if not config:
             ctx.fail("Failed to load configuration for training phase.")
    except Exception as e:
         ctx.fail(f"Error loading training configuration: {e}")
    
    # If no run_id is provided, use the latest existing run
    if run_id is None:
        run_id = get_latest_run_id(config)
        if run_id is None:
            logger.error("No existing run IDs found. Please run the 'conform' phase first to create a run.")
            ctx.fail("No existing run IDs found.")
        else:
            logger.info(f"No run-id specified. Using latest run: {run_id}")
    
    # Dynamically import the train module only when needed
    train_module = import_phase("phase03_train_model")
    
    success = train_module.run_phase(config, run_id=run_id)
    if not success:
        ctx.fail("Phase 03: Train failed. Check logs.")
    logger.info("Phase 03: Train completed successfully.")


@main_cli.command()
@click.option('--run-id', default=None, help='Specify the run ID of the model to evaluate. If None, the latest existing run ID is used.')
@click.pass_context
def evaluate(ctx, run_id: Optional[str]):
    """Phase 04: Evaluate the trained model on the test set."""
    logger.info("Executing Phase 04: Evaluate")
    # Load global config merged with phase-specific config
    try:
        config = load_config(phase_config_path=PHASE_CONFIGS['evaluate'])
        if not config:
             ctx.fail("Failed to load configuration for evaluation phase.")
    except Exception as e:
         ctx.fail(f"Error loading evaluation configuration: {e}")
    
    # If no run_id is provided, use the latest existing run
    if run_id is None:
        run_id = get_latest_run_id(config)
        if run_id is None:
            logger.error("No existing run IDs found. Please run the 'train' phase first to create a model.")
            ctx.fail("No existing run IDs found.")
        else:
            logger.info(f"No run-id specified. Using latest run: {run_id}")
    
    # Dynamically import the evaluate module only when needed
    evaluate_module = import_phase("phase04_evaluate")
    
    success = evaluate_module.run_phase(config, run_id=run_id)
    if not success:
        ctx.fail("Phase 04: Evaluate failed. Check logs.")
    logger.info("Phase 04: Evaluate completed successfully.")


@main_cli.command()
@click.option('--run-id', default=None, help='Specify the run ID of the trained model to use. If None, the latest existing run ID is used.')
@click.option('--batch-id', default=None, help='Specify the batch ID for input/output. If None, the next sequential ID is used.')
@click.pass_context
def predict(ctx, run_id: Optional[str], batch_id: Optional[str]):
    """Phase 05: Predict on new images using a trained model."""
    logger.info("Executing Phase 05: Predict")
    # Load global config merged with phase-specific config
    try:
        config = load_config(phase_config_path=PHASE_CONFIGS['predict'])
        if not config:
             ctx.fail("Failed to load configuration for prediction phase.")
    except Exception as e:
         ctx.fail(f"Error loading prediction configuration: {e}")
    
    # If no run_id is provided, use the latest existing run
    if run_id is None:
        run_id = get_latest_run_id(config)
        if run_id is None:
            logger.error("No existing run IDs found. Please run the 'train' phase first to create a model.")
            ctx.fail("No existing run IDs found.")
        else:
            logger.info(f"No run-id specified. Using latest run: {run_id}")
    
    # Dynamically import the predict module only when needed
    predict_module = import_phase("phase05_predict")
    
    success = predict_module.run_phase(config, run_id=run_id, batch_id=batch_id)
    if not success:
        ctx.fail("Phase 05: Predict failed. Check logs.")
    logger.info("Phase 05: Predict completed successfully.")


# Note: main.py will import main_cli and run it.
# Example:
# if __name__ == '__main__':
#     # This allows running commands like: python src/cli.py ingest
#     main_cli()
