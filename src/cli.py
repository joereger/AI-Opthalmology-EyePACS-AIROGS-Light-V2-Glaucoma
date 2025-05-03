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
    'visualize_gradcam': 'src/pipeline/phase06_visualize_gradcam/config.yaml',
    'activation_maximization': 'src/pipeline/phase07_visualize_activation_maximization/config.yaml',
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
@click.pass_context
def predict(ctx, run_id: Optional[str]):
    """Phase 05: Predict on new images using a trained model.
    
    Images should be placed in data/05_predict/inference_input_default_dataset/"""
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
    
    success = predict_module.run_phase(config, run_id=run_id)
    if not success:
        ctx.fail("Phase 05: Predict failed. Check logs.")
    logger.info("Phase 05: Predict completed successfully.")


@main_cli.command()
@click.option('--run-id', default=None, help='Specify the run ID of the model/evaluation to visualize. If None, the latest existing run ID is used.')
@click.option('--batch-id', default=None, help='For prediction visualizations: batch ID to visualize. Only needed when source is "predict".')
@click.option('--source', type=click.Choice(['evaluate', 'predict']), default='evaluate', 
              help='Source of the predictions to visualize (evaluate or predict results).')
@click.option('--samples', type=int, default=None, 
              help='Number of random samples to visualize. If not specified, all available samples are used.')
@click.option('--filter', 'filter_type', type=click.Choice(['all', 'correct', 'incorrect']), default='all', 
              help='Filter which predictions to visualize. Only applies to evaluation visualizations.')
@click.pass_context
def gradcam(ctx, run_id: Optional[str], batch_id: Optional[str], source: str, samples: Optional[int], filter_type: str):
    """Phase 06: Generate GradCAM visualizations to explain model predictions.
    
    This phase creates visual explanations showing which regions of the retinal images 
    most influenced the model's glaucoma detection decisions.
    
    Examples:
        # Visualize evaluation results
        python main.py gradcam --run-id run_1
        
        # Visualize only incorrect predictions from evaluation
        python main.py gradcam --run-id run_1 --filter incorrect
        
        # Visualize prediction results
        python main.py gradcam --run-id run_1 --source predict --batch-id batch_1
        
        # Visualize only 10 random samples
        python main.py gradcam --run-id run_1 --samples 10
    """
    logger.info(f"Executing Phase 06: Visualize GradCAM ({source} source)")
    
    # Load global config merged with phase-specific config
    try:
        config = load_config(phase_config_path=PHASE_CONFIGS['visualize_gradcam'])
        if not config:
            ctx.fail("Failed to load configuration for GradCAM visualization phase.")
    except Exception as e:
        ctx.fail(f"Error loading GradCAM visualization configuration: {e}")
    
    # If no run_id is provided, use the latest existing run
    if run_id is None:
        run_id = get_latest_run_id(config)
        if run_id is None:
            logger.error("No existing run IDs found. Please run the 'train' phase first to create a model.")
            ctx.fail("No existing run IDs found.")
        else:
            logger.info(f"No run-id specified. Using latest run: {run_id}")
    
    # Validate batch_id requirement for 'predict' source
    if source == 'predict' and batch_id is None:
        logger.info("No batch-id specified for predict source visualization. Using default batch location.")
    
    # Dynamically import the gradcam visualization module
    gradcam_module = import_phase("phase06_visualize_gradcam")
    
    # Run the phase
    success = gradcam_module.run_phase(
        config, 
        run_id=run_id, 
        batch_id=batch_id, 
        source=source,
        samples=samples,
        filter_type=filter_type
    )
    
    if not success:
        ctx.fail("Phase 06: Visualize GradCAM failed. Check logs.")
    
    logger.info(f"Phase 06: Visualize GradCAM completed successfully. View results in data/06_visualize_gradcam/{run_id}/{source}/report.html")


@main_cli.command()
@click.option('--run-id', default=None, help='Specify the run ID of the model to analyze. If None, the latest existing run ID is used.')
# Add other relevant options if needed, e.g., overriding specific config values
@click.pass_context
def activation_maximization(ctx, run_id: Optional[str]):
    """Phase 07: Generate Activation Maximization visualizations.
    
    This phase analyzes a trained model to generate synthetic images 
    representing the features learned by specific neurons or channels. 
    It helps understand what patterns the model focuses on.
    
    Example:
        python main.py activation-maximization --run-id run_1
    """
    logger.info("Executing Phase 07: Visualize Activation Maximization")
    
    # Load global config merged with phase-specific config
    try:
        config = load_config(phase_config_path=PHASE_CONFIGS['activation_maximization'])
        if not config:
            ctx.fail("Failed to load configuration for Activation Maximization phase.")
    except Exception as e:
        ctx.fail(f"Error loading Activation Maximization configuration: {e}")
    
    # If no run_id is provided, use the latest existing run (from training phase)
    if run_id is None:
        # Assuming get_latest_run_id uses the 'train_model_dir' from global config if needed
        run_id = get_latest_run_id(config) 
        if run_id is None:
            logger.error("No existing run IDs found. Please run the 'train' phase first to create a model.")
            ctx.fail("No existing run IDs found.")
        else:
            logger.info(f"No run-id specified. Using latest run: {run_id}")
            
    # Dynamically import the activation maximization module
    activation_module = import_phase("phase07_visualize_activation_maximization")
    
    # Run the phase - assumes run_phase takes config_path, run_id, and global_config
    # Logging level is handled by main_cli setup
    global_config = ctx.obj['CONFIG'] # Get global config from context
    success = activation_module.run_phase(
        config_path=PHASE_CONFIGS['activation_maximization'], 
        run_id=run_id,
        global_config=global_config # Pass global config
    )
    
    if not success:
        ctx.fail("Phase 07: Visualize Activation Maximization failed. Check logs.")
    
    # Construct potential output path for user info (adjust based on actual output structure)
    output_base = config.get('output_dir_base', 'data/07_visualize_activation_maximization')
    # We don't know the exact run_X dir created inside run_phase here, so give general location
    logger.info(f"Phase 07: Visualize Activation Maximization completed successfully. Check results in {output_base}/run_X/")


# Note: main.py will import main_cli and run it.
# Example:
# if __name__ == '__main__':
#     # This allows running commands like: python src/cli.py ingest
#     main_cli()
