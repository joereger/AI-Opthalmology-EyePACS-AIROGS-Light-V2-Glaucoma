import logging
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional, Any, Union

# Project-specific imports
from src.config_loader import load_config
from src.utils.file_utils import ensure_dir_exists
from src.models.mobilenetv3.model import get_model
from src.pipeline.phase06_visualize_gradcam.utils import (
    load_image, preprocess_image, save_visualization, generate_html_report
)

logger = logging.getLogger(__name__)


def get_image_path(
    image_info: Dict, 
    source_type: str, 
    run_id: str, 
    batch_id: Optional[str] = None
) -> Path:
    """
    Get the path to the image file based on source type and image info.
    
    Args:
        image_info: Dictionary containing image information
        source_type: 'evaluate' or 'predict'
        run_id: Run ID
        batch_id: Optional batch ID for prediction (only needed for 'predict' source)
        
    Returns:
        Path: Path to the image file
    """
    if source_type == 'evaluate':
        # For evaluate, images are from test set
        class_name = image_info['true_label']
        image_name = image_info['image_path']
        return Path(f'data/02_conformed_to_imagefolder/{run_id}/test/{class_name}/{image_name}')
    else:
        # For predict, images are from predict input directory
        image_name = image_info['image_path']
        if batch_id:
            # Custom batch location
            return Path(f'data/05_predict/{run_id}/{batch_id}/inference_input/{image_name}')
        else:
            # Default dataset location
            return Path(f'data/05_predict/inference_input_default_dataset/{image_name}')


def load_model_and_metadata(run_id: str, config: Dict) -> Tuple[torch.nn.Module, Dict]:
    """
    Load the trained model and its metadata.
    
    Args:
        run_id: Run ID of the model to load
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, metadata)
    """
    # Paths to model and metadata
    model_dir = Path(config.get('train_model_dir', 'data/03_train_model'))
    model_name = config.get('model_name', 'mobilenetv3')
    model_path = model_dir / run_id / model_name / 'best_model.pth'
    metadata_path = model_dir / run_id / model_name / 'model_metadata.json'
    
    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load metadata if it exists
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                logger.info(f"Loaded model metadata from {metadata_path}")
        except Exception as e:
            logger.warning(f"Could not load model metadata: {e}. Using default configuration.")
    
    # Get the number of classes
    num_classes = metadata.get('num_classes', config.get('num_classes', 2))
    
    # Load model
    logger.info(f"Loading model from {model_path} with {num_classes} classes")
    model = get_model(num_classes=num_classes, pretrained=False)
    
    # Load weights
    device = get_device()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, metadata


def get_device() -> torch.device:
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_predictions(
    run_id: str, 
    source_type: str, 
    config: Dict, 
    batch_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Load prediction results from a previous phase.
    
    Args:
        run_id: Run ID
        source_type: 'evaluate' or 'predict'
        config: Configuration dictionary
        batch_id: Optional batch ID for prediction
        
    Returns:
        DataFrame containing predictions
    """
    if source_type == 'evaluate':
        # Load test predictions from evaluate phase - standardized filename
        predictions_path = Path(config.get('evaluate_dir', 'data/04_evaluate')) / run_id / 'predictions.csv'
    else:
        # Load predictions from predict phase
        predict_dir = Path(config.get('predict_dir', 'data/05_predict'))
        predictions_path = predict_dir / run_id / batch_id / 'inference_output' / 'predictions.csv'
    
    # Check if file exists
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found at {predictions_path}")
    
    # Load predictions
    logger.info(f"Loading predictions from {predictions_path}")
    return pd.read_csv(predictions_path)


def filter_predictions(
    predictions_df: pd.DataFrame,
    filter_type: str,
    samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Filter predictions based on filter type and optionally sample a subset.
    
    Args:
        predictions_df: DataFrame containing predictions
        filter_type: 'all', 'correct', or 'incorrect'
        samples: Optional number of samples to select
        
    Returns:
        Filtered DataFrame
    """
    # Apply filter
    if filter_type == 'correct' and 'true_label' in predictions_df.columns:
        filtered_df = predictions_df[predictions_df['true_label'] == predictions_df['predicted_label']]
    elif filter_type == 'incorrect' and 'true_label' in predictions_df.columns:
        filtered_df = predictions_df[predictions_df['true_label'] != predictions_df['predicted_label']]
    else:
        filtered_df = predictions_df
    
    # Check if we need to sample
    if samples is not None and samples < len(filtered_df):
        return filtered_df.sample(samples, random_state=42)
    
    return filtered_df


def generate_gradcam_visualizations(
    model: torch.nn.Module,
    predictions_df: pd.DataFrame,
    source_type: str,
    run_id: str,
    config: Dict,
    batch_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate GradCAM visualizations for the given predictions.
    
    Args:
        model: The trained model
        predictions_df: DataFrame containing predictions
        source_type: 'evaluate' or 'predict'
        run_id: Run ID
        config: Configuration dictionary
        batch_id: Optional batch ID
        
    Returns:
        List of dictionaries with visualization data
    """
    # Get configuration values
    gradcam_layer_name = config.get('gradcam_layer_name', 'features[-1]')
    overlay_alpha = config.get('overlay_alpha', 0.6)
    colormap_name = config.get('colormap', 'jet')
    device = get_device()
    class_names = ['NRG', 'RG']  # TODO: Make this configurable or load from metadata
    
    # Get model normalization parameters
    normalization = {}
    if 'metadata' in config and 'normalization' in config['metadata']:
        normalization = config['metadata']['normalization']
    mean = normalization.get('mean', [0.485, 0.456, 0.406])
    std = normalization.get('std', [0.229, 0.224, 0.225])
    
    # Set up output directory
    visualize_dir = Path(config.get('visualize_gradcam_dir', 'data/06_visualize_gradcam'))
    visualize_run_dir = visualize_dir / run_id / source_type
    if batch_id and source_type == 'predict':
        visualize_run_dir = visualize_run_dir / batch_id
    
    # Create directory for each class and prediction outcome if evaluate
    visualizations = []
    
    # Process each image
    for i, row in predictions_df.iterrows():
        # Get image path
        image_path = get_image_path(row, source_type, run_id, batch_id)
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}, skipping.")
            continue
        
        # Get prediction info
        predicted_label = row['predicted_label']
        true_label = row.get('true_label', '')
        
        # Determine output directory based on prediction outcome
        if source_type == 'evaluate':
            if predicted_label == true_label:
                output_dir = visualize_run_dir / f"{predicted_label}_correct"
            else:
                output_dir = visualize_run_dir / f"{predicted_label}_incorrect"
        else:
            output_dir = visualize_run_dir / f"predicted_{predicted_label}"
        
        # Ensure output directory exists
        ensure_dir_exists(output_dir)
        
        # Generate filename
        filename = Path(image_path).stem
        
        try:
            # Load and preprocess image
            original_image = load_image(image_path)
            preprocessed_image = preprocess_image(original_image, mean, std, device)
            
            # Get GradCAM heatmap
            target_class = class_names.index(predicted_label)
            heatmap = model.get_gradcam(preprocessed_image, gradcam_layer_name, target_class)
            
            # Resize heatmap to original image size if needed
            if heatmap.shape != original_image.shape[:2]:
                import cv2
                heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
            # Save visualizations
            original_path, heatmap_path, overlay_path = save_visualization(
                original_image, heatmap, output_dir, filename, 
                overlay_alpha, colormap_name
            )
            
            # Get confidence
            if f'probability_{predicted_label}' in row:
                confidence = row[f'probability_{predicted_label}']
            else:
                # If we don't have class-specific probabilities, use 1.0
                confidence = 1.0
            
            # Add to visualizations list
            vis_info = {
                'image_id': filename,
                'image_path': str(image_path),
                'prediction': predicted_label,
                'confidence': confidence,
                'original_path': str(original_path),
                'heatmap_path': str(heatmap_path),
                'overlay_path': str(overlay_path)
            }
            
            # Add true label if available
            if true_label:
                vis_info['actual'] = true_label
            
            visualizations.append(vis_info)
            
        except Exception as e:
            logger.error(f"Error generating visualization for {image_path}: {e}")
    
    return visualizations


def run_phase(
    config: Dict[str, Any], 
    run_id: str, 
    batch_id: Optional[str] = None, 
    source: str = 'evaluate',
    samples: Optional[int] = None,
    filter_type: str = 'all'
) -> bool:
    """
    Executes Phase 06: Visualize GradCAM.
    
    Args:
        config: Configuration dictionary
        run_id: Run ID of the model/evaluation to visualize
        batch_id: Batch ID for prediction (only needed for 'predict' source)
        source: Source of predictions ('evaluate' or 'predict')
        samples: Optional number of random samples to visualize
        filter_type: Type of predictions to visualize ('all', 'correct', 'incorrect')
        
    Returns:
        bool: Success status
    """
    logger.info(f"--- Starting Phase 06: Visualize GradCAM for run_id='{run_id}', source='{source}' ---")
    
    try:
        # Load model and metadata
        model, metadata = load_model_and_metadata(run_id, config)
        
        # Update config with metadata
        config['metadata'] = metadata
        
        # Load predictions
        predictions_df = load_predictions(run_id, source, config, batch_id)
        
        # Filter predictions
        filtered_df = filter_predictions(predictions_df, filter_type, samples)
        logger.info(f"Selected {len(filtered_df)} images for visualization")
        
        if len(filtered_df) == 0:
            logger.warning("No images selected for visualization after filtering.")
            return False
        
        # Generate GradCAM visualizations
        visualizations = generate_gradcam_visualizations(
            model, filtered_df, source, run_id, config, batch_id
        )
        
        # Create HTML report
        if visualizations:
            visualize_dir = Path(config.get('visualize_gradcam_dir', 'data/06_visualize_gradcam'))
            report_dir = visualize_dir / run_id / source
            if batch_id and source == 'predict':
                report_dir = report_dir / batch_id
            
            ensure_dir_exists(report_dir)
            report_path = report_dir / 'report.html'
            
            # Get report title and description
            title = config.get('report_title', 'Grad-CAM Visualization Report')
            description = config.get('report_description', 
                "Visual explanations showing which regions of the retinal images "
                "influenced the model's glaucoma detection decisions.")
            
            # Generate report
            generate_html_report(
                visualizations,
                report_path,
                title=f"{title} - Run {run_id}",
                description=description,
                thumbnail_size=config.get('thumbnail_size', 128)
            )
            
            logger.info(f"HTML report generated at: {report_path}")
            return True
        else:
            logger.warning("No visualizations were generated.")
            return False
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during Phase 06: {e}", exc_info=True)
        return False
