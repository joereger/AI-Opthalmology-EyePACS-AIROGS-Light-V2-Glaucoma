import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import pandas as pd
from PIL import Image
import shutil # Added for copying files
import cv2 # For potential image processing like blurring
from pathlib import Path
# Removed os import as it's only used in the moved generate_html_report
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm # For progress bars

# Project-specific imports (adjust as needed)
from src.data_handling.datasets import ImageFolderWithPaths # Assuming this dataset type
from src.data_handling.transforms import get_transforms # Corrected import
from src.utils.file_utils import ensure_dir_exists

logger = logging.getLogger(__name__)

# ============================================
# 1. Target Identification & Importance
# ============================================

def get_target_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    """
    Retrieves a layer module from the model using its string name.
    Handles names like 'features[N]', 'features[-N]', 'features[N][M]', 'named_module.sub_module'.
    """
    import re
    
    current_layer = model
    # Split by '.' or by '[index]' patterns, keeping the delimiters for indices
    parts = re.split(r'(\.|\[\s*-?\d+\s*\])', layer_name) 
    # Filter out None, empty strings, and '.' delimiters
    parts = [p for p in parts if p and p.strip() and p != '.'] 

    for part in parts:
        part = part.strip()
        if part.startswith('[') and part.endswith(']'):
            # Handle indexing e.g., '[0]', '[-1]'
            try:
                idx_str = part[1:-1].strip()
                idx = int(idx_str)
                if isinstance(current_layer, (torch.nn.Sequential, list)):
                    current_layer = current_layer[idx]
                else:
                    raise TypeError(f"Cannot index into layer of type {type(current_layer)} with index {idx} for part '{part}' in '{layer_name}'")
            except (IndexError, ValueError, TypeError) as e:
                 raise ValueError(f"Invalid index '{part}' in layer name '{layer_name}'. Error: {e}")
        else:
            # Handle attribute access e.g., 'features', 'conv'
            try:
                current_layer = getattr(current_layer, part)
            except AttributeError:
                 raise ValueError(f"Could not resolve attribute '{part}' in layer name '{layer_name}'")

    if isinstance(current_layer, torch.nn.Module):
        return current_layer
    else:
        # This case might happen if the path resolves to something other than a module
        raise ValueError(f"Resolved layer name '{layer_name}' did not result in a torch.nn.Module (got {type(current_layer)}).")


def calculate_importance_scores(
    model: torch.nn.Module, 
    config: Dict, # Phase-specific config
    source_run_id: str, # Run ID of the model being analyzed
    global_config: Dict, # Global config for base paths
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Identifies target units (neurons/channels) and calculates their importance 
    based on activation on the positive class samples from the corresponding validation set.
    
    Args:
        model: The loaded PyTorch model.
        config: The configuration dictionary for this phase.
        device: The device (CPU/GPU) to run calculations on.
        
    Returns:
        A list of dictionaries, each representing a target unit with its 
        layer name, index, and calculated importance score, sorted by importance.
    """
    logger.info("Calculating importance scores for target units...")
    target_configs = config.get('targets', [])
    try:
        # Get the target positive class INDEX from config
        positive_class_idx = int(config['positive_class_label']) 
    except (KeyError, ValueError):
        logger.error("Invalid or missing 'positive_class_label' in config. Must be an integer index (e.g., 0 or 1).")
        raise
    batch_size = config.get('find_real_examples', {}).get('batch_size', 32) 
    
    # Determine validation data path dynamically
    base_conformed_dir = Path(global_config.get('conformed_data_dir', 'data/02_conformed_to_imagefolder'))
    validation_data_dir = base_conformed_dir / source_run_id / 'validation'
    if not validation_data_dir.is_dir():
        raise FileNotFoundError(f"Validation data directory not found for run {source_run_id} at: {validation_data_dir}")
    logger.info(f"Using validation data from: {validation_data_dir}")

    # Get image size and normalization from loaded metadata
    loaded_metadata = config.get('loaded_metadata', {})
    if 'input_size' not in loaded_metadata or len(loaded_metadata['input_size']) != 3:
         raise ValueError("Valid 'input_size' [C, H, W] not found in loaded model metadata.")
    img_size = loaded_metadata['input_size'][1] # Get H from [C, H, W]
    
    mean = loaded_metadata.get('normalization', {}).get('mean', [0.485, 0.456, 0.406])
    std = loaded_metadata.get('normalization', {}).get('std', [0.229, 0.224, 0.225])
    logger.info(f"Using image size: {img_size}x{img_size}, mean: {mean}, std: {std}")

    # Get validation transforms
    try:
        # Pass global config to get_transforms as it might read other params like imagenet_mean/std
        all_transforms = get_transforms(global_config) 
        data_transforms = all_transforms['validation'] # Get the validation transforms
    except Exception as e:
         logger.error(f"Failed to get validation transforms: {e}. Using basic fallback.", exc_info=True)
         # Fallback might still be useful if get_transforms fails for some reason
         data_transforms = transforms.Compose([
             transforms.Resize((img_size, img_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)
         ])

    # Load the full validation dataset first to get class mapping and filter
    try:
        full_val_dataset = ImageFolderWithPaths(str(validation_data_dir), transform=data_transforms)
        if not full_val_dataset:
             raise ValueError("Validation dataset is empty or could not be loaded.")

        # Verify the target positive_class_idx (read from config) exists in the loaded dataset's indices
        if positive_class_idx not in full_val_dataset.class_to_idx.values():
             # Raise error if the configured index is not a valid index in the dataset
             raise ValueError(f"Configured positive_class_label index '{positive_class_idx}' not found in dataset indices: {list(full_val_dataset.class_to_idx.values())}. Class map: {full_val_dataset.class_to_idx}")

        # Find the class name corresponding to the index for logging purposes
        positive_class_name = next((name for name, idx in full_val_dataset.class_to_idx.items() if idx == positive_class_idx), None)
        if positive_class_name is None:
             # This case should ideally not happen if the previous check passed, but added for safety
             raise RuntimeError(f"Could not find class name for valid index {positive_class_idx}. Class map: {full_val_dataset.class_to_idx}")
        logger.info(f"Targeting positive class '{positive_class_name}' with index {positive_class_idx}.")

        # Filter the dataset samples to keep only the positive class (using the index)
        positive_indices = [i for i, (_, label) in enumerate(full_val_dataset.samples) if label == positive_class_idx]
        if not positive_indices:
             raise ValueError(f"No samples found for the positive class index {positive_class_idx} ('{positive_class_name}') in the validation set.")
             
        positive_dataset = torch.utils.data.Subset(full_val_dataset, positive_indices)
        
        # Create DataLoader for the positive class subset
        num_workers = global_config.get('num_workers', 1) # Get num_workers from global config
        dataloader = DataLoader(positive_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        logger.info(f"Created DataLoader with {len(positive_dataset)} images from positive class '{positive_class_name}' (index {positive_class_idx}) for importance calculation.")
    except Exception as e:
        # Corrected error message to use existing validation_data_dir variable
        logger.error(f"Failed to load or process dataset from {validation_data_dir}: {e}", exc_info=True)
        raise

    all_units = []
    model.eval() # Ensure model is in eval mode

    # Store activations per layer
    layer_activations = {target['layer_name']: [] for target in target_configs}
    hooks = []

    def get_hook(layer_name):
        def hook(module, input, output):
            # Detach and move to CPU to save GPU memory during accumulation
            layer_activations[layer_name].append(output.detach().cpu())
        return hook

    # Register forward hooks for all target layers
    for target in target_configs:
        layer_name = target['layer_name']
        try:
            layer = get_target_layer(model, layer_name)
            hooks.append(layer.register_forward_hook(get_hook(layer_name)))
            logger.debug(f"Registered hook for layer: {layer_name}")
        except ValueError as e:
            logger.error(f"Skipping target layer {layer_name}: {e}")
            # Remove from layer_activations if hook failed
            if layer_name in layer_activations: del layer_activations[layer_name]


    # --- Run positive class data through model to collect activations ---
    logger.info("Running positive class data through model to collect activations...")
    with torch.no_grad():
        # Dataloader for Subset returns original data tuple (img, label, path)
        for inputs, _, _ in tqdm(dataloader, desc="Collecting Activations"): 
            inputs = inputs.to(device)
            _ = model(inputs) # Forward pass to trigger hooks

    # Remove hooks
    for handle in hooks:
        handle.remove()
    logger.debug("Removed forward hooks.")

    # --- Calculate importance scores ---
    logger.info("Calculating importance scores...")
    for target in target_configs:
        layer_name = target['layer_name']
        if layer_name not in layer_activations or not layer_activations[layer_name]:
            logger.warning(f"No activations collected for layer {layer_name}. Skipping.")
            continue
            
        # Concatenate activations from all batches for this layer
        activations = torch.cat(layer_activations[layer_name], dim=0) # Shape: (N, C, H, W) or (N, Features)
        
        # Calculate mean activation per channel/neuron across the dataset
        # Average over batch (N) and spatial dims (H, W) if they exist
        if activations.dim() == 4: # Conv layer
            mean_activations = activations.mean(dim=(0, 2, 3)) # Shape: (C,)
        elif activations.dim() == 2: # Linear layer / Flattened features
            mean_activations = activations.mean(dim=0) # Shape: (Features,)
        else:
            logger.warning(f"Unexpected activation shape {activations.shape} for layer {layer_name}. Skipping.")
            continue

        num_units = mean_activations.shape[0]
        logger.info(f"Layer {layer_name}: Found {num_units} units. Mean activation range: [{mean_activations.min():.4f}, {mean_activations.max():.4f}]")

        # Select top N units based on mean activation
        num_top_units = min(target.get('num_top_units', num_units), num_units) # Ensure not asking for more than available
        
        # TODO: Implement other selection methods if needed (e.g., weight magnitude)
        if target['unit_selection_method'] == 'mean_activation_positive_class':
            # Sort by mean activation (descending)
            sorted_indices = torch.argsort(mean_activations, descending=True)
            top_indices = sorted_indices[:num_top_units]
            
            for i, unit_index in enumerate(top_indices):
                unit_index = unit_index.item() # Convert tensor index to int
                score = mean_activations[unit_index].item()
                all_units.append({
                    'layer_name': layer_name,
                    'unit_index': unit_index, # This is the channel/neuron index
                    'importance_score': score,
                    'rank': i + 1
                })
        else:
             logger.warning(f"Unsupported unit_selection_method: {target['unit_selection_method']} for layer {layer_name}")

    # Sort all collected units by importance score (descending)
    all_units.sort(key=lambda x: x['importance_score'], reverse=True)
    
    # Assign overall rank
    for i, unit in enumerate(all_units):
        unit['overall_rank'] = i + 1

    logger.info(f"Identified and ranked {len(all_units)} target units across all layers.")
    return all_units


# ============================================
# 2. Activation Maximization Core
# ============================================

def tv_loss(img, beta=2.0):
    """Total Variation loss for image regularization."""
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    loss = torch.sum(dx**beta) + torch.sum(dy**beta)
    return loss

def perform_activation_maximization(
    model: torch.nn.Module, 
    unit_info: Dict, 
    opt_config: Dict, 
    config: Dict, # Added config parameter to access loaded_metadata
    output_dir: Path, 
    device: torch.device
) -> Optional[Path]:
    """
    Generates a synthetic image that maximizes the activation of a specific unit.
    
    Args:
        model: The loaded PyTorch model.
        unit_info: Dict containing 'layer_name' and 'unit_index'.
        opt_config: Configuration dictionary for the optimization process.
        output_dir: Directory to save the generated visualization.
        device: The device (CPU/GPU) to run calculations on.
        
    Returns:
        Path to the saved visualization image, or None if failed.
    """
    layer_name = unit_info['layer_name']
    unit_index = unit_info['unit_index']
    # Get image size from metadata stored in phase config
    img_size = config.get('loaded_metadata', {}).get('input_size', [None, 512, None])[1] 
    if img_size is None:
         logger.warning("Could not determine image size from metadata for optimization. Defaulting to 512.")
         img_size = 512
    steps = opt_config['steps']
    lr = opt_config['learning_rate']
    l2_decay = opt_config.get('regularization', {}).get('l2_decay', 0.0)
    tv_weight = opt_config.get('regularization', {}).get('total_variation', {}).get('weight', 0.0)
    tv_beta = opt_config.get('regularization', {}).get('total_variation', {}).get('beta', 2.0)
    blur_freq = opt_config.get('blur_frequency', 0)
    blur_sigma = opt_config.get('blur_sigma', 0.5)
    initial_image_type = opt_config.get('initial_image', 'noise')

    try:
        target_layer = get_target_layer(model, layer_name)
    except ValueError as e:
        logger.error(f"Cannot perform activation maximization: {e}")
        return None

    # Initialize input image
    if initial_image_type == 'noise':
        input_img = torch.randn(1, 3, img_size, img_size, device=device) * 0.1 + 0.5
    # elif initial_image_type == 'mean':
        # TODO: Calculate mean image from dataset if needed
        # input_img = mean_image_tensor.clone().unsqueeze(0).to(device)
    else:
         logger.warning(f"Unknown initial_image type: {initial_image_type}. Using noise.")
         input_img = torch.randn(1, 3, img_size, img_size, device=device) * 0.1 + 0.5
         
    input_img.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([input_img], lr=lr, weight_decay=l2_decay)

    # Hook to capture the target layer's activation
    activation = None
    def hook(module, input, output):
        nonlocal activation
        activation = output
    
    handle = target_layer.register_forward_hook(hook)

    logger.debug(f"Starting optimization for {layer_name} unit {unit_index}...")
    for i in range(steps):
        optimizer.zero_grad()
        
        # Forward pass
        _ = model(input_img)
        
        if activation is None:
            logger.error("Activation hook did not run. Aborting optimization.")
            handle.remove()
            return None
            
        # Loss is the negative activation of the target unit
        # Handle spatial dimensions if it's a conv layer activation (N, C, H, W)
        if activation.dim() == 4:
            # Maximize the mean activation across spatial dimensions for the target channel
            loss = -activation[0, unit_index].mean() 
        elif activation.dim() == 2:
             # Maximize the activation of the target neuron (N, Features)
             loss = -activation[0, unit_index]
        else:
             logger.error(f"Unexpected activation shape {activation.shape}. Aborting.")
             handle.remove()
             return None

        # Add Total Variation regularization
        if tv_weight > 0:
            loss += tv_weight * tv_loss(input_img, tv_beta)
            
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Optional: Apply Gaussian blur periodically
        if blur_freq > 0 and (i + 1) % blur_freq == 0:
             with torch.no_grad():
                 # Convert to numpy, blur, convert back
                 img_np = input_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                 # Clamp values before blurring? Maybe not necessary.
                 # img_np = np.clip(img_np, 0, 1) # Assuming image is roughly in [0,1]
                 blurred_np = cv2.GaussianBlur(img_np, (0,0), sigmaX=blur_sigma, sigmaY=blur_sigma)
                 input_img.data = torch.from_numpy(blurred_np).permute(2, 0, 1).unsqueeze(0).to(device)


        # Optional: Clamp image values (might not be needed with Adam/regularization)
        # with torch.no_grad():
        #     input_img.clamp_(0, 1) 

    handle.remove() # Clean up hook
    logger.debug(f"Optimization finished for {layer_name} unit {unit_index}.")

    # Save the resulting image
    output_filename = f"{layer_name.replace('.', '_').replace('[', '').replace(']', '')}_unit_{unit_index}.png"
    output_path = output_dir / output_filename
    
    # Normalize image to [0, 1] for saving, based on its actual range
    img_data = input_img.detach().squeeze(0)
    min_val = img_data.min()
    max_val = img_data.max()
    if max_val > min_val:
        img_data = (img_data - min_val) / (max_val - min_val)
    else:
        img_data = torch.zeros_like(img_data) # Handle case of flat image

    try:
        save_image(img_data, output_path)
        logger.info(f"Saved visualization to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save image {output_path}: {e}")
        return None


# ============================================
# 3. Find Real Examples
# ============================================

def find_real_examples(
    model: torch.nn.Module, 
    visualization_results: List[Dict], 
    config: Dict, # Phase-specific config
    source_run_id: str, # Run ID of the model being analyzed
    global_config: Dict, # Global config for base paths
    examples_high_dir: Path, 
    examples_low_dir: Path, 
    device: torch.device
) -> None:
    """
    Finds real images from the corresponding validation dataset that 
    maximally and minimally activate the target units.
    
    Args:
        model: The loaded PyTorch model.
        visualization_results: List of dicts, each containing info about a visualized unit.
        config: The configuration dictionary for this phase.
        examples_high_dir: Directory to save high-activation examples.
        examples_low_dir: Directory to save low-activation examples.
        device: The device (CPU/GPU) to run calculations on.
    """
    logger.info("Finding real examples for visualized units...")
    find_config = config.get('find_real_examples', {})
    num_examples = find_config.get('num_examples', 5)
    # Determine validation data path dynamically
    base_conformed_dir = Path(global_config.get('conformed_data_dir', 'data/02_conformed_to_imagefolder'))
    validation_data_dir = base_conformed_dir / source_run_id / 'validation'
    if not validation_data_dir.is_dir():
        logger.error(f"Validation data directory not found for run {source_run_id} at: {validation_data_dir}. Skipping example finding.")
        return

    logger.info(f"Using validation data from: {validation_data_dir} for finding examples.")
    batch_size = find_config.get('batch_size', 32)
    
    # Removed incorrect check using undefined 'data_dir'

    # Get transforms (same as used for importance calculation, using metadata)
    loaded_metadata = config.get('loaded_metadata', {})
    mean = loaded_metadata.get('normalization', {}).get('mean', [0.485, 0.456, 0.406])
    std = loaded_metadata.get('normalization', {}).get('std', [0.229, 0.224, 0.225])
    img_size = loaded_metadata.get('input_size', [None, 512, None])[1]
    if img_size is None: img_size = 512 # Fallback

    try:
        all_transforms = get_transforms(global_config) # Use global config
        data_transforms = all_transforms['validation'] 
    except Exception as e:
         logger.error(f"Failed to get validation transforms for examples: {e}. Using basic fallback.", exc_info=True)
         data_transforms = transforms.Compose([
             transforms.Resize((img_size, img_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)
         ])

    # Load the full validation dataset
    try:
        # Use ImageFolderWithPaths to get original file paths
        dataset = ImageFolderWithPaths(str(validation_data_dir), transform=data_transforms)
        if not dataset:
             raise ValueError("Validation dataset is empty.")
        num_workers = global_config.get('num_workers', 1) # Get from global config
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        logger.info(f"Loaded {len(dataset)} images from {validation_data_dir} for finding examples.")
    except Exception as e:
        # Use correct variable 'validation_data_dir' in error message
        logger.error(f"Failed to load dataset from {validation_data_dir}: {e}", exc_info=True) 
        return

    # --- Collect activations for all target units across the dataset ---
    model.eval()
    target_layers = list(set(unit['layer_name'] for unit in visualization_results))
    layer_activations = {layer_name: [] for layer_name in target_layers}
    all_image_paths = []
    hooks = []

    def get_hook(layer_name):
        def hook(module, input, output):
            layer_activations[layer_name].append(output.detach().cpu())
        return hook

    for layer_name in target_layers:
        try:
            layer = get_target_layer(model, layer_name)
            hooks.append(layer.register_forward_hook(get_hook(layer_name)))
        except ValueError as e:
            logger.error(f"Cannot register hook for layer {layer_name}: {e}")
            if layer_name in layer_activations: del layer_activations[layer_name]

    logger.info("Running full dataset through model to collect activations for example finding...")
    with torch.no_grad():
        for inputs, _, paths in tqdm(dataloader, desc="Collecting Activations for Examples"):
            inputs = inputs.to(device)
            _ = model(inputs)
            all_image_paths.extend(list(paths))

    for handle in hooks:
        handle.remove()

    # Concatenate activations
    for layer_name in list(layer_activations.keys()): # Use list copy for safe deletion
         if layer_activations[layer_name]:
             layer_activations[layer_name] = torch.cat(layer_activations[layer_name], dim=0)
         else:
             logger.warning(f"No activations collected for layer {layer_name} during example finding. Removing.")
             del layer_activations[layer_name]
             # Remove units associated with this layer from visualization_results?
             visualization_results = [unit for unit in visualization_results if unit['layer_name'] != layer_name]


    # --- Find and save examples for each unit ---
    logger.info("Identifying and saving high/low activation examples...")
    for unit_info in tqdm(visualization_results, desc="Finding Examples"):
        layer_name = unit_info['layer_name']
        unit_index = unit_info['unit_index']
        
        if layer_name not in layer_activations:
            logger.warning(f"Skipping examples for {layer_name} unit {unit_index} as activations were not collected.")
            continue
            
        activations = layer_activations[layer_name] # Shape (N, C, H, W) or (N, Features)
        
        # Get the activation for the specific unit across all images
        if activations.dim() == 4:
            unit_activations = activations[:, unit_index, :, :].mean(dim=(1, 2)) # Mean spatial activation (N,)
        elif activations.dim() == 2:
            unit_activations = activations[:, unit_index] # (N,)
        else:
            continue # Should not happen if previous checks passed

        # Sort images by activation for this unit
        sorted_indices = torch.argsort(unit_activations, descending=True)
        
        # Get top N activating images
        top_indices = sorted_indices[:num_examples]
        unit_info['high_activation_examples'] = []
        for idx in top_indices:
            img_path = Path(all_image_paths[idx.item()])
            activation_value = unit_activations[idx.item()].item()
            # Copy original image to output dir
            dest_filename = f"{layer_name.replace('.', '_').replace('[', '').replace(']', '')}_unit_{unit_index}_high_{img_path.name}"
            dest_path = examples_high_dir / dest_filename
            try:
                shutil.copy(img_path, dest_path) # Use imported shutil
                unit_info['high_activation_examples'].append({
                    'path': str(dest_path), # Store path relative to CWD initially
                    'activation': activation_value
                })
            except Exception as e:
                 logger.error(f"Failed to copy image {img_path} to {dest_path}: {e}")

        # Get bottom N activating images
        bottom_indices = sorted_indices[-num_examples:]
        unit_info['low_activation_examples'] = []
        for idx in bottom_indices:
            img_path = Path(all_image_paths[idx.item()])
            activation_value = unit_activations[idx.item()].item()
            dest_filename = f"{layer_name.replace('.', '_').replace('[', '').replace(']', '')}_unit_{unit_index}_low_{img_path.name}"
            dest_path = examples_low_dir / dest_filename
            try:
                shutil.copy(img_path, dest_path) # Use imported shutil
                unit_info['low_activation_examples'].append({
                    'path': str(dest_path), # Store path relative to CWD initially
                    'activation': activation_value
                })
            except Exception as e:
                 logger.error(f"Failed to copy image {img_path} to {dest_path}: {e}")
                 
    logger.info("Finished finding and saving real examples.")

# HTML Report Generation moved to html_report_generator.py
