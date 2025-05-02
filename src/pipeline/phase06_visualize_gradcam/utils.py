import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import logging
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

def load_image(image_path: Union[str, Path], resize: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file
        resize: Optional tuple of (height, width) to resize the image
        
    Returns:
        np.ndarray: Loaded image in RGB format with shape [H, W, 3]
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
        if resize:
            img = img.resize(resize, Image.LANCZOS)
        return np.array(img)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise

def preprocess_image(
    image: np.ndarray, 
    mean: List[float] = [0.485, 0.456, 0.406], 
    std: List[float] = [0.229, 0.224, 0.225],
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Preprocess an image for model input.
    
    Args:
        image: Input image as numpy array [H, W, 3] in RGB format
        mean: Normalization mean for each channel
        std: Normalization standard deviation for each channel
        device: Device to place the tensor on
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input [1, 3, H, W]
    """
    # Convert to float and normalize to [0, 1]
    img = image.astype(np.float32) / 255.0
    
    # Convert from [H, W, C] to [C, H, W]
    img = img.transpose(2, 0, 1)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).float()
    
    # Normalize using mean and std
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    img_tensor = (img_tensor - mean_tensor) / std_tensor
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor.to(device)

def deprocess_image(
    image: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406], 
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Convert a normalized tensor back to a numpy image.
    
    Args:
        image: Input tensor with shape [1, 3, H, W] or [3, H, W]
        mean: Normalization mean for each channel
        std: Normalization standard deviation for each channel
        
    Returns:
        np.ndarray: Denormalized image as numpy array with shape [H, W, 3] in RGB format
    """
    # Remove batch dimension if present
    if image.dim() == 4:
        image = image.squeeze(0)
    
    # Move to CPU if needed
    image = image.cpu().clone()
    
    # Denormalize
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    image = image * std_tensor + mean_tensor
    
    # Convert to numpy, clamp to [0, 1], and transpose to [H, W, C]
    image = image.numpy()
    image = np.clip(image, 0, 1)
    image = image.transpose(1, 2, 0)
    
    # Convert to uint8 in range [0, 255]
    return (image * 255).astype(np.uint8)

def apply_colormap(
    heatmap: np.ndarray, 
    colormap_name: str = 'jet'
) -> np.ndarray:
    """
    Apply a colormap to a grayscale heatmap.
    
    Args:
        heatmap: Grayscale heatmap as numpy array with shape [H, W], values in [0, 1]
        colormap_name: Name of the matplotlib colormap to use
        
    Returns:
        np.ndarray: Colorized heatmap with shape [H, W, 3], values in [0, 255]
    """
    # Create colormap
    colormap = cm.get_cmap(colormap_name)
    
    # Apply colormap - returns [H, W, 4] RGBA with values in [0, 1]
    colored_heatmap = colormap(heatmap)
    
    # Convert to RGB and scale to [0, 255]
    colored_heatmap = colored_heatmap[:, :, :3] * 255
    
    return colored_heatmap.astype(np.uint8)

def create_overlay(
    image: np.ndarray, 
    heatmap: np.ndarray, 
    alpha: float = 0.6
) -> np.ndarray:
    """
    Create an overlay of the original image and a heatmap.
    
    Args:
        image: Original image as numpy array with shape [H, W, 3], values in [0, 255]
        heatmap: Colorized heatmap as numpy array with shape [H, W, 3], values in [0, 255]
        alpha: Transparency of the heatmap (0.0 = transparent, 1.0 = opaque)
        
    Returns:
        np.ndarray: Overlay image with shape [H, W, 3], values in [0, 255]
    """
    # Resize heatmap to match image size if needed
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Create the overlay using alpha blending
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay

def save_visualization(
    image: np.ndarray,
    heatmap: np.ndarray,
    output_dir: Union[str, Path],
    filename: str,
    overlay_alpha: float = 0.6,
    colormap_name: str = 'jet'
) -> Tuple[Path, Path, Path]:
    """
    Save the original image, colorized heatmap, and overlay to disk.
    
    Args:
        image: Original image as numpy array with shape [H, W, 3]
        heatmap: Grayscale heatmap as numpy array with shape [H, W], values in [0, 1]
        output_dir: Directory to save the visualizations
        filename: Base filename to use (without extension)
        overlay_alpha: Transparency for the overlay
        colormap_name: Colormap to use for the heatmap
        
    Returns:
        Tuple of Paths to the saved original image, heatmap, and overlay
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original image
    original_path = output_dir / f"{filename}_original.png"
    Image.fromarray(image).save(original_path)
    
    # Colorize heatmap and save
    colored_heatmap = apply_colormap(heatmap, colormap_name)
    heatmap_path = output_dir / f"{filename}_heatmap.png"
    Image.fromarray(colored_heatmap).save(heatmap_path)
    
    # Create and save overlay
    overlay = create_overlay(image, colored_heatmap, overlay_alpha)
    overlay_path = output_dir / f"{filename}_overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    
    return original_path, heatmap_path, overlay_path

def generate_html_report(
    visualizations: List[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "GradCAM Visualization Report",
    description: str = "",
    thumbnail_size: int = 128
) -> None:
    """
    Generate an HTML report with GradCAM visualizations.
    
    Args:
        visualizations: List of dicts with visualization metadata. Each dict should have:
            - 'image_id': Identifier for the image
            - 'prediction': Predicted class
            - 'confidence': Prediction confidence
            - 'actual': Actual class (if available)
            - 'original_path': Path to original image
            - 'heatmap_path': Path to heatmap image
            - 'overlay_path': Path to overlay image
        output_path: Path to save the HTML report
        title: Report title
        description: Report description
        thumbnail_size: Size of thumbnails in the report
    """
    output_path = Path(output_path)
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Start building HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }}
            .filters {{
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
            .visualizations {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 25px;
            }}
            .vis-card {{
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s ease;
            }}
            .vis-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }}
            .card-header {{
                padding: 10px 15px;
                background-color: #f0f0f0;
                font-weight: bold;
                border-bottom: 1px solid #ddd;
            }}
            .card-content {{
                padding: 15px;
            }}
            .vis-images {{
                display: flex;
                flex-direction: column;
                gap: 10px;
                margin-top: 10px;
            }}
            .vis-image {{
                text-align: center;
            }}
            .vis-image img {{
                max-width: 100%;
                height: auto;
                cursor: pointer;
                border-radius: 4px;
            }}
            .vis-caption {{
                font-size: 12px;
                color: #666;
                margin-top: 4px;
            }}
            .badge {{
                display: inline-block;
                padding: 3px 7px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                margin-right: 5px;
            }}
            .badge-correct {{
                background-color: #d4edda;
                color: #155724;
            }}
            .badge-incorrect {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            .badge-NRG {{
                background-color: #e2e3e5;
                color: #383d41;
            }}
            .badge-RG {{
                background-color: #cce5ff;
                color: #004085;
            }}
            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.8);
                overflow: auto;
            }}
            .modal-content {{
                display: block;
                margin: 60px auto;
                max-width: 90%;
                max-height: 80vh;
            }}
            .close {{
                position: absolute;
                top: 20px;
                right: 30px;
                color: white;
                font-size: 35px;
                font-weight: bold;
                cursor: pointer;
            }}
            .tabs {{
                display: flex;
                margin-top: 10px;
            }}
            .tab-btn {{
                padding: 8px 15px;
                cursor: pointer;
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-radius: 4px 4px 0 0;
                border-bottom: none;
            }}
            .tab-btn.active {{
                background-color: white;
                font-weight: bold;
            }}
            .tab-content {{
                display: none;
                padding: 10px;
                border: 1px solid #ddd;
                border-top: none;
            }}
            .tab-content.active {{
                display: block;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <p>{description}</p>
            
            <div class="filters">
                <h3>Filters</h3>
                <div>
                    <label for="class-filter">Class:</label>
                    <select id="class-filter">
                        <option value="all">All</option>
                        <option value="NRG">Non-Referable Glaucoma (NRG)</option>
                        <option value="RG">Referable Glaucoma (RG)</option>
                    </select>
                    
                    <label for="result-filter" style="margin-left: 15px;">Prediction:</label>
                    <select id="result-filter">
                        <option value="all">All</option>
                        <option value="correct">Correct Predictions</option>
                        <option value="incorrect">Incorrect Predictions</option>
                    </select>
                </div>
            </div>
            
            <div class="visualizations" id="vis-container">
    """
    
    # Add visualization cards
    for i, vis in enumerate(visualizations):
        # Extract paths and convert them to relative paths for HTML
        original_path = Path(vis['original_path'])
        heatmap_path = Path(vis['heatmap_path'])
        overlay_path = Path(vis['overlay_path'])
        
        # Make paths relative to the HTML file location
        try:
            rel_original = os.path.relpath(original_path, output_path.parent)
            rel_heatmap = os.path.relpath(heatmap_path, output_path.parent)
            rel_overlay = os.path.relpath(overlay_path, output_path.parent)
        except ValueError:
            # Fall back to absolute paths if relative paths can't be computed
            rel_original = str(original_path)
            rel_heatmap = str(heatmap_path)
            rel_overlay = str(overlay_path)
        
        # Prepare badges
        prediction_class = vis.get('prediction', 'Unknown')
        actual_class = vis.get('actual', '')
        confidence = vis.get('confidence', 0) * 100
        
        # Determine if prediction is correct (if actual class is available)
        is_correct = prediction_class == actual_class if actual_class else None
        result_class = ""
        
        if is_correct is not None:
            result_badge = f'<span class="badge badge-{"correct" if is_correct else "incorrect"}">{"Correct" if is_correct else "Incorrect"}</span>'
            result_class = f'result-{"correct" if is_correct else "incorrect"}'
        else:
            result_badge = ""
            
        prediction_badge = f'<span class="badge badge-{prediction_class}">Predicted: {prediction_class} ({confidence:.1f}%)</span>'
        
        if actual_class:
            actual_badge = f'<span class="badge badge-{actual_class}">Actual: {actual_class}</span>'
        else:
            actual_badge = ""
        
        # Add card to HTML
        html_content += f"""
                <div class="vis-card class-{prediction_class} {result_class}">
                    <div class="card-header">
                        Image {vis['image_id']}
                    </div>
                    <div class="card-content">
                        <div>
                            {prediction_badge}
                            {actual_badge}
                            {result_badge}
                        </div>
                        <div class="tabs">
                            <div class="tab-btn active" onclick="switchTab(this, 'tab-overlay-{i}')">Overlay</div>
                            <div class="tab-btn" onclick="switchTab(this, 'tab-original-{i}')">Original</div>
                            <div class="tab-btn" onclick="switchTab(this, 'tab-heatmap-{i}')">Heatmap</div>
                        </div>
                        <div class="tab-content active" id="tab-overlay-{i}">
                            <div class="vis-image">
                                <img src="{rel_overlay}" alt="Overlay visualization" onclick="openModal(this.src)">
                                <div class="vis-caption">Overlay showing regions influencing the model's decision</div>
                            </div>
                        </div>
                        <div class="tab-content" id="tab-original-{i}">
                            <div class="vis-image">
                                <img src="{rel_original}" alt="Original image" onclick="openModal(this.src)">
                                <div class="vis-caption">Original fundus image</div>
                            </div>
                        </div>
                        <div class="tab-content" id="tab-heatmap-{i}">
                            <div class="vis-image">
                                <img src="{rel_heatmap}" alt="GradCAM heatmap" onclick="openModal(this.src)">
                                <div class="vis-caption">GradCAM heatmap (red = high activation)</div>
                            </div>
                        </div>
                    </div>
                </div>
        """
    
    # Close HTML tags and add JavaScript for filtering and modal
    html_content += """
            </div>
        </div>
        
        <!-- Image modal -->
        <div id="imageModal" class="modal">
            <span class="close" onclick="closeModal()">&times;</span>
            <img class="modal-content" id="modalImage">
        </div>
        
        <script>
            // Filtering logic
            const classFilter = document.getElementById('class-filter');
            const resultFilter = document.getElementById('result-filter');
            const visContainer = document.getElementById('vis-container');
            
            function applyFilters() {
                const classValue = classFilter.value;
                const resultValue = resultFilter.value;
                
                const cards = visContainer.querySelectorAll('.vis-card');
                
                cards.forEach(card => {
                    let showCard = true;
                    
                    // Apply class filter
                    if (classValue !== 'all' && !card.classList.contains(`class-${classValue}`)) {
                        showCard = false;
                    }
                    
                    // Apply result filter
                    if (resultValue !== 'all' && !card.classList.contains(`result-${resultValue}`)) {
                        showCard = false;
                    }
                    
                    card.style.display = showCard ? 'block' : 'none';
                });
            }
            
            classFilter.addEventListener('change', applyFilters);
            resultFilter.addEventListener('change', applyFilters);
            
            // Modal for image zoom
            function openModal(src) {
                const modal = document.getElementById('imageModal');
                const modalImg = document.getElementById('modalImage');
                modal.style.display = "block";
                modalImg.src = src;
            }
            
            function closeModal() {
                document.getElementById('imageModal').style.display = "none";
            }
            
            // Tab switching logic
            function switchTab(tabBtn, tabId) {
                // Get parent element (card content)
                const parentEl = tabBtn.closest('.card-content');
                
                // Remove active class from all tab buttons and contents in this card
                parentEl.querySelectorAll('.tab-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                parentEl.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                
                // Add active class to clicked button and corresponding content
                tabBtn.classList.add('active');
                document.getElementById(tabId).classList.add('active');
            }
        </script>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
        
    logger.info(f"HTML report generated at: {output_path}")
