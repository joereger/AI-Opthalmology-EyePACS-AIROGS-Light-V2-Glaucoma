import torch.nn as nn
import datetime

class BaseModel(nn.Module):
    """Base model class with metadata support for all project models."""
    
    def get_metadata(self, training_params=None):
        """
        Get model metadata as a dictionary.
        
        Args:
            training_params (dict, optional): Training-specific parameters to include
                
        Returns:
            dict: Model metadata
        """
        metadata = {
            "architecture": "Unknown",
            "variant": "Unknown",
            "num_classes": None,
            "input_size": None,
            "normalization": None,
            "created_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add training params if provided
        if training_params:
            metadata.update(training_params)
            
        return metadata
    
    def get_gradcam(self, image, target_layer_name=None, target_class=None):
        """
        Generate Grad-CAM visualization for the specified image.
        
        This is a base implementation that should be overridden by specific model classes.
        
        Args:
            image: Input tensor (already preprocessed)
            target_layer_name: Name or index of the layer to visualize
            target_class: Class index to visualize (default: predicted class)
        
        Returns:
            numpy array: Grad-CAM heatmap normalized to [0,1]
        """
        raise NotImplementedError("Subclasses must implement get_gradcam")
