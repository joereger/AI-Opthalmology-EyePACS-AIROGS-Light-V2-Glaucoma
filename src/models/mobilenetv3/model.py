import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights
import numpy as np
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class MobileNetV3Model(BaseModel):
    """
    MobileNetV3 Large model implementation that inherits from BaseModel
    and provides metadata about its architecture.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize a MobileNetV3 Large model with specified number of output classes.
        
        Args:
            num_classes (int): The number of output classes for the final layer. Defaults to 2.
            pretrained (bool): Whether to load pre-trained weights (ImageNet). Defaults to True.
            
        Raises:
            ValueError: If the loaded model doesn't have the expected classifier structure.
        """
        super().__init__()
        
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model_name = "MobileNetV3 Large"
        
        if pretrained:
            logger.info(f"Loading pre-trained {model_name} model with default weights.")
        else:
            logger.info(f"Loading {model_name} model without pre-trained weights.")
            
        # Initialize the torchvision model
        base_model = models.mobilenet_v3_large(weights=weights)
        
        # Copy all attributes from the base model to this instance 
        for attr_name in dir(base_model):
            # Skip private attributes and methods
            if attr_name.startswith('_'):
                continue
                
            # Get the attribute
            attr = getattr(base_model, attr_name)
            
            # Skip methods, only copy properties and modules
            if callable(attr) and not isinstance(attr, nn.Module):
                continue
                
            setattr(self, attr_name, attr)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch of images)
            
        Returns:
            Model output tensor
        """
        # Forward through the features
        x = self.features(x)
        
        # Apply global average pooling
        x = self.avgpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Forward through classifier
        x = self.classifier(x)
        
        return x
    
    def get_metadata(self, training_params=None):
        """
        Get model metadata including architecture details.
        
        Args:
            training_params (dict, optional): Training-specific parameters to include
                
        Returns:
            dict: Model metadata
        """
        metadata = super().get_metadata(training_params)
        
        # Get number of classes from the model
        num_classes = None
        if hasattr(self, 'classifier') and isinstance(self.classifier, nn.Sequential):
            last_layer = self.classifier[-1]
            if isinstance(last_layer, nn.Linear):
                num_classes = last_layer.out_features
        
        metadata.update({
            "architecture": "MobileNetV3",
            "variant": "Large",
            "num_classes": num_classes,
            "input_size": [3, 512, 512],  # Standard input size expected by the model
            "normalization": {
                "mean": [0.485, 0.456, 0.406],  # ImageNet means
                "std": [0.229, 0.224, 0.225]    # ImageNet stds
            }
        })
        
        return metadata
    
    def get_gradcam(self, image, target_layer_name=None, target_class=None):
        """
        Generate Grad-CAM visualization for the specified image.
        
        Args:
            image: Input tensor (already preprocessed) with shape [1, C, H, W]
            target_layer_name: Name or index of the layer to visualize (default: last conv layer)
            target_class: Class index to visualize (default: predicted class)
        
        Returns:
            numpy array: Grad-CAM heatmap normalized to [0,1] with shape [H, W]
        """
        # Ensure model is in eval mode
        self.eval()
        
        # Convert string layer name to actual module if needed
        target_layer = None
        if target_layer_name is None:
            # Default to last convolutional layer in the features
            target_layer = self.features[-1]
        elif isinstance(target_layer_name, str):
            if target_layer_name.startswith("features[") and target_layer_name.endswith("]"):
                # Extract index from string like "features[12]"
                try:
                    idx = int(target_layer_name[9:-1])
                    target_layer = self.features[idx]
                except (IndexError, ValueError) as e:
                    logger.error(f"Invalid layer index in {target_layer_name}: {e}")
                    raise ValueError(f"Invalid layer name: {target_layer_name}")
            elif target_layer_name == "features[-1]":
                target_layer = self.features[-1]
            else:
                raise ValueError(f"Unsupported layer name format: {target_layer_name}")
        elif isinstance(target_layer_name, int):
            # If target_layer_name is an integer index
            try:
                target_layer = self.features[target_layer_name]
            except IndexError:
                raise ValueError(f"Invalid layer index: {target_layer_name}")
        
        if target_layer is None:
            raise ValueError("Could not resolve target layer")
        
        # Register hooks to get activations and gradients
        activations = None
        gradients = None
        
        def save_activation(module, input, output):
            nonlocal activations
            activations = output.detach()
        
        def save_gradient(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()
        
        # Register hooks
        handle1 = target_layer.register_forward_hook(save_activation)
        handle2 = target_layer.register_full_backward_hook(save_gradient)
        
        try:
            # Ensure image has batch dimension
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            # Forward pass
            with torch.set_grad_enabled(True):
                # Get model output (ensure we create graph)
                output = self(image)
                
                # If target class not specified, use predicted class
                if target_class is None:
                    target_class = output.argmax(dim=1).item()
                
                # Zero all gradients
                self.zero_grad()
                
                # Create one-hot encoding for backprop
                one_hot = torch.zeros_like(output)
                one_hot[0, target_class] = 1
                
                # Backward pass
                output.backward(gradient=one_hot)
            
            # Calculate Grad-CAM
            # Global average pooling the gradients
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
            
            # Compute weighted activations and apply ReLU
            cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]
            cam = F.relu(cam)  # Only keep positive attributions
            
            # Normalize heatmap to [0, 1]
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:  # Avoid division by zero
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            
            # Convert to numpy
            cam = cam.squeeze().cpu().numpy()  # [H, W]
            
            return cam
            
        finally:
            # Clean up
            handle1.remove()
            handle2.remove()

def get_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Factory function that returns a MobileNetV3Model instance.
    Maintained for backward compatibility.

    Args:
        num_classes (int): The number of output classes for the final layer. Defaults to 2.
        pretrained (bool): Whether to load pre-trained weights (ImageNet). Defaults to True.

    Returns:
        MobileNetV3Model: The initialized model.
    """
    return MobileNetV3Model(num_classes=num_classes, pretrained=pretrained)

# Example usage (optional)
# if __name__ == '__main__':
#     from utils.logger import setup_logging
#     setup_logging() # Setup logging first
#
#     # Get model for 2 classes (default)
#     model_glaucoma = get_model(num_classes=2)
#     print("\nModel Summary (Glaucoma - 2 classes):")
#     # print(model_glaucoma) # Print full model structure (can be long)
#     print(f"Final classifier layer: {model_glaucoma.classifier[-1]}")
#
#     # Get model for 10 classes, not pretrained
#     model_custom = get_model(num_classes=10, pretrained=False)
#     print("\nModel Summary (Custom - 10 classes, no pretraining):")
#     print(f"Final classifier layer: {model_custom.classifier[-1]}")
#
#     # Test error handling (optional - requires modifying the function temporarily or mocking)
#     # try:
#     #     # Example: Simulate unexpected structure
#     #     dummy_model = models.mobilenet_v3_large(weights=None)
#     #     dummy_model.classifier = nn.Linear(10, 5) # Replace sequential with single layer
#     #     # This would now raise ValueError in get_model if passed
#     # except Exception as e:
#     #      print(f"\nCaught expected error: {e}")
