import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AddNoise(object):
    """
    Adds random noise to a PIL Image.

    Args:
        noise_level (float): Maximum intensity of the noise to be added.
                             Noise will be uniformly sampled from [0, noise_level).
    """
    def __init__(self, noise_level: float):
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("Noise level must be between 0.0 and 1.0")
        self.noise_level = noise_level
        logger.debug(f"Initialized AddNoise with noise_level={noise_level}")

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL Image): Image to add noise to.

        Returns:
            PIL Image: Image with added noise.
        """
        if self.noise_level == 0:
            return img
            
        try:
            # Convert PIL image to tensor
            img_tensor = F.to_tensor(img)
            # Generate noise tensor of the same shape
            noise = torch.rand_like(img_tensor) * self.noise_level
            # Add noise and clamp values to [0, 1]
            noisy_img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)
            # Convert back to PIL image
            noisy_img = F.to_pil_image(noisy_img_tensor)
            return noisy_img
        except Exception as e:
            logger.error(f"Error applying AddNoise: {e}", exc_info=True)
            # Return original image in case of error
            return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(noise_level={self.noise_level})"


def get_transforms(config: Dict[str, Any]) -> Dict[str, T.Compose]:
    """
    Creates a dictionary of torchvision transforms for train, validation, and test sets
    based on the provided configuration.

    Args:
        config (Dict[str, Any]): The merged configuration dictionary.

    Returns:
        Dict[str, T.Compose]: A dictionary containing 'train', 'validation', and 'test' transforms.
    """
    try:
        img_size = config.get('image_size', 512)
        mean = config.get('imagenet_mean', [0.485, 0.456, 0.406])
        std = config.get('imagenet_std', [0.229, 0.224, 0.225])

        # Training transforms parameters from config
        noise_level = config.get('add_noise_level', 0.01)
        cj_brightness = config.get('color_jitter_brightness', 0.01)
        cj_contrast = config.get('color_jitter_contrast', 0.01)
        cj_saturation = config.get('color_jitter_saturation', 0.01)
        cj_hue = config.get('color_jitter_hue', 0.01)
        # Ensure hue is within [-0.5, 0.5] if specified as a single value
        # The reference notebook uses (-0.01, 0.01), so we adapt
        hue_param = (-cj_hue, cj_hue) if cj_hue > 0 else 0 
        
        # Brightness/Contrast/Saturation are factors relative to 1.0
        brightness_param = (1.0 - cj_brightness, 1.0 + cj_brightness)
        contrast_param = (1.0 - cj_contrast, 1.0 + cj_contrast)
        saturation_param = (1.0 - cj_saturation, 1.0 + cj_saturation)

        logger.info(f"Creating transforms with size={img_size}, noise={noise_level}, jitter(b={brightness_param}, c={contrast_param}, s={saturation_param}, h={hue_param})")

        transforms = {
            'train': T.Compose([
                T.Resize(size=(img_size, img_size)), # Resize takes (h, w) tuple or single int
                AddNoise(noise_level=noise_level),
                T.ColorJitter(brightness=brightness_param, contrast=contrast_param, saturation=saturation_param, hue=hue_param),
                # RandomRotation and RandomAffine were 0 degrees in notebook, effectively no-op, so omitted unless configured otherwise
                # T.RandomRotation(degrees=config.get('random_rotation_degrees', 0)),
                # T.RandomAffine(degrees=config.get('random_affine_degrees', 0)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]),
            'validation': T.Compose([
                T.Resize(size=(img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]),
            'test': T.Compose([
                T.Resize(size=(img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        }
        logger.debug("Transforms created successfully.")
        return transforms

    except KeyError as e:
        logger.error(f"Missing key in configuration for transforms: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating transforms: {e}", exc_info=True)
        raise

# Example usage (optional)
# if __name__ == '__main__':
#     # Assumes config.yaml and logger_config.yaml exist and are valid
#     from utils.logger import setup_logging
#     from config_loader import load_config
#     setup_logging()
#     cfg = load_config()
#     if cfg:
#         all_transforms = get_transforms(cfg)
#         print("Train Transforms:")
#         print(all_transforms['train'])
#         print("\nValidation Transforms:")
#         print(all_transforms['validation'])
#         print("\nTest Transforms:")
#         print(all_transforms['test'])
#
#         # Test AddNoise
#         try:
#             # Create a dummy black image
#             dummy_img = Image.new('RGB', (60, 30), color = 'black')
#             noise_transform = AddNoise(noise_level=0.1)
#             noisy_img = noise_transform(dummy_img)
#             print("\nApplied AddNoise to dummy image.")
#             # noisy_img.show() # Uncomment to display
#         except Exception as e:
#             print(f"\nError testing AddNoise: {e}")
#     else:
#         print("Could not load config to test transforms.")
