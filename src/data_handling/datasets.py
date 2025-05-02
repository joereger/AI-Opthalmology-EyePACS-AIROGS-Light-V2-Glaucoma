import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

# Import from sibling module
from .transforms import get_transforms

logger = logging.getLogger(__name__)

def get_datasets(
    config: Dict[str, Any],
    run_id: str,
    base_conformed_dir: Optional[str] = None
) -> Optional[Dict[str, Dataset]]:
    """
    Creates ImageFolder datasets for train, validation, and test splits.

    Args:
        config (Dict[str, Any]): The merged configuration dictionary.
        run_id (str): The specific run ID for which to load data.
        base_conformed_dir (Optional[str]): Overrides the conformed data directory from config.

    Returns:
        Optional[Dict[str, Dataset]]: A dictionary containing 'train', 'validation',
                                      and 'test' ImageFolder datasets, or None if an error occurs.
    """
    try:
        # Determine the base directory for conformed data
        if base_conformed_dir is None:
            base_conformed_dir = config.get('conformed_data_dir', 'data/02_conformed_to_imagefolder')
        
        run_data_dir = Path(base_conformed_dir) / run_id
        logger.info(f"Attempting to load datasets from: {run_data_dir}")

        if not run_data_dir.is_dir():
            logger.error(f"Conformed data directory for run '{run_id}' not found at: {run_data_dir}")
            return None

        # Get transforms based on config
        transforms = get_transforms(config)

        datasets = {}
        for split in ['train', 'validation', 'test']:
            split_dir = run_data_dir / split
            if not split_dir.is_dir():
                logger.error(f"Directory for split '{split}' not found at: {split_dir}")
                # Allow continuing if only one split is missing, but log error
                # Consider if this should be a fatal error depending on usage
                continue 
                # return None # Make it fatal if all splits are required

            logger.info(f"Loading {split} dataset from {split_dir}...")
            datasets[split] = ImageFolder(root=str(split_dir), transform=transforms[split])
            logger.info(f"Loaded {split} dataset with {len(datasets[split])} images.")
            # Log class names and indices
            logger.info(f"{split} classes: {datasets[split].class_to_idx}")


        if not datasets: # Check if any dataset was loaded
             logger.error(f"No datasets could be loaded from {run_data_dir}.")
             return None
             
        return datasets

    except Exception as e:
        logger.error(f"Failed to create datasets for run '{run_id}': {e}", exc_info=True)
        return None


def get_dataloaders(
    datasets: Dict[str, Dataset],
    config: Dict[str, Any]
) -> Optional[Dict[str, DataLoader]]:
    """
    Creates DataLoader instances for the provided datasets.

    Args:
        datasets (Dict[str, Dataset]): Dictionary of datasets ('train', 'validation', 'test').
        config (Dict[str, Any]): The merged configuration dictionary.

    Returns:
        Optional[Dict[str, DataLoader]]: Dictionary of DataLoaders, or None if input is invalid.
    """
    if not datasets:
        logger.error("Cannot create dataloaders: datasets dictionary is empty or None.")
        return None
        
    try:
        batch_size = config.get('batch_size', 4)
        num_workers = config.get('num_workers', 1)
        pin_memory = config.get('pin_memory', True)
        logger.info(f"Creating dataloaders with batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")

        dataloaders = {}
        for split, dataset in datasets.items():
            shuffle = (split == 'train') # Shuffle only training data
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            logger.debug(f"Created DataLoader for {split} split.")

        return dataloaders

    except KeyError as e:
         logger.error(f"Missing key in configuration for dataloaders: {e}")
         raise # Or return None
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}", exc_info=True)
        raise # Or return None


# Example usage (optional)
# if __name__ == '__main__':
#     # Assumes config.yaml, logger_config.yaml exist
#     # Assumes Phase 02 has been run for 'run_1' and created dummy data in:
#     # data/02_conformed_to_imagefolder/run_1/train/NRG/dummy.jpg
#     # data/02_conformed_to_imagefolder/run_1/train/RG/dummy.jpg
#     # data/02_conformed_to_imagefolder/run_1/validation/NRG/dummy.jpg
#     # ... etc. for test split
#
#     from utils.logger import setup_logging
#     from config_loader import load_config
#     from utils.file_utils import ensure_dir_exists
#     from PIL import Image
#
#     setup_logging()
#     cfg = load_config()
#
#     if cfg:
#         test_run_id = 'run_1'
#         base_dir = Path(cfg['conformed_data_dir']) / test_run_id
#
#         # Create dummy data structure for testing
#         print("Creating dummy data structure for testing...")
#         for split in ['train', 'validation', 'test']:
#             for class_label in ['NRG', 'RG']:
#                 dummy_dir = base_dir / split / class_label
#                 ensure_dir_exists(dummy_dir)
#                 dummy_img_path = dummy_dir / f'dummy_{split}_{class_label}.jpg'
#                 if not dummy_img_path.exists():
#                     try:
#                         Image.new('RGB', (60, 30), color = 'red').save(dummy_img_path)
#                         print(f"Created dummy image: {dummy_img_path}")
#                     except Exception as e:
#                         print(f"Error creating dummy image {dummy_img_path}: {e}")
#
#         # Test get_datasets
#         print(f"\nTesting get_datasets for run_id='{test_run_id}'...")
#         all_datasets = get_datasets(cfg, test_run_id)
#
#         if all_datasets:
#             print("Datasets loaded successfully:")
#             for split, ds in all_datasets.items():
#                 print(f"  {split}: {len(ds)} samples, Classes: {ds.classes}")
#
#             # Test get_dataloaders
#             print("\nTesting get_dataloaders...")
#             all_dataloaders = get_dataloaders(all_datasets, cfg)
#
#             if all_dataloaders:
#                 print("DataLoaders created successfully:")
#                 for split, dl in all_dataloaders.items():
#                     print(f"  {split}: Batch size {dl.batch_size}")
#                     # Optional: Iterate over one batch
#                     # try:
#                     #     inputs, labels = next(iter(dl))
#                     #     print(f"    First batch shape: {inputs.shape}, Labels: {labels}")
#                     # except StopIteration:
#                     #     print(f"    DataLoader for {split} is empty.")
#                     # except Exception as e:
#                     #     print(f"    Error iterating dataloader {split}: {e}")
#             else:
#                 print("Failed to create dataloaders.")
#         else:
#             print("Failed to load datasets.")
#
#         # Clean up dummy data (optional)
#         # import shutil
#         # print("\nCleaning up dummy data...")
#         # if base_dir.exists():
#         #     shutil.rmtree(base_dir)
#         #     print(f"Removed dummy data directory: {base_dir}")
#
#     else:
#         print("Could not load config to test datasets/dataloaders.")
