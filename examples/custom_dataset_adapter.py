"""
Example: Creating a custom dataset adapter.

This template shows how to integrate your own dataset into the framework.
Copy this file and modify it for your specific data format.

Usage:
    1. Copy this file to your project
    2. Modify CustomAdapter class for your data
    3. Import and register before running pipeline scripts
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Any, Dict, Optional
from pathlib import Path

# Import the adapter system
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset_adapter import DatasetAdapter, register_dataset


# =============================================================================
# Example 1: Custom Regression Dataset Adapter
# =============================================================================

@register_dataset("my_regression_data")
class CustomRegressionAdapter(DatasetAdapter):
    """
    Example adapter for a custom regression dataset.
    
    Assumes data is organized as:
        data_path/
            sample1/
                images/
                    patch_001.png
                    patch_002.png
                targets.csv  (columns: patch_id, target1, target2, ...)
            sample2/
                ...
    """
    
    def __init__(self):
        self._num_outputs = None
    
    @property
    def task_type(self) -> str:
        return "regression"
    
    def load_data(
        self,
        data_path: str,
        train_samples: List[str],
        val_samples: List[str],
        transforms: Any,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """Load training and validation data."""
        
        # Create your custom dataset instances
        train_dataset = CustomRegressionDataset(
            data_path=data_path,
            sample_ids=train_samples,
            transform=transforms
        )
        
        val_dataset = CustomRegressionDataset(
            data_path=data_path,
            sample_ids=val_samples,
            transform=transforms
        )
        
        # Store number of outputs
        self._num_outputs = train_dataset.num_targets
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_num_outputs(self, **kwargs) -> int:
        if self._num_outputs is None:
            raise ValueError("Call load_data first")
        return self._num_outputs
    
    def get_loss_function(self) -> torch.nn.Module:
        return torch.nn.MSELoss()
    
    def get_default_hyperparams(self) -> Dict[str, Any]:
        return {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'epochs': 100,
            'weight_decay': 1e-5
        }


class CustomRegressionDataset(Dataset):
    """Custom PyTorch Dataset for the regression task."""
    
    def __init__(
        self,
        data_path: str,
        sample_ids: List[str],
        transform: Any = None
    ):
        self.data_path = Path(data_path)
        self.sample_ids = sample_ids
        self.transform = transform
        
        # Load all patches and targets
        self.patches = []
        self.targets = []
        
        for sample_id in sample_ids:
            sample_dir = self.data_path / sample_id
            
            # Load targets
            import pandas as pd
            targets_df = pd.read_csv(sample_dir / 'targets.csv')
            
            # Load each patch
            images_dir = sample_dir / 'images'
            for _, row in targets_df.iterrows():
                patch_path = images_dir / f"{row['patch_id']}.png"
                if patch_path.exists():
                    self.patches.append(str(patch_path))
                    # Extract target values (all columns except patch_id)
                    target_cols = [c for c in targets_df.columns if c != 'patch_id']
                    self.targets.append(row[target_cols].values.astype(float))
        
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        self.num_targets = self.targets.shape[1]
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        # Load image
        img = Image.open(self.patches[idx]).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, self.targets[idx]


# =============================================================================
# Example 2: Custom Classification Dataset Adapter
# =============================================================================

@register_dataset("my_classification_data")
class CustomClassificationAdapter(DatasetAdapter):
    """
    Example adapter for a custom classification dataset.
    
    Assumes data is organized as:
        data_path/
            class_0/
                image1.png
                image2.png
            class_1/
                image3.png
                image4.png
    """
    
    def __init__(self):
        self._num_classes = None
    
    @property
    def task_type(self) -> str:
        return "classification"
    
    def load_data(
        self,
        data_path: str,
        train_samples: List[str],
        val_samples: List[str],
        transforms: Any,
        batch_size: int = 64,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """Load training and validation data."""
        from torchvision.datasets import ImageFolder
        
        # For ImageFolder-style datasets, train_samples and val_samples
        # could be paths to train/val directories
        train_dataset = ImageFolder(
            root=f"{data_path}/train",
            transform=transforms
        )
        
        val_dataset = ImageFolder(
            root=f"{data_path}/val",
            transform=transforms
        )
        
        self._num_classes = len(train_dataset.classes)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_num_outputs(self, **kwargs) -> int:
        if self._num_classes is None:
            raise ValueError("Call load_data first")
        return self._num_classes
    
    def get_loss_function(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()


# =============================================================================
# How to use your custom adapter
# =============================================================================

if __name__ == "__main__":
    """
    To use your custom adapter:
    
    1. Import this file before running the pipeline:
        
        import examples.custom_dataset_adapter
        
    2. Then run with your dataset name:
        
        python pipeline/finetune.py --dataset my_regression_data \\
            --data_path /path/to/your/data \\
            --output_dir ./results
    
    3. Or use programmatically:
        
        from utils.dataset_adapter import get_dataset_adapter
        
        adapter = get_dataset_adapter("my_regression_data")
        train_loader, val_loader = adapter.load_data(
            data_path="/path/to/data",
            train_samples=["sample1", "sample2"],
            val_samples=["sample3"],
            transforms=your_transforms
        )
    """
    
    # List all registered datasets (including custom ones)
    from utils.dataset_adapter import list_available_datasets
    print("Available datasets:", list_available_datasets())

