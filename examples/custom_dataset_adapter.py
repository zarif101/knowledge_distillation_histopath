"""
Example: Creating a custom dataset adapter for YOUR data format.

The built-in pipeline scripts work with HEST format. But if you have your own
dataset with a DIFFERENT format, you can use this adapter system to integrate it!

YOUR DATA CAN BE IN ANY FORMAT - just implement the DatasetAdapter interface
to load and process your data however you need.

Usage:
    1. Copy this file to your project
    2. Modify the adapter class for YOUR specific data format
    3. Import your adapter before running training
    4. Use the adapter in your training script

Example workflow:
    # In your training script:
    from my_adapter import MyCustomAdapter
    
    adapter = MyCustomAdapter()
    train_loader, val_loader = adapter.load_data(
        train_samples=["sample1", "sample2"],
        val_samples=["sample3"],
        transforms=my_transforms,
        data_path="/path/to/my/data"  # YOUR paths
    )
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
# Example 1: Custom Regression Dataset (e.g., predicting continuous values)
# =============================================================================

@register_dataset("my_regression_data")
class CustomRegressionAdapter(DatasetAdapter):
    """
    Example adapter for a custom regression dataset.
    
    YOUR DATA FORMAT - Modify this to match how YOUR data is organized!
    
    This example assumes:
        data_path/
            sample1/
                images/
                    patch_001.png
                    patch_002.png
                targets.csv  (columns: patch_id, target1, target2, ...)
            sample2/
                ...
    
    But you can change this to ANY format you want!
    """
    
    def __init__(self):
        self._num_outputs = None
    
    @property
    def task_type(self) -> str:
        return "regression"
    
    def load_data(
        self,
        train_samples: List[str],
        val_samples: List[str],
        transforms: Any,
        data_path: str = None,  # YOUR custom argument
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Load YOUR data in YOUR format.
        
        Modify this method to load your specific data format!
        """
        if data_path is None:
            raise ValueError("Must provide data_path")
        
        # Create YOUR custom dataset instances
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
        
        self._num_outputs = train_dataset.num_targets
        
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
    """
    Custom PyTorch Dataset - modify this for YOUR data format!
    """
    
    def __init__(
        self,
        data_path: str,
        sample_ids: List[str],
        transform: Any = None
    ):
        self.data_path = Path(data_path)
        self.sample_ids = sample_ids
        self.transform = transform
        
        # Load YOUR data here - modify for YOUR format!
        self.patches = []
        self.targets = []
        
        for sample_id in sample_ids:
            sample_dir = self.data_path / sample_id
            
            # Example: Load targets from CSV
            import pandas as pd
            targets_df = pd.read_csv(sample_dir / 'targets.csv')
            
            # Example: Load each patch image
            images_dir = sample_dir / 'images'
            for _, row in targets_df.iterrows():
                patch_path = images_dir / f"{row['patch_id']}.png"
                if patch_path.exists():
                    self.patches.append(str(patch_path))
                    target_cols = [c for c in targets_df.columns if c != 'patch_id']
                    self.targets.append(row[target_cols].values.astype(float))
        
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        self.num_targets = self.targets.shape[1]
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img = Image.open(self.patches[idx]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.targets[idx]


# =============================================================================
# Example 2: Custom Classification Dataset
# =============================================================================

@register_dataset("my_classification_data")
class CustomClassificationAdapter(DatasetAdapter):
    """
    Example adapter for a custom classification dataset.
    
    This example uses torchvision's ImageFolder format:
        data_path/
            train/
                class_0/
                    image1.png
                class_1/
                    image2.png
            val/
                class_0/
                    image3.png
                ...
    
    But again, you can modify this for ANY format!
    """
    
    def __init__(self):
        self._num_classes = None
    
    @property
    def task_type(self) -> str:
        return "classification"
    
    def load_data(
        self,
        train_samples: List[str],  # Not used in ImageFolder, but required by interface
        val_samples: List[str],
        transforms: Any,
        data_path: str = None,
        batch_size: int = 64,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """Load classification data using ImageFolder format."""
        from torchvision.datasets import ImageFolder
        
        if data_path is None:
            raise ValueError("Must provide data_path")
        
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
    Example training script using a custom adapter.
    
    You can write a similar script for your own data!
    """
    
    print("=== Custom Dataset Adapter Example ===\n")
    
    # List all registered datasets
    from utils.dataset_adapter import list_available_datasets
    print("Available datasets:", list_available_datasets())
    
    print("\n--- Example Usage ---")
    print("""
    # 1. Import your adapter
    from examples.custom_dataset_adapter import CustomRegressionAdapter
    
    # 2. Create adapter instance
    adapter = CustomRegressionAdapter()
    
    # 3. Load your data (with YOUR paths and YOUR format)
    train_loader, val_loader = adapter.load_data(
        train_samples=["sample1", "sample2", "sample3"],
        val_samples=["sample4"],
        transforms=my_transforms,
        data_path="/path/to/my/data"  # YOUR data location
    )
    
    # 4. Get model configuration
    num_outputs = adapter.get_num_outputs()
    loss_fn = adapter.get_loss_function()
    
    # 5. Train your model!
    # ... (use the loaders in your training loop)
    """)
    
    print("\nThe key point: YOUR data can be in ANY format!")
    print("Just implement the DatasetAdapter interface to load it.")
