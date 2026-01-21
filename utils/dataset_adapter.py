"""
Dataset adapter system for extensible dataset support.

Provides a plugin architecture for users to integrate custom datasets
without modifying core framework code.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader


# Registry for dataset adapters
_DATASET_REGISTRY: Dict[str, type] = {}


def register_dataset(name: str):
    """
    Decorator to register a dataset adapter.
    
    Usage:
        @register_dataset("my_dataset")
        class MyDatasetAdapter(DatasetAdapter):
            ...
    """
    def decorator(cls):
        _DATASET_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_dataset_adapter(name: str) -> "DatasetAdapter":
    """
    Get a dataset adapter by name.
    
    Args:
        name: Registered name of the dataset adapter
        
    Returns:
        Instance of the dataset adapter
    """
    name = name.lower()
    if name not in _DATASET_REGISTRY:
        available = list(_DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return _DATASET_REGISTRY[name]()


def list_available_datasets() -> List[str]:
    """List all registered dataset adapters."""
    return list(_DATASET_REGISTRY.keys())


class DatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.
    
    Implement this class to add support for a new dataset type.
    """
    
    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return task type: 'regression' or 'classification'"""
        pass
    
    @abstractmethod
    def load_data(
        self,
        data_path: str,
        train_samples: List[str],
        val_samples: List[str],
        transforms: Any,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Load training and validation data.
        
        Args:
            data_path: Path to the dataset
            train_samples: List of training sample IDs
            val_samples: List of validation sample IDs
            transforms: Image transforms to apply
            **kwargs: Additional dataset-specific arguments
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        pass
    
    @abstractmethod
    def get_num_outputs(self, **kwargs) -> int:
        """
        Get number of output features/classes.
        
        For regression: number of target features
        For classification: number of classes
        """
        pass
    
    @abstractmethod
    def get_loss_function(self) -> torch.nn.Module:
        """Return appropriate loss function for this dataset."""
        pass
    
    def get_default_hyperparams(self) -> Dict[str, Any]:
        """Return default hyperparameters for this dataset."""
        return {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'epochs': 50,
            'weight_decay': 1e-5
        }


# ============================================================================
# Built-in Dataset Adapters
# ============================================================================

@register_dataset("hest")
class HESTAdapter(DatasetAdapter):
    """Adapter for HEST spatial transcriptomics data."""
    
    def __init__(self):
        self._num_genes = None
    
    @property
    def task_type(self) -> str:
        return "regression"
    
    def load_data(
        self,
        data_path: str,
        train_samples: List[str],
        val_samples: List[str],
        transforms: Any,
        gene_list: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """Load HEST data with gene list filtering."""
        from .data_utils import STPatchDatasetHEST
        
        # Create datasets
        train_dataset = STPatchDatasetHEST(
            data_path=data_path,
            sample_ids=train_samples,
            gene_list=gene_list,
            transform=transforms
        )
        
        val_dataset = STPatchDatasetHEST(
            data_path=data_path,
            sample_ids=val_samples,
            gene_list=gene_list,
            transform=transforms
        )
        
        # Store num genes
        self._num_genes = len(gene_list) if gene_list else train_dataset.num_genes
        
        # Create loaders
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
    
    def get_num_outputs(self, gene_list: Optional[List[str]] = None, **kwargs) -> int:
        if gene_list is not None:
            return len(gene_list)
        if self._num_genes is not None:
            return self._num_genes
        raise ValueError("Must call load_data first or provide gene_list")
    
    def get_loss_function(self) -> torch.nn.Module:
        return torch.nn.MSELoss()
    
    def get_default_hyperparams(self) -> Dict[str, Any]:
        return {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'epochs': 100,
            'weight_decay': 1e-5
        }


@register_dataset("wsiclass")
class WSICLASSAdapter(DatasetAdapter):
    """Adapter for WSI Classification data."""
    
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
        metadata_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """Load WSI classification data."""
        from .data_utils import WSIClassificationDataset
        
        # Create datasets
        train_dataset = WSIClassificationDataset(
            patches_path=data_path,
            metadata_path=metadata_path,
            sample_ids=train_samples,
            transform=transforms
        )
        
        val_dataset = WSIClassificationDataset(
            patches_path=data_path,
            metadata_path=metadata_path,
            sample_ids=val_samples,
            transform=transforms
        )
        
        # Store num classes
        self._num_classes = train_dataset.num_classes
        
        # Create loaders
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
        if self._num_classes is not None:
            return self._num_classes
        raise ValueError("Must call load_data first")
    
    def get_loss_function(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()
    
    def get_default_hyperparams(self) -> Dict[str, Any]:
        return {
            'learning_rate': 1e-4,
            'batch_size': 64,
            'epochs': 50,
            'weight_decay': 1e-5
        }

