"""
Dataset adapter system for custom datasets.

This module provides a plugin architecture for users to integrate their OWN datasets
into the framework - with ANY data format they want!

The built-in pipeline scripts (finetune.py, distill.py, evaluate.py) work with HEST format.
For custom datasets, implement a DatasetAdapter and write your own training script,
or modify the pipeline scripts to use your adapter.

See examples/custom_dataset_adapter.py for a complete template.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
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
    
    Implement this class to add support for YOUR dataset with YOUR data format.
    There are no restrictions on how your data is organized - just implement
    the required methods to load and process your data.
    
    Required methods:
        - task_type: Return 'regression' or 'classification'
        - load_data: Load your data and return DataLoaders
        - get_num_outputs: Return number of output features/classes
        - get_loss_function: Return appropriate loss function
    
    See examples/custom_dataset_adapter.py for complete examples.
    """
    
    @property
    @abstractmethod
    def task_type(self) -> str:
        """
        Return the task type: 'regression' or 'classification'
        
        This determines how the model head is built and how metrics are computed.
        """
        pass
    
    @abstractmethod
    def load_data(
        self,
        train_samples: List[str],
        val_samples: List[str],
        transforms: Any,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Load your training and validation data.
        
        This is where you implement YOUR data loading logic for YOUR format.
        
        Args:
            train_samples: List of training sample identifiers (can be anything)
            val_samples: List of validation sample identifiers
            transforms: Image transforms to apply
            **kwargs: Any additional arguments your dataset needs
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        pass
    
    @abstractmethod
    def get_num_outputs(self, **kwargs) -> int:
        """
        Get number of output features/classes.
        
        For regression: number of target features (e.g., number of genes)
        For classification: number of classes
        """
        pass
    
    @abstractmethod
    def get_loss_function(self) -> torch.nn.Module:
        """
        Return appropriate loss function for your task.
        
        Common choices:
        - Regression: torch.nn.MSELoss(), torch.nn.L1Loss()
        - Classification: torch.nn.CrossEntropyLoss()
        """
        pass
    
    def get_default_hyperparams(self) -> Dict[str, Any]:
        """
        Return default hyperparameters for your dataset.
        
        Override this to provide sensible defaults for your data.
        """
        return {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'epochs': 50,
            'weight_decay': 1e-5
        }


# ============================================================================
# Built-in HEST Adapter (for reference)
# ============================================================================

@register_dataset("hest")
class HESTAdapter(DatasetAdapter):
    """
    Adapter for HEST spatial transcriptomics data.
    
    This is the built-in adapter for HEST format. The pipeline scripts
    (finetune.py, distill.py, evaluate.py) use this format directly.
    
    HEST data format:
        patches_path/
            SAMPLE1.h5    # H5 file with 'img' and 'barcode' arrays
            SAMPLE2.h5
            ...
        adata_path/
            SAMPLE1.h5ad  # AnnData with gene expression
            SAMPLE2.h5ad
            ...
    
    If your data is NOT in this format, create your own adapter!
    See examples/custom_dataset_adapter.py
    """
    
    def __init__(self):
        self._num_genes = None
    
    @property
    def task_type(self) -> str:
        return "regression"
    
    def load_data(
        self,
        train_samples: List[str],
        val_samples: List[str],
        transforms: Any,
        patches_path: str = None,
        adata_path: str = None,
        gene_list: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """Load HEST data."""
        from .data_utils import STPatchDatasetHEST
        
        if patches_path is None or adata_path is None:
            raise ValueError("HEST requires patches_path and adata_path")
        if gene_list is None:
            raise ValueError("HEST requires gene_list")
        
        train_dataset = STPatchDatasetHEST(
            patches_path=patches_path,
            adata_path=adata_path,
            samples=train_samples,
            gene_list=gene_list,
            transforms=transforms
        )
        
        val_dataset = STPatchDatasetHEST(
            patches_path=patches_path,
            adata_path=adata_path,
            samples=val_samples,
            gene_list=gene_list,
            transforms=transforms
        )
        
        self._num_genes = len(gene_list) if isinstance(gene_list, list) else None
        
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
