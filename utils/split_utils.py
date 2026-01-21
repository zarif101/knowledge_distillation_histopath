"""
Utilities for managing train/val/test splits.

Supports:
- Default random splits
- User-provided split files
- Reproducible splitting with seeds
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import os


def get_samples_from_paths(data_path: Union[str, Path]) -> List[str]:
    """
    Get list of sample IDs from a data directory.
    
    Assumes each subdirectory or .h5ad file is a sample.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        List of sample IDs
    """
    data_path = Path(data_path)
    samples = []
    
    for item in data_path.iterdir():
        if item.is_dir():
            samples.append(item.name)
        elif item.suffix in ['.h5ad', '.h5']:
            samples.append(item.stem)
    
    return sorted(samples)


def load_split_file(split_path: Union[str, Path]) -> List[str]:
    """
    Load sample IDs from a split file.
    
    Supports:
    - JSON file with list of sample IDs
    - Text file with one sample ID per line
    
    Args:
        split_path: Path to split file
        
    Returns:
        List of sample IDs
    """
    split_path = Path(split_path)
    
    if split_path.suffix == '.json':
        with open(split_path, 'r') as f:
            data = json.load(f)
            # Handle both list format and dict format
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Assume it has a 'samples' key or return values
                return data.get('samples', list(data.values())[0])
    else:
        # Assume text file
        with open(split_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]


def save_split_file(
    samples: List[str],
    split_path: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    Save sample IDs to a split file.
    
    Args:
        samples: List of sample IDs
        split_path: Path to save split file
        format: 'json' or 'txt'
    """
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(split_path, 'w') as f:
            json.dump(samples, f, indent=2)
    else:
        with open(split_path, 'w') as f:
            f.write('\n'.join(samples))
    
    print(f"Saved {len(samples)} samples to {split_path}")


def get_train_val_split(
    samples: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
    train_split_path: Optional[Union[str, Path]] = None,
    val_split_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Get train/val split, either from files or by random splitting.
    
    Args:
        samples: List of all sample IDs (used if split files not provided)
        val_ratio: Fraction for validation (used if split files not provided)
        seed: Random seed for reproducibility
        train_split_path: Optional path to predefined train split file
        val_split_path: Optional path to predefined val split file
        
    Returns:
        Tuple of (train_samples, val_samples)
    """
    # If both split files provided, load them
    if train_split_path is not None and val_split_path is not None:
        train_samples = load_split_file(train_split_path)
        val_samples = load_split_file(val_split_path)
        print(f"Loaded splits from files: {len(train_samples)} train, {len(val_samples)} val")
        return train_samples, val_samples
    
    # If only one provided, raise error
    if train_split_path is not None or val_split_path is not None:
        raise ValueError("Must provide both train_split_path and val_split_path, or neither")
    
    # Random split
    np.random.seed(seed)
    samples = list(samples)
    np.random.shuffle(samples)
    
    n_val = int(len(samples) * val_ratio)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    
    print(f"Random split (seed={seed}): {len(train_samples)} train, {len(val_samples)} val")
    return train_samples, val_samples


def get_train_val_test_split(
    samples: List[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    train_split_path: Optional[Union[str, Path]] = None,
    val_split_path: Optional[Union[str, Path]] = None,
    test_split_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Get train/val/test split.
    
    Args:
        samples: List of all sample IDs
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed
        train_split_path: Optional path to predefined train split
        val_split_path: Optional path to predefined val split
        test_split_path: Optional path to predefined test split
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    # If all split files provided, load them
    if all(p is not None for p in [train_split_path, val_split_path, test_split_path]):
        train_samples = load_split_file(train_split_path)
        val_samples = load_split_file(val_split_path)
        test_samples = load_split_file(test_split_path)
        print(f"Loaded splits: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
        return train_samples, val_samples, test_samples
    
    # Random split
    np.random.seed(seed)
    samples = list(samples)
    np.random.shuffle(samples)
    
    n_test = int(len(samples) * test_ratio)
    n_val = int(len(samples) * val_ratio)
    
    test_samples = samples[:n_test]
    val_samples = samples[n_test:n_test + n_val]
    train_samples = samples[n_test + n_val:]
    
    print(f"Random split (seed={seed}): {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
    return train_samples, val_samples, test_samples


def create_split_files(
    data_path: Union[str, Path],
    output_dir: Union[str, Path],
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42
) -> Dict[str, Path]:
    """
    Create and save split files for a dataset.
    
    Args:
        data_path: Path to data directory
        output_dir: Directory to save split files
        val_ratio: Fraction for validation
        test_ratio: Fraction for test (0 for no test set)
        seed: Random seed
        
    Returns:
        Dictionary with paths to created split files
    """
    samples = get_samples_from_paths(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if test_ratio > 0:
        train, val, test = get_train_val_test_split(
            samples, val_ratio, test_ratio, seed
        )
        paths = {
            'train': output_dir / 'train_split.json',
            'val': output_dir / 'val_split.json',
            'test': output_dir / 'test_split.json'
        }
        save_split_file(train, paths['train'])
        save_split_file(val, paths['val'])
        save_split_file(test, paths['test'])
    else:
        train, val = get_train_val_split(samples, val_ratio, seed)
        paths = {
            'train': output_dir / 'train_split.json',
            'val': output_dir / 'val_split.json'
        }
        save_split_file(train, paths['train'])
        save_split_file(val, paths['val'])
    
    return paths

