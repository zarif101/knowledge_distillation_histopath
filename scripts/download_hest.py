#!/usr/bin/env python
"""
Download HEST dataset from HuggingFace.

This script downloads HEST spatial transcriptomics data and organizes it
into the directory structure expected by StainFactor.

Usage:
    # Download lung data with paper filters (human only, no Xenium)
    python scripts/download_hest.py --tissue lung --output_dir ./hest_data --paper_filters
    
    # Download all paper tissues with paper filters
    python scripts/download_hest.py --tissue all --output_dir ./hest_data --paper_filters
    
    # Download without filters (all samples for a tissue)
    python scripts/download_hest.py --tissue lung --output_dir ./hest_data
    
    # List available tissues
    python scripts/download_hest.py --list_tissues

Prerequisites:
    1. pip install huggingface_hub pandas
    2. Create a HuggingFace account and get an access token
    3. Accept HEST terms of use: https://huggingface.co/datasets/MahmoodLab/hest
    4. Set your token below or use --token argument
"""

import argparse
import os
import shutil
from pathlib import Path

# =============================================================================
# CONFIGURATION - Set your HuggingFace token here
# =============================================================================
# Option 1: Set token directly (not recommended for shared code)
HF_TOKEN = None  # e.g., "hf_xxxxxxxxxxxxxxxxxxxx"

# Option 2: Set via environment variable (recommended)
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Option 3: Pass via command line: --token "hf_xxxxxxxxxxxxxxxxxxxx"
# =============================================================================

REPO_ID = "MahmoodLab/hest"
METADATA_URL = "hf://datasets/MahmoodLab/hest/HEST_v1_2_1.csv"

# Tissues used in the StainFactor paper
PAPER_TISSUES = ["Lung", "Breast", "Colon", "Prostate", "Skin"]

# Mapping from tissue name to directory name
TISSUE_TO_DIR = {
    "Lung": "lung",
    "Breast": "breast",
    "Colon": "colon",
    "Prostate": "prostate",
    "Skin": "skin",
    "Bowel": "bowel",
    "Brain": "brain",
    "Kidney": "kidney",
    "Liver": "liver",
    "Lymph Node": "lymph_node",
    "Pancreas": "pancreas",
    "Stomach": "stomach",
}


def get_token(args_token):
    """Get HuggingFace token from args, config, or environment."""
    if args_token:
        return args_token
    if HF_TOKEN:
        return HF_TOKEN
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    return None


def load_metadata():
    """Load HEST metadata from HuggingFace."""
    import pandas as pd
    print("Loading HEST metadata...")
    meta_df = pd.read_csv(METADATA_URL)
    print(f"  Found {len(meta_df)} total samples")
    return meta_df


def apply_paper_filters(meta_df):
    """
    Apply the filters used in the StainFactor paper:
    - Human samples only (Homo sapiens)
    - Exclude Xenium technology
    """
    print("Applying paper filters...")
    print(f"  Before: {len(meta_df)} samples")
    
    # Filter to human only
    meta_df = meta_df[meta_df['species'] == 'Homo sapiens']
    print(f"  After species filter (Homo sapiens): {len(meta_df)} samples")
    
    # Exclude Xenium
    meta_df = meta_df[meta_df['st_technology'] != 'Xenium']
    print(f"  After technology filter (no Xenium): {len(meta_df)} samples")
    
    return meta_df


def get_tissue_samples(meta_df, tissue):
    """Get sample IDs for a specific tissue."""
    # Handle tissue name variations
    tissue_lower = tissue.lower()
    
    # Map common variations
    tissue_map = {
        "colon": "Colon",
        "bowel": "Bowel", 
        "lung": "Lung",
        "breast": "Breast",
        "prostate": "Prostate",
        "skin": "Skin",
        "brain": "Brain",
        "kidney": "Kidney",
        "liver": "Liver",
        "lymph_node": "Lymph Node",
        "pancreas": "Pancreas",
        "stomach": "Stomach",
    }
    
    organ_name = tissue_map.get(tissue_lower, tissue)
    
    tissue_df = meta_df[meta_df['organ'] == organ_name]
    return tissue_df['id'].values, organ_name


def download_samples(sample_ids, output_dir, token):
    """Download samples from HuggingFace."""
    from huggingface_hub import snapshot_download, login
    
    if token:
        print("Logging into HuggingFace...")
        login(token=token)
    
    # Create patterns for the sample IDs
    patterns = []
    for sid in sample_ids:
        patterns.append(f"*{sid}[_.]**")
    
    print(f"Downloading {len(sample_ids)} samples...")
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=patterns,
        repo_type="dataset",
        local_dir=output_dir
    )
    print("Download complete!")


def organize_data(raw_dir, tissue_dir, sample_ids):
    """
    Organize downloaded data into patches/ and st/ directories.
    
    HEST downloads files in format:
    - {raw_dir}/patches/{sample_id}.h5
    - {raw_dir}/st/{sample_id}.h5ad
    
    We organize into:
    - {tissue_dir}/patches/{sample_id}.h5
    - {tissue_dir}/st/{sample_id}.h5ad
    """
    raw_dir = Path(raw_dir)
    tissue_dir = Path(tissue_dir)
    
    patches_src = raw_dir / "patches"
    st_src = raw_dir / "st"
    
    patches_dst = tissue_dir / "patches"
    st_dst = tissue_dir / "st"
    
    patches_dst.mkdir(parents=True, exist_ok=True)
    st_dst.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for sid in sample_ids:
        # Move patch file
        patch_file = patches_src / f"{sid}.h5"
        if patch_file.exists():
            shutil.move(str(patch_file), str(patches_dst / f"{sid}.h5"))
            moved_count += 1
        
        # Move st file
        st_file = st_src / f"{sid}.h5ad"
        if st_file.exists():
            shutil.move(str(st_file), str(st_dst / f"{sid}.h5ad"))
    
    print(f"  Organized {moved_count} samples into {tissue_dir}")


def list_tissues(meta_df):
    """List available tissues and sample counts."""
    print("\nAvailable tissues in HEST:")
    print("-" * 50)
    
    organ_counts = meta_df['organ'].value_counts()
    for organ, count in organ_counts.items():
        paper_marker = " [paper]" if organ in PAPER_TISSUES else ""
        print(f"  {organ}: {count} samples{paper_marker}")
    
    print("-" * 50)
    print(f"Total: {len(meta_df)} samples")
    print("\n[paper] = tissues used in StainFactor paper")


def main():
    parser = argparse.ArgumentParser(
        description="Download HEST dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download lung data with paper filters
    python scripts/download_hest.py --tissue lung --output_dir ./hest_data --paper_filters
    
    # Download all paper tissues
    python scripts/download_hest.py --tissue all --output_dir ./hest_data --paper_filters
    
    # List available tissues
    python scripts/download_hest.py --list_tissues

Note: You must set your HuggingFace token via:
    1. Edit HF_TOKEN in this script
    2. Set HF_TOKEN environment variable
    3. Pass --token argument
        """
    )
    
    parser.add_argument('--tissue', type=str, default=None,
                        help='Tissue to download (e.g., lung, breast, colon). Use "all" for all paper tissues.')
    parser.add_argument('--output_dir', type=str, default='./hest_data',
                        help='Output directory for downloaded data')
    parser.add_argument('--paper_filters', action='store_true',
                        help='Apply paper filters: human only, no Xenium')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace access token')
    parser.add_argument('--list_tissues', action='store_true',
                        help='List available tissues and exit')
    parser.add_argument('--save_metadata', action='store_true',
                        help='Save metadata CSV to output directory')
    
    args = parser.parse_args()
    
    # Load metadata
    meta_df = load_metadata()
    
    # List tissues and exit
    if args.list_tissues:
        list_tissues(meta_df)
        return
    
    # Validate arguments
    if not args.tissue:
        parser.error("--tissue is required (use --list_tissues to see options)")
    
    # Get token
    token = get_token(args.token)
    if not token:
        print("\n" + "="*60)
        print("WARNING: No HuggingFace token found!")
        print("="*60)
        print("You need to set your token via one of:")
        print("  1. Edit HF_TOKEN at the top of this script")
        print("  2. Set HF_TOKEN environment variable")
        print("  3. Pass --token argument")
        print("\nGet your token at: https://huggingface.co/settings/tokens")
        print("Accept HEST terms at: https://huggingface.co/datasets/MahmoodLab/hest")
        print("="*60 + "\n")
        return
    
    # Apply paper filters if requested
    if args.paper_filters:
        meta_df = apply_paper_filters(meta_df)
    
    # Determine tissues to download
    if args.tissue.lower() == 'all':
        tissues = PAPER_TISSUES
    else:
        tissues = [args.tissue]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata if requested
    if args.save_metadata:
        meta_path = output_dir / "hest_metadata.csv"
        meta_df.to_csv(meta_path, index=False)
        print(f"Saved metadata to {meta_path}")
    
    # Download each tissue
    for tissue in tissues:
        print(f"\n{'='*60}")
        print(f"Processing: {tissue}")
        print('='*60)
        
        sample_ids, organ_name = get_tissue_samples(meta_df, tissue)
        
        if len(sample_ids) == 0:
            print(f"  No samples found for {tissue}")
            continue
        
        print(f"  Found {len(sample_ids)} samples for {organ_name}")
        
        # Download
        download_samples(sample_ids, str(output_dir / "raw"), token)
        
        # Organize
        dir_name = TISSUE_TO_DIR.get(organ_name, tissue.lower())
        tissue_dir = output_dir / f"hest_data_{dir_name}"
        organize_data(output_dir / "raw", tissue_dir, sample_ids)
    
    # Cleanup raw directory
    raw_dir = output_dir / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    
    print(f"\n{'='*60}")
    print("Download complete!")
    print('='*60)
    print(f"\nData organized in: {output_dir}")
    print("\nDirectory structure:")
    for tissue in tissues:
        sample_ids, organ_name = get_tissue_samples(meta_df, tissue)
        if len(sample_ids) > 0:
            dir_name = TISSUE_TO_DIR.get(organ_name, tissue.lower())
            print(f"  {output_dir}/hest_data_{dir_name}/")
            print(f"    ├── patches/  ({len(sample_ids)} .h5 files)")
            print(f"    └── st/       ({len(sample_ids)} .h5ad files)")
    
    if args.paper_filters:
        print("\nPaper filters applied:")
        print("  - Species: Homo sapiens only")
        print("  - Technology: Xenium excluded")


if __name__ == '__main__':
    main()
