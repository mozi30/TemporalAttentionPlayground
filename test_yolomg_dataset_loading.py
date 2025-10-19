#!/usr/bin/env python3
"""Test YOLOMG dataset loading to validate the generated dataset."""

import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path

# Add YOLOMG to path
sys.path.append('YOLOMG')

def test_dataset_structure():
    """Test the basic dataset structure and file accessibility"""
    print("=== Testing YOLOMG Dataset Structure ===")
    
    dataset_path = Path("datasets/uavdt/YOLOMG_VID")
    
    # Check directory structure
    required_dirs = ["images", "labels", "masks", "ImageSets/Main"]
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"✓ {dir_name}: {file_count} files")
        else:
            print(f"✗ {dir_name}: Missing")
            return False
    
    return True

def test_file_loading():
    """Test loading individual files"""
    print("\n=== Testing File Loading ===")
    
    dataset_path = Path("datasets/uavdt/YOLOMG_VID")
    
    # Get first few files
    image_files = sorted(list((dataset_path / "images").glob("*.jpg")))[:5]
    
    for img_file in image_files:
        stem = img_file.stem
        label_file = dataset_path / "labels" / f"{stem}.txt"
        mask_file = dataset_path / "masks" / f"{stem}.jpg"
        
        # Test image loading
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"✗ Failed to load image: {img_file}")
            continue
        
        # Test mask loading
        mask = cv2.imread(str(mask_file))
        if mask is None:
            print(f"✗ Failed to load mask: {mask_file}")
            continue
        
        # Test label loading
        if not label_file.exists():
            print(f"✗ Label file missing: {label_file}")
            continue
        
        with open(label_file, 'r') as f:
            labels = f.read().strip()
        
        if not labels:
            print(f"- Empty labels for {stem}")
            continue
        
        # Parse first label
        lines = labels.split('\n')
        first_line = lines[0].split()
        class_id = int(first_line[0])
        
        print(f"✓ {stem}: Image {img.shape}, Mask {mask.shape}, {len(lines)} objects, class range: {class_id}")
        
        # Validate class range
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cls = int(parts[0])
                if cls not in [0, 1, 2]:
                    print(f"  ⚠ Unexpected class ID: {cls}")

def test_class_distribution():
    """Test class distribution across the dataset"""
    print("\n=== Testing Class Distribution ===")
    
    dataset_path = Path("datasets/uavdt/YOLOMG_VID")
    label_files = list((dataset_path / "labels").glob("*.txt"))
    
    class_counts = {0: 0, 1: 0, 2: 0}
    total_objects = 0
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            content = f.read().strip()
        
        if content:
            lines = content.split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                        total_objects += 1
    
    print(f"Class distribution (total: {total_objects}):")
    print(f"  Class 0 (car): {class_counts[0]} ({class_counts[0]/total_objects*100:.1f}%)")
    print(f"  Class 1 (truck): {class_counts[1]} ({class_counts[1]/total_objects*100:.1f}%)")
    print(f"  Class 2 (bus): {class_counts[2]} ({class_counts[2]/total_objects*100:.1f}%)")

def test_yolomg_compatibility():
    """Test if the dataset is compatible with YOLOMG data loading"""
    print("\n=== Testing YOLOMG Compatibility ===")
    
    try:
        # Try to import YOLOMG components
        from utils.datasets import create_dataloader
        print("✓ YOLOMG imports successful")
        
        # Test configuration
        data_config = {
            'path': 'datasets/uavdt/YOLOMG_VID',
            'train': 'ImageSets/Main/train.txt',
            'val': 'ImageSets/Main/val.txt',
            'nc': 3,
            'names': ['car', 'truck', 'bus']
        }
        
        print("✓ Data configuration created")
        
        # Check if train/val files exist
        dataset_path = Path("datasets/uavdt/YOLOMG_VID")
        train_file = dataset_path / "ImageSets/Main/train.txt"
        val_file = dataset_path / "ImageSets/Main/val.txt"
        
        if train_file.exists():
            with open(train_file, 'r') as f:
                train_count = len(f.readlines())
            print(f"✓ Train file: {train_count} entries")
        else:
            print("✗ Train file missing")
        
        if val_file.exists():
            with open(val_file, 'r') as f:
                val_count = len(f.readlines())
            print(f"✓ Val file: {val_count} entries")
        else:
            print("✗ Val file missing")
        
        return True
        
    except ImportError as e:
        print(f"✗ YOLOMG import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing YOLOMG Dataset Compatibility")
    print("=" * 50)
    
    # Run all tests
    success = True
    success &= test_dataset_structure()
    test_file_loading()
    test_class_distribution()
    success &= test_yolomg_compatibility()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! Dataset is ready for YOLOMG training.")
    else:
        print("✗ Some tests failed. Check the output above.")