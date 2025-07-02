#!/usr/bin/env python3
# Script to normalize .npy files between 0 and 1

import os
import numpy as np
from pathlib import Path
import glob

# Source path containing the .npy files


def normalize_npy_files(feature, env, seq):

    SOURCE_PATH = f"/home/arvc/Marcos/INVESTIGACION/0_DATASETS/COLD/FEATURES_PRUEBA/{feature}/{env}/{seq}/"

    """
    Read all .npy files in the source directory, find the global maximum value,
    and normalize all arrays to the range [0, 1].
    """
    # Check if the source directory exists
    if not os.path.exists(SOURCE_PATH):
        print(f"Source path {SOURCE_PATH} does not exist.")
        return
    
    # Get all .npy files in the directory and its subdirectories
    npy_files = glob.glob(os.path.join(SOURCE_PATH, "**", "*.npy"), recursive=True)
    
    if not npy_files:
        print(f"No .npy files found in {SOURCE_PATH}")
        return
    
    # print(f"Found {len(npy_files)} .npy files in {SOURCE_PATH}")
    
    # First pass: find the maximum value across all .npy files
    global_max = float('-inf')
    
    # print("Finding maximum value across all files...")
    for file_path in npy_files:
        try:
            array = np.load(file_path)
            file_max = np.max(array)
            global_max = max(global_max, file_max)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # print(f"Global maximum value found: {global_max}")
    
    if global_max <= 0:
        print("Invalid maximum value. Cannot normalize.")
        return
    else:
        return global_max


def normalize_npy_files_360Loc(feature, env, seq, ilum, database=True):

    if database:
        SOURCE_PATH = f"/home/arvc/Marcos/INVESTIGACION/0_DATASETS/360LOC/{env}/mapping/{ilum}_360_{seq}/FEATURES/{feature}/"
    else:
        SOURCE_PATH = f"/home/arvc/Marcos/INVESTIGACION/0_DATASETS/360LOC/{env}/query_360/{ilum}_360_{seq}/FEATURES/{feature}/"
    """
    Read all .npy files in the source directory, find the global maximum value,
    and normalize all arrays to the range [0, 1].
    """
    # Check if the source directory exists
    if not os.path.exists(SOURCE_PATH):
        print(f"Source path {SOURCE_PATH} does not exist.")
        return

    # Get all .npy files in the directory and its subdirectories
    npy_files = glob.glob(os.path.join(SOURCE_PATH, "**", "*.npy"), recursive=True)

    if not npy_files:
        print(f"No .npy files found in {SOURCE_PATH}")
        return

    # print(f"Found {len(npy_files)} .npy files in {SOURCE_PATH}")

    # First pass: find the maximum value across all .npy files
    global_max = float('-inf')

    # print("Finding maximum value across all files...")
    for file_path in npy_files:
        try:
            array = np.load(file_path)
            file_max = np.max(array)
            global_max = max(global_max, file_max)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # print(f"Global maximum value found: {global_max}")

    if global_max <= 0:
        print("Invalid maximum value. Cannot normalize.")
        return
    else:
        return global_max