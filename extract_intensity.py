import os
import glob
import numpy as np
from pypcd4 import PointCloud

# --- DIRECTORY DEFINITIONS ---
PCD_BASE_DIR = "/home/ezarisma/Downloads/Practice/Semantic-KITTI-API-DPAI-main-ver3/pcd_data/07/07/lidar_point_cloud_top_rear_lidar"

BIN_BASE_DIR = "/home/ezarisma/Downloads/Practice/Semantic-KITTI-API-DPAI-main-ver3/filtered/sequences_lidar1/07/velodyne" 

# Define the new directory to save the MODIFIED BIN files
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(PCD_BASE_DIR)), "extracted_intensity")

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory created at: {OUTPUT_DIR}")
print("-" * 50)

# 1. Get a sorted list of all numeric .pcd files
all_pcd_files = sorted(glob.glob(os.path.join(PCD_BASE_DIR, "*.pcd")))
target_stems = []

for full_path in all_pcd_files:
    filename_stem = os.path.splitext(os.path.basename(full_path))[0]
    try:
        int(filename_stem)
        target_stems.append(filename_stem)
    except ValueError:
        continue

print(f"Found {len(target_stems)} PCD files to process.")
print(f"Attempting to read existing BIN files from: {BIN_BASE_DIR}")
print("-" * 50)

# 2. Process each pair (PCD and BIN)
for stem in target_stems:
    pcd_filename = f"{stem}.pcd"
    
    # Pad stem to 3 digits for BIN filename
    try:
        padded_stem = f"{int(stem):03d}"  
    except ValueError:
        print(f"  ERROR: Could not parse numeric stem from {pcd_filename}. Skipping.")
        continue
        
    bin_filename = f"{padded_stem}.bin"

    pcd_path = os.path.join(PCD_BASE_DIR, pcd_filename)
    bin_path = os.path.join(BIN_BASE_DIR, bin_filename)
    output_path = os.path.join(OUTPUT_DIR, bin_filename)

    print(f"Processing: {pcd_filename} -> {bin_filename} (Merging)")

    try:
        # A. EXTRACT NEW INTENSITY VALUE FROM PCD
        pc: PointCloud = PointCloud.from_path(pcd_path)
        
        # Safely extract intensity field to be the new remission
        if 'intensity' in pc.fields:
            intensity_array = pc.numpy(('intensity',)).flatten()
        elif 'remission' in pc.fields:
            intensity_array = pc.numpy(('remission',)).flatten()
        else:
            raise ValueError("PCD file is missing both 'intensity' and 'remission' fields.")

        # NORMALIZE INTENSITY TO RANGE [0, 1] BY DIVIDING BY 255.0
        if len(intensity_array) > 0:
            normalized_intensity = intensity_array / 255.0
  
            normalized_intensity = np.clip(normalized_intensity, 0.0, 1.0)
            
            original_min = np.min(intensity_array)
            original_max = np.max(intensity_array)
            normalized_min = np.min(normalized_intensity)
            normalized_max = np.max(normalized_intensity)
            
            print(f"  Intensity range: [{original_min:.3f}, {original_max:.3f}] -> Normalized to [{normalized_min:.3f}, {normalized_max:.3f}]")
        else:
            normalized_intensity = intensity_array  # Empty array

        # B. LOAD EXISTING BINARY DATA
        bin_data = np.fromfile(bin_path, dtype=np.float32)
        
        # Reshape to (N, 4) if the size is a multiple of 4 floats
        if bin_data.size % 4 != 0:
            raise ValueError(f"BIN file size ({bin_data.size} floats) is not a multiple of 4. Data corrupt.")
            
        bin_data = bin_data.reshape((-1, 4))
        
        # C. VALIDATE AND MERGE
        if len(bin_data) != len(normalized_intensity):
            print(f"  ERROR: Point counts do not match! BIN: {len(bin_data)}, PCD: {len(normalized_intensity)}. Skipping.")
            continue
            
        # Replace the remission column (index 3) with the NORMALIZED intensity data
        bin_data[:, 3] = normalized_intensity
        
        # D. SAVE MODIFIED DATA
        bin_data.tofile(output_path)
        
        print(f"  SUCCESS: Replaced remission channel for {len(bin_data)} points and saved to {output_path}")

    except FileNotFoundError:
        # Catches the initial error if the BIN file is missing
        print(f"  ERROR: Could not find matching BIN file at {bin_path}. Skipping.")
    except Exception as e:
        print(f"  An unexpected error occurred for {pcd_filename}: {e}. Skipping.")
        
    print("-" * 50)

print("All file merging complete!")
