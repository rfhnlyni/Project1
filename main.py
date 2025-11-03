import os
from lidar_filter import LidarFilter
from intensity_extractor import IntensityExtractor
from lidar_merger import LidarMerger

BASE_INPUT_DIR = "/home/ezarisma/workspace/Project1/sequences"         # Original raw LIDAR data
PCD_BASE_DIR = "/home/ezarisma/workspace/Project1/pcd_data" 
BIN_BASE_DIR = "./filtered"           # PCD files          
OUTPUT_EXTRACT_DIR = "./extracted_intensity"
MERGE_OUTPUT_DIR = "./merged_lidar_points"

lidars = ["lidar0", "lidar1"]  # Changes required
filtered_dirs = [f"./filtered/{lidar}" for lidar in lidars]

# Filter LIDAR points
print("\nStep1: Filtering LIDAR points")
for i, lidar in enumerate(lidars):
    lidar_filter = LidarFilter(BASE_INPUT_DIR, filtered_dirs[i], filter_value=i)
    lidar_filter.process_scenes()

# Extract intensity
print("\nStep2: Extracting intensity")
extractor = IntensityExtractor(pcd_base_dir=PCD_BASE_DIR, bin_base_dir=BIN_BASE_DIR, output_base_dir=OUTPUT_EXTRACT_DIR, lidars=lidars)
extractor.extract_all_lidars()

# Merge LIDAR points
print("\nStep 3: Merging LIDAR points ")
merger = LidarMerger(lidar0_dir=os.path.join(OUTPUT_EXTRACT_DIR, "lidar0"), lidar1_dir=os.path.join(OUTPUT_EXTRACT_DIR, "lidar1"), output_dir=MERGE_OUTPUT_DIR)
merger.merge_all_scenes()

print("\nComplete!")
