import os
from lidar_filter import LidarFilter
from intensity_extractor import IntensityExtractor
from lidar_merger import LidarMerger

BASE_INPUT_DIR = "/home/ezarisma/workspace/Project1/sequences"         # Original raw LIDAR data
FILTERED_DIR_LIDAR0 = "./filtered/lidar0"
FILTERED_DIR_LIDAR1 = "./filtered/lidar1"

PCD_BASE_DIR = "/home/ezarisma/workspace/Project1/pcd_data"           # PCD files
BIN_BASE_DIR = "./filtered"                  
OUTPUT_EXTRACT_DIR = "./extracted_intensity"
MERGE_OUTPUT_DIR = "./merged_lidar_points"

# Filter LIDAR points
print("\nStep1: Filtering LIDAR points")
filter0 = LidarFilter(BASE_INPUT_DIR, FILTERED_DIR_LIDAR0, filter_value=0)
filter1 = LidarFilter(BASE_INPUT_DIR, FILTERED_DIR_LIDAR1, filter_value=1)

filter0.process_scenes()
filter1.process_scenes()

# Extract intensity
print("\nStep2: Extracting intensity")
extractor = IntensityExtractor(pcd_base_dir=PCD_BASE_DIR, bin_base_dir=BIN_BASE_DIR, output_base_dir=OUTPUT_EXTRACT_DIR, lidars=["lidar0", "lidar1"])
extractor.extract_all_lidars()

# Merge LIDAR points
print("\nStep 3: Merging LIDAR points ")
merger = LidarMerger(lidar0_dir=os.path.join(OUTPUT_EXTRACT_DIR, "lidar0"), lidar1_dir=os.path.join(OUTPUT_EXTRACT_DIR, "lidar1"), output_dir=MERGE_OUTPUT_DIR)
merger.merge_all_scenes()

print("\nComplete!")
