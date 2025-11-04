from lidar_filter import LidarFilter
from intensity_extractor import IntensityExtractor
from lidar_merger import LidarMerger
import os

BASE_INPUT_DIR = "/home/ezarisma/workspace/Project1/sequences"  
PCD_BASE_DIR = "/home/ezarisma/workspace/Project1/pcd_data"   
OUTPUT_EXTRACT_DIR = "./extracted_intensity"
MERGE_OUTPUT_DIR = "./merged_lidar_points"

os.makedirs(OUTPUT_EXTRACT_DIR, exist_ok=True)
os.makedirs(MERGE_OUTPUT_DIR, exist_ok=True)

lidars = ["lidar0", "lidar1"]
filter_values = [0, 1]
filtered_results = {}

print("\nStep 1: Filtering LIDAR points (in memory)")
for lidar, fval in zip(lidars, filter_values):
    print(f"\nFiltering {lidar} with filter_value={fval}")
    lidar_filter = LidarFilter(BASE_INPUT_DIR, filter_value=fval)
    filtered_results[lidar] = lidar_filter.process_scenes(return_data=True)

print("\nStep 2: Extracting intensity directly")
extractor = IntensityExtractor(pcd_base_dir=PCD_BASE_DIR, output_base_dir=OUTPUT_EXTRACT_DIR)

for lidar, fval in zip(lidars, filter_values):
    lidar_pcd_dir = os.path.join(PCD_BASE_DIR, lidar)
    lidar_output_dir = os.path.join(OUTPUT_EXTRACT_DIR, lidar)
    os.makedirs(lidar_output_dir, exist_ok=True)
    
    extractor.extract_from_memory(filtered_data=filtered_results[lidar], pcd_base_dir=lidar_pcd_dir, lidar_output_dir=lidar_output_dir, filter_value=fval)

print("\nStep 3: Merging LIDAR points from memory")
extracted_dirs = [os.path.join(OUTPUT_EXTRACT_DIR, lidar) for lidar in lidars]
merger = LidarMerger(extracted_dirs=extracted_dirs, output_dir=MERGE_OUTPUT_DIR, sequence_base_dir=BASE_INPUT_DIR)
merger.merge_all_scenes()


print("\nComplete!")
