import os
import numpy as np

# Directory paths
LIDAR0_LABEL_DIR = "/home/ezarisma/Downloads/Practice/Semantic-KITTI-API-DPAI-main-ver3/filtered/sequences_lidar0"
LIDAR1_LABEL_DIR = "/home/ezarisma/Downloads/Practice/Semantic-KITTI-API-DPAI-main-ver3/filtered/sequences_lidar1"
OUTPUT_LABEL_DIR = "/home/ezarisma/Downloads/Practice/Semantic-KITTI-API-DPAI-main-ver3/merged_lidar_points"

# Scenes and frames to merge
scenes = ["00", "01", "02","03", "04", "05","06", "07"]
frames_to_merge = ["000", "005", "010", "015", "020","025", "030", "035", "040", "045","050", "055", "060", "065", "070","075", "080", "085", "090", "095"]

print("COMBINING LABEL FILES FOR MERGED LIDAR DATA")
print("=" * 80)

# Create output directory
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

for scene in scenes:
    print(f"\nScene {scene}")
    
    # Create scene directory in output
    scene_output_label_dir = os.path.join(OUTPUT_LABEL_DIR, scene, "labels")
    os.makedirs(scene_output_label_dir, exist_ok=True)
    
    lidar0_label_dir = os.path.join(LIDAR0_LABEL_DIR, scene, "labels")
    lidar1_label_dir = os.path.join(LIDAR1_LABEL_DIR, scene, "labels")
    
    if not os.path.exists(lidar0_label_dir) or not os.path.exists(lidar1_label_dir):
        print(f"Missing one or both label directories")
        continue
    
    for frame in frames_to_merge:
        lidar0_label_file = os.path.join(lidar0_label_dir, f"{frame}.label")
        lidar1_label_file = os.path.join(lidar1_label_dir, f"{frame}.label")
        output_label_file = os.path.join(scene_output_label_dir, f"{frame}.label")
        
        print(f"Processing labels for frame {frame}: ", end="")
        
        if not os.path.exists(lidar0_label_file) or not os.path.exists(lidar1_label_file):
            print("Missing one or both label files")
            continue
        
        try:
            # Load both label files
            labels0 = np.fromfile(lidar0_label_file, dtype=np.uint32)
            labels1 = np.fromfile(lidar1_label_file, dtype=np.uint32)
            
            # Concatenate labels 
            merged_labels = np.concatenate((labels0, labels1))
            
            # Save merged label file
            merged_labels.tofile(output_label_file)
            
            print(f"MERGED - {len(labels0):,} + {len(labels1):,} = {len(merged_labels):,} labels")
            
            # Show label distribution
            unique_labels0 = np.unique(labels0)
            unique_labels1 = np.unique(labels1)
            unique_merged = np.unique(merged_labels)
            
            print(f"        LIDAR0 unique labels: {len(unique_labels0)}")
            print(f"        LIDAR1 unique labels: {len(unique_labels1)}")
            print(f"        MERGED unique labels: {len(unique_merged)}")
            
        except Exception as e:
            print(f"Error: {e}")
