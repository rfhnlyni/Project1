import os
import numpy as np

# Directory paths
LIDAR0_DIR = "/home/ezarisma/Downloads/Practice/Semantic-KITTI-API-DPAI-main-ver3/extracted_intensity_lidar0"
LIDAR1_DIR = "/home/ezarisma/Downloads/Practice/Semantic-KITTI-API-DPAI-main-ver3/extracted_intensity_lidar1"
OUTPUT_DIR = "/home/ezarisma/Downloads/Practice/Semantic-KITTI-API-DPAI-main-ver3/merged_lidar_points"

# Scenes and frames to merge
scenes = ["00", "01", "02","03", "04", "05","06", "07"]
frames_to_merge = ["000", "005", "010", "015", "020","025", "030", "035", "040", "045","050", "055", "060", "065", "070","075", "080", "085", "090", "095"]

print("MERGING LIDAR0 AND LIDAR1 POINTS INTO SINGLE FILES")
print("=" * 80)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

for scene in scenes:
    print(f"\nScene {scene}")
    
    # Create scene directory in output
    scene_output_dir = os.path.join(OUTPUT_DIR, scene, "velodyne")
    os.makedirs(scene_output_dir, exist_ok=True)
    
    lidar0_scene_dir = os.path.join(LIDAR0_DIR, scene, "velodyne")
    lidar1_scene_dir = os.path.join(LIDAR1_DIR, scene, "velodyne")
    
    if not os.path.exists(lidar0_scene_dir):
        print(f"Missing LIDAR0 directory")
        continue
    
    for frame in frames_to_merge:
        lidar0_file = os.path.join(lidar0_scene_dir, f"{frame}.bin")
        lidar1_file = os.path.join(lidar1_scene_dir, f"{frame}.bin")
        output_file = os.path.join(scene_output_dir, f"{frame}.bin")
        
        print(f"Processing frame {frame}: ", end="")
        
        if not os.path.exists(lidar0_file):
            print("Missing LIDAR0 file")
            continue
        
        try:
            # Load LIDAR0 data
            points0 = np.fromfile(lidar0_file, dtype=np.float32).reshape(-1, 4)
            
            # Check if LIDAR1 file exists and has data
            if os.path.exists(lidar1_file) and os.path.getsize(lidar1_file) > 0:
                # Load LIDAR1 data
                points1 = np.fromfile(lidar1_file, dtype=np.float32).reshape(-1, 4)
                
                # Concatenate points from both lidars
                merged_points = np.vstack((points0, points1))
                
                print(f"MERGED - {len(points0):,} + {len(points1):,} = {len(merged_points):,} points")
                print(f"        LIDAR0 intensity: [{points0[:, 3].min():.4f}, {points0[:, 3].max():.4f}]")
                print(f"        LIDAR1 intensity: [{points1[:, 3].min():.4f}, {points1[:, 3].max():.4f}]")
                print(f"        MERGED intensity: [{merged_points[:, 3].min():.4f}, {merged_points[:, 3].max():.4f}]")
            else:
                # Use only LIDAR0 data
                merged_points = points0
                print(f"LIDAR0 ONLY - {len(points0):,} points (LIDAR1 empty/missing)")
                print(f"        LIDAR0 intensity: [{points0[:, 3].min():.4f}, {points0[:, 3].max():.4f}]")
            
            # Save merged file
            merged_points.tofile(output_file)
            
        except Exception as e:
            print(f"Error: {e}")
