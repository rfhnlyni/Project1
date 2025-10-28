import os
import numpy as np

# Base input directory
base_input_dir = "/home/ezarisma/Downloads/Practice/Semantic-KITTI-API-DPAI-main-ver3/data/sequences"

# Base output directory
base_output_dir = "./filtered/sequences"

# Process scenes 00 to 07
scenes = [f"{i:02d}" for i in range(8)]  # ['00', '01', '02', '03', '04', '05', '06', '07']

for scene in scenes:
    print(f"\nProcessing Scene {scene}")
    
    # Input folders for current scene
    velodyne_dir = os.path.join(base_input_dir, scene, "velodyne")
    label_dir = os.path.join(base_input_dir, scene, "labels")
    
    # Output folders for current scene
    output_velodyne_dir = os.path.join(base_output_dir, scene, "velodyne")
    output_label_dir = os.path.join(base_output_dir, scene, "labels")
    
    # Create output directories
    os.makedirs(output_velodyne_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Check if input directories exist
    if not os.path.exists(velodyne_dir):
        print(f"[WARNING] Velodyne directory not found: {velodyne_dir}")
        continue
    if not os.path.exists(label_dir):
        print(f"[WARNING] Label directory not found: {label_dir}")
        continue
    
    # Get sorted .bin files
    bin_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    
    if not bin_files:
        print(f"[WARNING] No .bin files found in {velodyne_dir}")
        continue
    
    processed_count = 0
    total_points_before = 0
    total_points_after = 0
    
    for filename in bin_files:
        bin_path = os.path.join(velodyne_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".bin", ".label"))
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"[WARNING] Label file not found: {label_path}")
            continue
        
        try:
            # Load point cloud: Nx4 [x, y, z, remission]
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            remissions = points[:, 3]
            
            # Load labels
            labels = np.fromfile(label_path, dtype=np.uint32)
            
            # Validate sizes
            if len(labels) != points.shape[0]:
                print(f"[ERROR] Size mismatch in {scene}/{filename}: {len(labels)} labels vs {points.shape[0]} points")
                continue
            
            # Filter where remission == 1 ONLY
            mask = remissions == 1
            filtered_points = points[mask]
            filtered_labels = labels[mask]
            
            # Save filtered .bin
            save_bin_path = os.path.join(output_velodyne_dir, filename)
            filtered_points.tofile(save_bin_path)
            
            # Save filtered .label
            save_label_path = os.path.join(output_label_dir, filename.replace(".bin", ".label"))
            filtered_labels.tofile(save_label_path)
            
            # Statistics
            processed_count += 1
            total_points_before += len(points)
            total_points_after += len(filtered_points)
            
            # CORRECTED: Changed message to show remission=1
            print(f"[OK] Scene {scene}: {filename} - {len(filtered_points)}/{len(points)} points kept (remission=1)")
            
        except Exception as e:
            print(f"[ERROR] Processing {scene}/{filename}: {str(e)}")
            continue
    
    # Print scene summary
    if processed_count > 0:
        reduction_percent = ((total_points_before - total_points_after) / total_points_before) * 100
        print(f"\nScene {scene} Summary")
        print(f"Processed files: {processed_count}/{len(bin_files)}")
        print(f"Points kept (remission=1): {total_points_after}")
        print(f"Output saved to: {output_velodyne_dir}")

print("\nProcessing Complete")
print(f"Processed scenes: {scenes}")
print(f"Output base directory: {base_output_dir}")
