import os
import numpy as np
import open3d as o3d

# Directory paths
LIDAR0_DIR = "/home/ezarisma/Downloads/Practice/Semantic-KITTI-API-DPAI-main-ver3/lidar_pcd/lidar_0"
LIDAR1_DIR = "/home/ezarisma/Downloads/Practice/Semantic-KITTI-API-DPAI-main-ver3/lidar_pcd/lidar_1"
OUTPUT_DIR = "/home/ezarisma/Downloads/Practice/Semantic-KITTI-API-DPAI-main-ver3/new"

# Scenes and frames to merge
scenes = ["00", "01", "02", "03", "04", "05", "06", "07"]
frames_to_merge = ["000", "005", "010", "015", "020", "025", "030", "035", "040", "045", 
                   "050", "055", "060", "065", "070", "075", "080", "085", "090", "095"]

print("MERGING LIDAR0 AND LIDAR1 .PCD FILES INTO SINGLE FILES")
print("=" * 80)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_pcd_file(pcd_path):
    """Read PCD file and return point cloud data"""
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        
        # Check if colors exist
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        return points, colors, len(points)
    except Exception as e:
        print(f"Error reading PCD file {pcd_path}: {e}")
        return None, None, 0

def merge_point_clouds(points1, colors1, points2, colors2):
    """Merge two point clouds"""
    if points2 is None or len(points2) == 0:
        return points1, colors1
    
    merged_points = np.vstack((points1, points2))
    
    merged_colors = None
    if colors1 is not None and colors2 is not None:
        merged_colors = np.vstack((colors1, colors2))
    elif colors1 is not None:
        merged_colors = colors1
    elif colors2 is not None:
        merged_colors = colors2
    
    return merged_points, merged_colors

def save_pcd_file(points, colors, output_path):
    """Save point cloud to PCD file"""
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(output_path, pcd)
        return True
    except Exception as e:
        print(f"Error saving PCD file {output_path}: {e}")
        return False

for scene in scenes:
    print(f"\nScene {scene}")
    
    # Create scene directory in output
    scene_output_dir = os.path.join(OUTPUT_DIR, scene, "velodyne")
    os.makedirs(scene_output_dir, exist_ok=True)
    
    lidar0_scene_dir = os.path.join(LIDAR0_DIR, scene, "velodyne")
    lidar1_scene_dir = os.path.join(LIDAR1_DIR, scene, "velodyne")
    
    if not os.path.exists(lidar0_scene_dir):
        print(f"Missing LIDAR0 directory: {lidar0_scene_dir}")
        continue
    
    for frame in frames_to_merge:
        lidar0_file = os.path.join(lidar0_scene_dir, f"{frame}.pcd")
        lidar1_file = os.path.join(lidar1_scene_dir, f"{frame}.pcd")
        output_file = os.path.join(scene_output_dir, f"{frame}.pcd")
        
        print(f"Processing frame {frame}: ", end="")
        
        if not os.path.exists(lidar0_file):
            print("Missing LIDAR0 file")
            continue
        
        try:
            # Read LIDAR0 data
            points0, colors0, count0 = read_pcd_file(lidar0_file)
            
            if points0 is None:
                print("Failed to read LIDAR0 file")
                continue
            
            # Check if LIDAR1 file exists and has data
            lidar1_exists = os.path.exists(lidar1_file) and os.path.getsize(lidar1_file) > 0
            points1, colors1, count1 = None, None, 0
            
            if lidar1_exists:
                # Read LIDAR1 data
                points1, colors1, count1 = read_pcd_file(lidar1_file)
            
            # Only merge if LIDAR1 has MORE THAN 0 points
            if lidar1_exists and points1 is not None and count1 > 0:
                # Merge point clouds
                merged_points, merged_colors = merge_point_clouds(points0, colors0, points1, colors1)
                total_points = len(merged_points)
                
                # Save merged file
                if save_pcd_file(merged_points, merged_colors, output_file):
                    print(f"MERGED - {count0:,} + {count1:,} = {total_points:,} points")
                else:
                    print("Failed to save merged file")
            else:
                # Use only LIDAR0 data (LIDAR1 has 0 points or doesn't exist)
                if save_pcd_file(points0, colors0, output_file):
                    if lidar1_exists and count1 == 0:
                        print(f"LIDAR0 ONLY - {count0:,} points (LIDAR1 has 0 points)")
                    elif lidar1_exists and points1 is None:
                        print(f"LIDAR0 ONLY - {count0:,} points (LIDAR1 file corrupted)")
                    else:
                        print(f"LIDAR0 ONLY - {count0:,} points (LIDAR1 file missing)")
                else:
                    print("Failed to save LIDAR0 file")
                    
        except Exception as e:
            print(f"Error: {e}")

print("\nPCD file merging completed!")
