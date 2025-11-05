import os
import numpy as np
from pypcd4 import PointCloud

class IntensityExtractor:
    def __init__(self, pcd_base_dir, output_base_dir):
        """
        Args:
            pcd_base_dir (str): Path to base directory containing PCD files.
            output_base_dir (str): Path to directory where new BIN files will be stored.
        """
        self.pcd_base_dir = pcd_base_dir
        self.output_base_dir = output_base_dir

    def extract_from_memory(self, filtered_data, pcd_base_dir, lidar_output_dir, filter_value):
        """
        Extracts intensity values from PCD files and saves them as new BIN files,
        using the filtered points from memory.
        
        Args:
            filtered_data (dict): {scene: {filename: (points, labels)}}
            pcd_base_dir (str): Path to directory containing PCD files.
            lidar_output_dir (str): Output folder for lidar data.
            filter_value (int/float): Filtered remission/intensity value.
        """
        
        print(f"\nProcessing LiDAR Data (Remission = {filter_value})")
        
        # Loop over each scene
        for scene, files in filtered_data.items():
            print(f"\nProcessing Scene {scene}")
            vel_output_dir = os.path.join(lidar_output_dir, scene, "velodyne")
            label_output_dir = os.path.join(lidar_output_dir, scene, "labels")
            os.makedirs(vel_output_dir, exist_ok=True)
            os.makedirs(label_output_dir, exist_ok=True)

            # Loop over each file in the scene
            for filename, (points, labels) in files.items():
                stem = os.path.splitext(filename)[0]
                pcd_path = os.path.join(pcd_base_dir, scene, f"{stem}.pcd")

                try:
                    pc = PointCloud.from_path(pcd_path)

                    # Check if intensity/remission field exists
                    if "intensity" in pc.fields:
                        intensity = pc.numpy(("intensity",)).flatten()
                    elif "remission" in pc.fields:
                        intensity = pc.numpy(("remission",)).flatten()
                    else:
                        print(f"[WARN] Missing intensity/remission for {stem}")
                        continue

                    # Only keep the filtered points
                    filtered_points = points
                    filtered_labels = labels
                    filtered_intensity = intensity[:len(points)]

                    # Skip files with no points
                    if len(filtered_points) == 0:
                        print(f"[SKIP] {scene}/{filename} - no points after filtering")
                        continue

                    # Update the 4th column of points with intensity (normalized 0–1)
                    filtered_points[:, 3] = filtered_intensity / 255.0

                    # Save new BIN and label
                    points_out_path = os.path.join(vel_output_dir, filename)
                    labels_out_path = os.path.join(label_output_dir, filename.replace(".bin", ".label"))

                    filtered_points.tofile(points_out_path)
                    filtered_labels.tofile(labels_out_path)

                    print(f"[EXTRACTED] {scene}/{filename} | points: {len(filtered_points)} | "
                          f"min: {filtered_points[:,3].min():.3f} max: {filtered_points[:,3].max():.3f}")

                except Exception as e:
                    print(f"[ERROR] {stem}: {e}")

