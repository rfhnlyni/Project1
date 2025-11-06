import os
import numpy as np
import shutil

class LidarMerger:
    def __init__(self, extracted_dirs: list, output_dir, sequence_base_dir=None):
        """
        Merge LiDAR outputs

        Args:
            extracted_dirs: Path to save after extract intensity
            output_dir (str): Path to directory where new BIN files will be stored
            sequence_base_dir (str): Path to copy extra files/folders
        """
        self.extracted_dirs = extracted_dirs
        self.output_dir = output_dir
        self.sequence_base_dir = sequence_base_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def merge_all_scenes(self):
        """
        Merges all scenes
        """
        scenes = sorted([
            s for s in os.listdir(self.extracted_dirs[0])
            if os.path.isdir(os.path.join(self.extracted_dirs[0], s))
        ])
        print(f"Detected scenes: {scenes}")

        for scene in scenes:
            self.merge_scene(scene)

    def merge_scene(self, scene):
        """
        Merge points and labels from all LiDARs for a single scene.
        """
        print(f"\nProcessing Scene {scene}")
        scene_output_dir = os.path.join(self.output_dir, scene)
        velodyne_output_dir = os.path.join(scene_output_dir, "velodyne")
        labels_output_dir = os.path.join(scene_output_dir, "labels")
        os.makedirs(velodyne_output_dir, exist_ok=True)
        os.makedirs(labels_output_dir, exist_ok=True)

        # Get all frame files from first LiDAR
        first_lidar_path = os.path.join(self.extracted_dirs[0], scene, "velodyne")
        frame_files = sorted([
            f for f in os.listdir(first_lidar_path)
            if f.endswith(".bin")
        ])

        for filename in frame_files:
            merged_points_list = []
            merged_labels_list = []

            lidar_stats = []

            # Loop over all LiDAR directories
            for idx, lidar_dir in enumerate(self.extracted_dirs):
                bin_path = os.path.join(lidar_dir, scene, "velodyne", filename)
                label_path = os.path.join(lidar_dir, scene, "labels", filename.replace(".bin", ".label"))

                if os.path.exists(bin_path):
                    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
                else:
                    points = np.zeros((0, 4), dtype=np.float32)

                if os.path.exists(label_path):
                    labels = np.fromfile(label_path, dtype=np.uint32)
                else:
                    labels = np.zeros(len(points), dtype=np.uint32)

                # Skip empty LiDAR data
                if len(points) > 0:
                    merged_points_list.append(points)
                    merged_labels_list.append(labels)
                    lidar_stats.append(f"L{idx}: {len(points)} pts")
                else:
                    lidar_stats.append(f"L{idx}: 0 pts (skipped)")

            # Skip if no valid LiDARs had points
            if not merged_points_list:
                print(f"No valid points found for {scene}/{filename}")
                continue

            merged_points = np.vstack(merged_points_list).astype(np.float32)
            merged_labels = np.hstack(merged_labels_list).astype(np.uint32)

            # Save merged files
            points_out_path = os.path.join(velodyne_output_dir, filename)
            labels_out_path = os.path.join(labels_output_dir, filename.replace(".bin", ".label"))
            merged_points.tofile(points_out_path)
            merged_labels.tofile(labels_out_path)

            print(f"[MERGED] {scene}/{filename} | points: {len(merged_points)}")

        # Copy extra files/folders
        if self.sequence_base_dir:
            sequence_scene_dir = os.path.join(self.sequence_base_dir, scene)
            self._copy_extras(sequence_scene_dir, scene_output_dir)

    def _copy_extras(self, sequence_scene_dir, merged_scene_dir):
        """
        Copy extra files and folders to the merged scene directory.
        """
        extras = ["calib.txt", "poses.txt", "instances.txt"]
        folders = ["cameras", "image_2"]

        for fname in extras:
            src = os.path.join(sequence_scene_dir, fname)
            dst = os.path.join(merged_scene_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)

        for folder in folders:
            src = os.path.join(sequence_scene_dir, folder)
            dst = os.path.join(merged_scene_dir, folder)
            if os.path.exists(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

