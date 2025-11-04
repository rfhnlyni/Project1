import os
import numpy as np
import shutil

class LidarMerger:
    def __init__(self, extracted_dirs: list, output_dir, sequence_base_dir=None):
        """
        Args:
            extracted_dir_lidar0 (str): extracted intensity LIDAR0
            extracted_dir_lidar1 (str): extracted intensity LIDAR1
            output_dir (str): store after merge
            sequence_base_dir (str): copy extra files/folders
        """
        self.extracted_dirs = extracted_dirs
        self.output_dir = output_dir
        self.sequence_base_dir = sequence_base_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def merge_all_scenes(self):
        scenes = sorted([s for s in os.listdir(self.extracted_dirs[0])
                         if os.path.isdir(os.path.join(self.extracted_dirs[0], s))])
        print(f"Detected scenes: {scenes}")

        for scene in scenes:
            self.merge_scene(scene)

    def merge_scene(self, scene):
        print(f"\nProcessing Scene {scene}")
        scene_output_dir = os.path.join(self.output_dir, scene)
        velodyne_output_dir = os.path.join(scene_output_dir, "velodyne")
        labels_output_dir = os.path.join(scene_output_dir, "labels")
        os.makedirs(velodyne_output_dir, exist_ok=True)
        os.makedirs(labels_output_dir, exist_ok=True)

        # Get frame files from first lidar
        frame_files = sorted(os.listdir(os.path.join(self.extracted_dirs[0], scene, "velodyne")))
        frame_files = [f for f in frame_files if f.endswith(".bin")]

        for filename in frame_files:
            merged_points_list = []
            merged_labels_list = []

            for lidar_dir in self.extracted_dirs:
                bin_path = os.path.join(lidar_dir, scene, "velodyne", filename)
                label_path = os.path.join(lidar_dir, scene, "labels", filename.replace(".bin", ".label"))

                points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4) if os.path.exists(bin_path) else np.zeros((0,4))
                labels = np.fromfile(label_path, dtype=np.uint32) if os.path.exists(label_path) else np.zeros(len(points), dtype=np.uint32)

                merged_points_list.append(points)
                merged_labels_list.append(labels)

            # Merge all points and labels
            merged_points = np.vstack(merged_points_list)
            merged_labels = np.hstack(merged_labels_list)

            # Save merged points and labels
            merged_points.tofile(os.path.join(velodyne_output_dir, filename))
            merged_labels.tofile(os.path.join(labels_output_dir, filename.replace(".bin", ".label")))
            print(f"[MERGED] {scene}/{filename} | points: {len(merged_points)}")

        # Copy extra files/folders
        if self.sequence_base_dir:
            sequence_scene_dir = os.path.join(self.sequence_base_dir, scene)
            self._copy_extras(sequence_scene_dir, scene_output_dir)

    def _copy_extras(self, sequence_scene_dir, merged_scene_dir):
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

