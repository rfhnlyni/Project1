import os
import numpy as np
import glob
import shutil


class LidarMerger:
    def __init__(self, lidar0_dir, lidar1_dir, output_dir):
        """
        Args:
            lidar0_dir (str): base folder for extracted intensity LIDAR0
            lidar1_dir (str): base folder for extracted intensity LIDAR1
            output_dir (str): base folder to save merged results
        """
        self.lidar0_dir = lidar0_dir
        self.lidar1_dir = lidar1_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def merge_all_scenes(self):
        scenes = sorted([
            d for d in os.listdir(self.lidar0_dir)
            if os.path.isdir(os.path.join(self.lidar0_dir, d))
        ])
        print(f"Detected scenes: {scenes}")

        for scene in scenes:
            self.merge_scene(scene)

    def merge_scene(self, scene):
        print(f"\nProcessing Scene {scene}")

        lidar0_scene_dir = os.path.join(self.lidar0_dir, scene, "velodyne")
        lidar1_scene_dir = os.path.join(self.lidar1_dir, scene, "velodyne")
        labels0_dir = os.path.join(self.lidar0_dir, scene, "labels")
        labels1_dir = os.path.join(self.lidar1_dir, scene, "labels")

        velodyne_output_dir = os.path.join(self.output_dir, scene, "velodyne")
        labels_output_dir = os.path.join(self.output_dir, scene, "labels")
        os.makedirs(velodyne_output_dir, exist_ok=True)
        os.makedirs(labels_output_dir, exist_ok=True)

        frame_files = sorted(glob.glob(os.path.join(lidar0_scene_dir, "*.bin")))
        frames = [os.path.splitext(os.path.basename(f))[0] for f in frame_files]

        for frame in frames:
            self.merge_frame(
                scene, frame,
                lidar0_scene_dir, lidar1_scene_dir,
                labels0_dir, labels1_dir,
                velodyne_output_dir, labels_output_dir
            )

        # Copy extra files
        self._copy_extras(os.path.join(self.lidar0_dir, scene), os.path.join(self.output_dir, scene))

    def merge_frame(self, scene, frame,
                    lidar0_scene_dir, lidar1_scene_dir,
                    labels0_dir, labels1_dir,
                    vel_output_dir, labels_output_dir):

        lidar0_file = os.path.join(lidar0_scene_dir, f"{frame}.bin")
        lidar1_file = os.path.join(lidar1_scene_dir, f"{frame}.bin")
        output_file = os.path.join(vel_output_dir, f"{frame}.bin")

        label0_file = os.path.join(labels0_dir, f"{frame}.label")
        label1_file = os.path.join(labels1_dir, f"{frame}.label")
        output_label_file = os.path.join(labels_output_dir, f"{frame}.label")

        print(f"Processing frame {frame}: ", end="")

        if not os.path.exists(lidar0_file):
            print("Missing LIDAR0 file. Skipping.")
            return

        try:
            # Merge points
            points0 = np.fromfile(lidar0_file, dtype=np.float32).reshape(-1, 4)

            if os.path.exists(lidar1_file) and os.path.getsize(lidar1_file) > 0:
                points1 = np.fromfile(lidar1_file, dtype=np.float32).reshape(-1, 4)
                merged_points = np.vstack((points0, points1))
                print(f"MERGED - {len(points0):,} + {len(points1):,} = {len(merged_points):,} points")
            else:
                points1 = np.zeros((0, 4), dtype=np.float32)
                merged_points = points0
                print(f"LIDAR0 ONLY - {len(points0):,} points")

            merged_points.tofile(output_file)

            # Merge labels
            labels0 = np.fromfile(label0_file, dtype=np.uint32) if os.path.exists(label0_file) else np.zeros(len(points0), dtype=np.uint32)
            if os.path.exists(label1_file) and os.path.getsize(label1_file) > 0:
                labels1 = np.fromfile(label1_file, dtype=np.uint32)
            else:
                labels1 = np.zeros(len(points1), dtype=np.uint32)

            merged_labels = np.hstack((labels0, labels1))
            merged_labels.tofile(output_label_file)

        except Exception as e:
            print(f"Error: {e}")

    def _copy_extras(self, input_scene_dir, output_scene_dir):
        extras = ["calib.txt", "poses.txt"]
        folders = ["cameras", "image_2"]

        # Copy text files
        for fname in extras:
            src = os.path.join(input_scene_dir, fname)
            dst = os.path.join(output_scene_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)

        # Copy folders
        for folder in folders:
            src = os.path.join(input_scene_dir, folder)
            dst = os.path.join(output_scene_dir, folder)
            if os.path.exists(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

