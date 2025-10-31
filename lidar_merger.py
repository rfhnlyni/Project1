import os
import numpy as np
import glob
import shutil

class LidarMerger:
    def __init__(self, lidar0_dir, lidar1_dir, output_dir):
        """
        Args:
            lidar0_dir (str): base folder to extracted intensity LIDAR0
            lidar1_dir (str): base folder to extracted intensity LIDAR1
            output_dir (str): base folder to save 
        """
        self.lidar0_dir = lidar0_dir
        self.lidar1_dir = lidar1_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def merge_all_scenes(self):
        scenes = sorted([d for d in os.listdir(self.lidar0_dir) 
                         if os.path.isdir(os.path.join(self.lidar0_dir, d))])
        print(f"Detected scenes: {scenes}")

        for scene in scenes:
            self.merge_scene(scene)

    def merge_scene(self, scene):
        print(f"\nProcessing Scene {scene}")
        
        lidar0_scene_dir = os.path.join(self.lidar0_dir, scene, "velodyne")
        lidar1_scene_dir = os.path.join(self.lidar1_dir, scene, "velodyne")
        scene_output_dir = os.path.join(self.output_dir, scene, "velodyne")
        os.makedirs(scene_output_dir, exist_ok=True)

        frame_files = sorted(glob.glob(os.path.join(lidar0_scene_dir, "*.bin")))
        frames = [os.path.splitext(os.path.basename(f))[0] for f in frame_files]

        for frame in frames:
            self.merge_frame(scene, frame, lidar0_scene_dir, lidar1_scene_dir, scene_output_dir)

    def merge_frame(self, scene, frame, lidar0_scene_dir, lidar1_scene_dir, scene_output_dir):
        lidar0_file = os.path.join(lidar0_scene_dir, f"{frame}.bin")
        lidar1_file = os.path.join(lidar1_scene_dir, f"{frame}.bin")
        output_file = os.path.join(scene_output_dir, f"{frame}.bin")

        print(f"Processing frame {frame}: ", end="")

        if not os.path.exists(lidar0_file):
            print("Missing LIDAR0 file. Skipping.")
            return

        try:
            points0 = np.fromfile(lidar0_file, dtype=np.float32).reshape(-1, 4)

            if os.path.exists(lidar1_file) and os.path.getsize(lidar1_file) > 0:
                points1 = np.fromfile(lidar1_file, dtype=np.float32).reshape(-1, 4)
                merged_points = np.vstack((points0, points1))
                print(f"MERGED - {len(points0):,} + {len(points1):,} = {len(merged_points):,} points")
            else:
                merged_points = points0
                print(f"LIDAR0 ONLY - {len(points0):,} points")

            merged_points.tofile(output_file)
        except Exception as e:
            print(f"Error: {e}")

