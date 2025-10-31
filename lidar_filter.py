import os
import shutil
import numpy as np

class LidarFilter:
    def __init__(self, base_input_dir, base_output_dir, filter_value):
        self.base_input_dir = base_input_dir
        self.base_output_dir = base_output_dir
        self.filter_value = filter_value

    def process_scenes(self):
        """Detect all scenes and process them."""
        scenes = sorted([
            d for d in os.listdir(self.base_input_dir)
            if os.path.isdir(os.path.join(self.base_input_dir, d))
        ])
        print(f"Detected scenes: {scenes}")

        for scene in scenes:
            print(f"\nProcessing Scene {scene}")
            
            input_scene_dir = os.path.join(self.base_input_dir, scene)
            output_scene_dir = os.path.join(self.base_output_dir, scene)
            os.makedirs(output_scene_dir, exist_ok=True)
            
            velodyne_dir = os.path.join(input_scene_dir, "velodyne")
            label_dir = os.path.join(input_scene_dir, "labels")
            output_velodyne_dir = os.path.join(output_scene_dir, "velodyne")
            output_label_dir = os.path.join(output_scene_dir, "labels")
            os.makedirs(output_velodyne_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)

            if not os.path.exists(velodyne_dir) or not os.path.exists(label_dir):
                print(f"[WARNING] Missing directory for scene {scene}")
                continue
            
            bin_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
            for filename in bin_files:
                self._process_file(scene, filename, velodyne_dir, label_dir, output_velodyne_dir, output_label_dir)
            
            # Copy extra files/folders 
            self._copy_extras(input_scene_dir, output_scene_dir)

        print("\nAll scenes processed!")

    def _process_file(self, scene, filename, velodyne_dir, label_dir, output_velodyne_dir, output_label_dir):
        bin_path = os.path.join(velodyne_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".bin", ".label"))

        try:
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            labels = np.fromfile(label_path, dtype=np.uint32)
        except Exception as e:
            print(f"[ERROR] Could not read {scene}/{filename}: {e}")
            return

        remissions = points[:, 3]
        mask = remissions == self.filter_value   
        filtered_points = points[mask]
        filtered_labels = labels[mask]

        # Save filtered outputs
        filtered_points.tofile(os.path.join(output_velodyne_dir, filename))
        filtered_labels.tofile(os.path.join(output_label_dir, filename.replace(".bin", ".label")))

        print(f"[OK] {scene}/{filename} - {len(filtered_points)} points kept (remission={self.filter_value})")

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

