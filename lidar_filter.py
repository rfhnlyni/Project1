import os
import numpy as np

class LidarFilter:
    def __init__(self, base_input_dir, filter_value):
        """
        Args:
            base_input_dir (str): Directory containing all scenes
            filter_value (int/float): Intensity/remission value used to filter points
        """
        self.base_input_dir = base_input_dir
        self.filter_value = filter_value

    def process_scenes(self, return_data=True):
        """
        Process all scenes and filter points based on the filter_value

        Args:
            return_data (bool): Whether to return filtered data in memory
        Returns:
            dict: {scene: {filename: (filtered_points, filtered_labels)}} if return_data=True
        """
        # Dictionary to store filtered points and labels for all scenes
        scenes_data = {} if return_data else None

        # Get all scene folders inside base_input_dir
        scenes = sorted([
            d for d in os.listdir(self.base_input_dir)
            if os.path.isdir(os.path.join(self.base_input_dir, d))
        ])
        print(f"Detected scenes: {scenes}")

        # Process each scene
        for scene in scenes:
            print(f"\nProcessing Scene {scene}")

            velodyne_dir = os.path.join(self.base_input_dir, scene, "velodyne")
            label_dir = os.path.join(self.base_input_dir, scene, "labels")

            if not os.path.exists(velodyne_dir) or not os.path.exists(label_dir):
                print(f"[WARNING] Missing directory for scene {scene}")
                continue

            bin_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith(".bin")])
            scene_dict = {}

            for filename in bin_files:
                bin_path = os.path.join(velodyne_dir, filename)
                label_path = os.path.join(label_dir, filename.replace(".bin", ".label"))

                try:
                    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
                    labels = np.fromfile(label_path, dtype=np.uint32)
                except Exception as e:
                    print(f"[ERROR] {scene}/{filename}: {e}")
                    continue

                # Filter points by intensity/remission value
                mask = points[:, 3] == self.filter_value
                filtered_points = points[mask]
                filtered_labels = labels[mask]

                if return_data:
                    scene_dict[filename] = (filtered_points, filtered_labels)

                print(f"[FILTERED] {scene}/{filename} - {len(filtered_points)} points kept (remission={self.filter_value})")

            # Save scene dictionary to main dictionary
            if return_data:
                scenes_data[scene] = scene_dict

        return scenes_data

