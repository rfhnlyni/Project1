import os
import numpy as np

class LidarFilter:
    def __init__(self, base_input_dir, filter_value):
        """
        Args:
            base_input_dir (str): all scenes
            filter_value (int/float): value from remission/intensity use to filter points
        """
        self.base_input_dir = base_input_dir
        self.filter_value = filter_value

    def process_scenes(self, return_data=True):
	
	# Store filtered data
        scenes_data = {} if return_data else None
     
        scenes = sorted([
            d for d in os.listdir(self.base_input_dir)
            if os.path.isdir(os.path.join(self.base_input_dir, d))
        ])
        print(f"Detected scenes: {scenes}")

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
                    
		# Filter points
                mask = points[:, 3] == self.filter_value
                filtered_points = points[mask]
                filtered_labels = labels[mask]

                if return_data:
                    scene_dict[filename] = (filtered_points, filtered_labels)

                print(f"[OK] {scene}/{filename} - {len(filtered_points)} points kept (remission={self.filter_value})")

            if return_data:
                scenes_data[scene] = scene_dict

        return scenes_data

