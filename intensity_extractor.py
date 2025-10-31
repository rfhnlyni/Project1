import os
import glob
import numpy as np
from pypcd4 import PointCloud

class IntensityExtractor:
    def __init__(self, pcd_base_dir, bin_base_dir, output_base_dir, lidars=None):
        """
        Args:
            pcd_base_dir (str): base folder for PCD files
            bin_base_dir (str): base folder for filtered BIN files
            output_base_dir (str): base folder to keep intensity extraction
            lidars (list[str]): list lidar folder names
        """
        self.pcd_base_dir = pcd_base_dir
        self.bin_base_dir = bin_base_dir
        self.output_base_dir = output_base_dir
        self.lidars = lidars if lidars is not None else ["lidar0", "lidar1"]

    def extract_all_lidars(self):
        for lidar_name in self.lidars:
            lidar_base_dir = os.path.join(self.pcd_base_dir, lidar_name)
            scenes = sorted([
                d for d in os.listdir(lidar_base_dir)
                if os.path.isdir(os.path.join(lidar_base_dir, d))
            ])
            for scene in scenes:
                pcd_dir = os.path.join(lidar_base_dir, scene)
                
                bin_dir = os.path.join(self.bin_base_dir, f"sequences_{lidar_name}", scene, "velodyne")
                
                output_dir = os.path.join(self.output_base_dir, lidar_name, scene, "velodyne")
                
                os.makedirs(output_dir, exist_ok=True)
                print(f"\nExtracting intensity for {lidar_name} / Scene {scene}")
                self.extract_scene(pcd_dir, bin_dir, output_dir)

    def extract_scene(self, pcd_dir, bin_dir, output_dir):
        """Extract intensity for all PCD/BIN in 1 scene/folder"""
        all_pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
        target_stems = []

        for full_path in all_pcd_files:
            stem = os.path.splitext(os.path.basename(full_path))[0]
            try:
                int(stem)
                target_stems.append(stem)
            except ValueError:
                continue

        print(f"Found {len(target_stems)} PCD files in {pcd_dir}")

        for stem in target_stems:
            self._process_single(stem, pcd_dir, bin_dir, output_dir)

    def _process_single(self, stem, pcd_dir, bin_dir, output_dir):
        try:
            padded_stem = f"{int(stem):03d}"
            pcd_path = os.path.join(pcd_dir, f"{stem}.pcd")
            bin_path = os.path.join(bin_dir, f"{padded_stem}.bin")
            output_path = os.path.join(output_dir, f"{padded_stem}.bin")

            if not os.path.exists(bin_path):
                print(f"  [SKIP] Missing BIN for {stem}")
                return

            pc: PointCloud = PointCloud.from_path(pcd_path)

            if 'intensity' in pc.fields:
                intensity_array = pc.numpy(('intensity',)).flatten()
            elif 'remission' in pc.fields:
                intensity_array = pc.numpy(('remission',)).flatten()
            else:
                print(f"  [WARN] Missing intensity/remission in {stem}")
                return

            normalized_intensity = intensity_array / 255.0 if len(intensity_array) > 0 else intensity_array

            bin_data = np.fromfile(bin_path, dtype=np.float32)
            if bin_data.size % 4 != 0:
                print(f"  [ERROR] BIN size invalid for {stem}")
                return
            bin_data = bin_data.reshape((-1, 4))

            if len(bin_data) != len(normalized_intensity):
                print(f"  [WARN] Point count mismatch ({len(bin_data)} vs {len(normalized_intensity)}) for {stem}")
                return

            bin_data[:, 3] = normalized_intensity
            bin_data.tofile(output_path)
            print(f"  [DONE] {stem} -> saved to {output_path}")

        except Exception as e:
            print(f"  [ERROR] {stem}: {e}")

