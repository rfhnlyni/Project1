import os
import sys
import argparse
from pathlib import Path
from lidar_filter import LidarFilter
from intensity_extractor import IntensityExtractor
from lidar_merger import LidarMerger

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

class Converter:
    def __init__(self):
        self.logs = []
    
    def log(self, message):
        timestamp = self.get_timestamp()
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.logs.append(log_message)
    
    def get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def validate_paths(self, input_seq, input_pcd, output_dir):
        """Validate all paths exist"""
        if not os.path.exists(input_seq):
            raise FileNotFoundError(f"Input sequences directory not found: {input_seq}")
        
        if not os.path.exists(input_pcd):
            raise FileNotFoundError(f"PCD data directory not found: {input_pcd}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        return True
    
    def run_conversion(self, input_seq, input_pcd, output_dir, lidars, filter_values):
        """Run the complete conversion pipeline"""
        try:
            self.log("=" * 50)
            self.log("Starting PCD to BIN Conversion")
            self.log("=" * 50)
            
            # Validate paths
            self.validate_paths(input_seq, input_pcd, output_dir)
            
            self.log(f"Sequence Data: {input_seq}")
            self.log(f"PCD Data: {input_pcd}")
            self.log(f"Output Directory: {output_dir}")
            self.log(f"LiDARs: {', '.join(lidars)}")
            self.log(f"Filter Values: {', '.join(map(str, filter_values))}\n")
            
            # Create output directories
            output_extract_dir = os.path.join(output_dir, "extracted_intensity")
            output_merge_dir = os.path.join(output_dir, "merged_lidar_points")
            
            os.makedirs(output_extract_dir, exist_ok=True)
            os.makedirs(output_merge_dir, exist_ok=True)
            
            # Step 1: Filtering
            self.log("=" * 50)
            self.log("STEP 1: FILTERING LIDAR POINTS")
            self.log("=" * 50)
            
            filtered_results = {}
            
            for lidar, fval in zip(lidars, filter_values):
                self.log(f"Filtering {lidar} with filter_value={fval}")
                lidar_filter = LidarFilter(input_seq, filter_value=int(fval))
                filtered_results[lidar] = lidar_filter.process_scenes(return_data=True)
                self.log(f"Completed filtering for {lidar}\n")
            
            # Step 2: Intensity Extraction
            self.log("=" * 50)
            self.log("STEP 2: EXTRACTING INTENSITY")
            self.log("=" * 50)
            
            extractor = IntensityExtractor(pcd_base_dir=input_pcd, output_base_dir=output_extract_dir)
            
            for lidar, fval in zip(lidars, filter_values):
                lidar_output_dir = os.path.join(output_extract_dir, lidar)
                os.makedirs(lidar_output_dir, exist_ok=True)
                
                self.log(f"Extracting intensity for {lidar}")
                extractor.extract_from_memory(filtered_data=filtered_results[lidar], pcd_base_dir=input_pcd, lidar_output_dir=lidar_output_dir, filter_value=int(fval), lidar_name=lidar)
                self.log(f"Completed extraction for {lidar}\n")
            
            # Step 3: Merging
            self.log("\n" + "=" * 50)
            self.log("STEP 3: MERGING LIDAR POINTS")
            self.log("=" * 50)
            
            extracted_dirs = [os.path.join(output_extract_dir, lidar) for lidar in lidars]
            merger = LidarMerger(extracted_dirs=extracted_dirs, output_dir=output_merge_dir, sequence_base_dir=input_seq)
            merger.merge_all_scenes()
            
            self.log("=" * 50)
            self.log("CONVERSION COMPLETE!")
            self.log("=" * 50)
            self.log(f"Results saved to: {output_merge_dir}")
            self.log("Intensity values have been successfully converted from PCD to BIN!")
            
            return True
            
        except Exception as e:
            self.log(f"\nERROR during conversion: {str(e)}")
            self.log("Troubleshooting tips:")
            self.log("   - Check that all directories exist and are accessible")
            self.log("   - Verify PCD files are not corrupted")
            self.log("   - Ensure LiDAR names match your folder structure")
            self.log("   - Check file permissions")
            return False

def main():
    parser = argparse.ArgumentParser(description='PCD to BIN IntensityConverter')
    
    # Required arguments
    parser.add_argument('--input-seq', '-i', required=True,
                       help='Input sequences directory path')
    parser.add_argument('--input-pcd', '-p', required=True,
                       help='PCD data directory path')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory path')
    
    # Optional arguments with defaults
    parser.add_argument('--lidars', '-l', 
                       default="lidar_point_cloud_top_lidar,lidar_point_cloud_top_rear_lidar,lidar_point_cloud_left_lidar,lidar_point_cloud_rear_lidar,lidar_point_cloud_right_lidar,lidar_point_cloud_front_lidar",
                       help='LiDAR names (comma separated)')
    parser.add_argument('--filters', '-f', default="0,1,2,3,4,5",
                       help='Filter values (comma separated integers)')
    
    args = parser.parse_args()
    
    # Parse LiDAR names and filter values
    lidars = [name.strip() for name in args.lidars.split(',')]
    filter_values = [int(val.strip()) for val in args.filters.split(',')]
    
    # Validate counts match
    if len(lidars) != len(filter_values):
        print(f"Error: Number of LiDAR names ({len(lidars)}) must match number of filter values ({len(filter_values)})")
        sys.exit(1)
    
    # Run conversion
    converter = Converter()
    success = converter.run_conversion(
        input_seq=args.input_seq,
        input_pcd=args.input_pcd,
        output_dir=args.output,
        lidars=lidars,
        filter_values=filter_values
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()


