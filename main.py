import os
import sys
import argparse  # CLI
from datetime import datetime

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import local modules
try:
    from lidar_filter import LidarFilter
    from intensity_extractor import IntensityExtractor
    from lidar_merger import LidarMerger
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure lidar_filter.py, intensity_extractor.py, and lidar_merger.py are in the same directory.")
    sys.exit(1)

class Converter:
    def __init__(self):
        """Initialize the Converter"""
        self.logs = []
        
        # Define expected lidar-filter mapping
        self.EXPECTED_MAPPING = {
            "lidar_point_cloud_top_lidar": 0,
            "lidar_point_cloud_top_rear_lidar": 1,
            "lidar_point_cloud_left_lidar": 2,
            "lidar_point_cloud_rear_lidar": 3,
            "lidar_point_cloud_right_lidar": 4,
            "lidar_point_cloud_front_lidar": 5
        }
    
    def log(self, message):
        """Log a message with timestamp"""
        timestamp = self.get_timestamp()
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.logs.append(log_message)
    
    def get_timestamp(self):
        """Get current timestamp"""
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
    
    def validate_lidar_filter_pairs(self, lidars, filter_values):
        """Validate that lidar names are paired with their expected filter values"""
        errors = []
        
        for lidar, filter_val in zip(lidars, filter_values):
            expected_filter = self.EXPECTED_MAPPING.get(lidar)
            
            if expected_filter is None:
                errors.append(f"Unknown LiDAR name: '{lidar}'. Expected one of: {', '.join(self.EXPECTED_MAPPING.keys())}")
            elif expected_filter != filter_val:
                errors.append(f"LiDAR '{lidar}' should use filter value {expected_filter}, but got {filter_val}")
        
        return errors
    
    def run_conversion(self, input_seq, input_pcd, output_dir, lidars, filter_values, progress_callback=None):
        """Run the complete conversion pipeline - CLI/GUI"""
        try:
            if progress_callback:
                progress_callback("Starting conversion...", 0)
                
            self.log("=" * 50)
            self.log("Starting PCD to BIN Conversion")
            self.log("=" * 50)
            
            # Validate paths
            if progress_callback:
                progress_callback("Validating paths...", 5)
                
            self.validate_paths(input_seq, input_pcd, output_dir)
            
            # Validate lidar-filter pairs
            if progress_callback:
                progress_callback("Validating LiDAR configuration...", 10)
            validation_errors = self.validate_lidar_filter_pairs(lidars, filter_values)
            if validation_errors:
                self.log("VALIDATION ERRORS:")
                for error in validation_errors:
                    self.log(f"  {error}")
                self.log("\nExpected LiDAR-Filter mapping:")
                for lidar, expected_filter in self.EXPECTED_MAPPING.items():
                    self.log(f"  {lidar} -> Filter: {expected_filter}")
                raise ValueError("Invalid LiDAR-filter pairs provided")
            
            self.log(f"Sequence Data: {input_seq}")
            self.log(f"PCD Data: {input_pcd}")
            self.log(f"Output Directory: {output_dir}")
            self.log(f"LiDARs: {', '.join(lidars)}")
            self.log(f"Filter Values: {', '.join(map(str, filter_values))}\n")
            
            # Create output directories
            output_extract_dir = os.path.join(output_dir, "extracted_intensity")
            output_merge_dir = os.path.join(output_dir, "converted_data")
            
            os.makedirs(output_extract_dir, exist_ok=True)
            os.makedirs(output_merge_dir, exist_ok=True)
            
            # Step 1: Filtering
            if progress_callback:
                progress_callback("Filtering LiDAR point clouds...", 15)
                
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
            if progress_callback:
                progress_callback("Extracting intensity values...", 55)
                
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
            if progress_callback:
                progress_callback("Merging LiDAR outputs...", 75)
                
            self.log("=" * 50)
            self.log("STEP 3: MERGING LIDAR POINTS")
            self.log("=" * 50)
            
            self.log(f"Starting merging process")
            extracted_dirs = [os.path.join(output_extract_dir, lidar) for lidar in lidars]
            merger = LidarMerger(extracted_dirs=extracted_dirs, output_dir=output_merge_dir, sequence_base_dir=input_seq)
            merger.merge_all_scenes()
            
            if progress_callback:
                progress_callback("Finalizing conversion...", 95)
                
            self.log(f"Completed merging\n")
            
            self.log("=" * 50)
            self.log("CONVERSION COMPLETE!")
            self.log("=" * 50)
            self.log(f"Results saved to: {output_merge_dir}")
            self.log("Intensity values have been successfully converted from PCD to BIN!")
            
            if progress_callback:
                progress_callback("Conversion complete!", 100)
            
            return True
            
        except Exception as e:
            self.log(f"\nERROR during conversion: {str(e)}")
            self.log("Troubleshooting tips:")
            self.log("   - Check that all directories exist and are accessible")
            self.log("   - Verify PCD files are not corrupted")
            self.log("   - Ensure LiDAR names match your folder structure")
            self.log("   - Check file permissions")
            
            if progress_callback:
                progress_callback(f"Error: {str(e)}", 100)
                
            return False

# CLI CODE
def main():
    """Command-line interface entry point"""
    parser = argparse.ArgumentParser(
        description='PCD to BIN IntensityConverter',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
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
                       help='Filter values (comma separated)')
    
    args = parser.parse_args()
    
    # Parse LiDAR names and filter values
    lidars = [name.strip() for name in args.lidars.split(',')]
    filter_values = [int(val.strip()) for val in args.filters.split(',')]
    
    # Validate counts match
    if len(lidars) != len(filter_values):
        print(f"Error: Number of LiDAR names ({len(lidars)}) must match number of filter values ({len(filter_values)})")
        sys.exit(1)
    
    # Run conversion (CLI doesn't use progress_callback)
    converter = Converter()
    success = converter.run_conversion(input_seq=args.input_seq, input_pcd=args.input_pcd, output_dir=args.output, lidars=lidars, filter_values=filter_values, progress_callback=None)
    
    # Exit
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
