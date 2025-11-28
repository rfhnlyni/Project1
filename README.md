# Extraction and Integration of LiDAR Intensity Data for SemanticKITTI Scenes

## Overview

This repository provides a tool for processing multi-LiDAR point cloud datasets to extract and integrate accurate intensity (remission) values for LiDAR sensors into the SemanticKITTI data format. The tool helps to filter point clouds based on LiDAR ID, extracts original intensity values from PCD files and replaces the remission field in BIN files, then merges the processed data following the SemanticKITTI format.

## Scripts Overview

- **`main.py`** - Main pipeline for processing all scenes with configurable LiDAR
- **`lidar_filter.py`** - Filters point clouds by LiDAR sensor identification
- **`intensity_extractor.py`** - Extracts, normalizes, and integrates intensity values from PCD files into BIN format
- **`lidar_merger.py`** - Merges multiple LiDAR sensor data into unified SemanticKITTI-compatible scenes

## Data Organization
```text
project_root/
├── data/
│   ├── sequences/
│   │   ├── 00/                        
│   │   │   ├── cameras/
│   │   │   ├── image_2/
│   │   │   ├── labels/
│   │   │   ├── velodyne/              
│   │   │   ├── calib.txt
│   │   │   ├── instances.txt
│   │   │   └── poses.txt
│   │   ├── 01/
│   │   ├── 02/
│   │   └── ...
│   │
│   └── pcd_data/
│       ├── 00/                        
│       │   ├── lidar_point_cloud_top_lidar/
│       │   │   ├── 000.pcd
│       │   │   ├── 001.pcd
│       │   │   └── ...
│       │   ├── lidar_point_cloud_top_rear_lidar/
│       │   ├── lidar_point_cloud_front_lidar/
│       │   ├── lidar_point_cloud_rear_lidar/
│       │   ├── lidar_point_cloud_left_lidar/
│       │   └── lidar_point_cloud_right_lidar/
│       ├── 01/
│       ├── 02/
│       └── ...
│
├── output/
│   ├── extracted_intensity/
│   │   ├── lidar_point_cloud_top_lidar/                        
│   │   │   ├── 00/
│   │   │   │    ├── velodyne/
│   │   │   │    └── labels/
│   │   │   ├── 01/
│   │   │   └── ...
│   │   └── ...
│   └── merged_lidar_points/
│       ├── 00/                        
│       │   ├── cameras/
│       │   ├── image_2/
│       │   ├── labels/
│       │   ├── velodyne/
│       │   ├── calib.txt
│       │   ├── instances.txt
│       │   └── poses.txt
│       ├── 01/
│       └── ...    
│
├── lidar_filter.py
├── lidar_merger.py
├── intensity_extractor.py
├── main.py
├── requirements.txt
└── README.md
```

#### Default LiDAR configuration

- lidar_point_cloud_top_lidar (Filter: 0)
- lidar_point_cloud_top_rear_lidar (Filter: 1)
- lidar_point_cloud_left_lidar (Filter: 2)
- lidar_point_cloud_rear_lidar (Filter: 3)
- lidar_point_cloud_right_lidar (Filter: 4)
- lidar_point_cloud_front_lidar (Filter: 5)

## Processing Pipeline
1. **Data Loading** - Read PCD files from LiDAR sensors and load BIN files
2. **Filtering** - Filter points by LiDAR ID to isolate data from individual sensors
3. **Intensity Extraction** - Extract and normalize intensity values from PCD files
4. **Intensity Replacement** - Integrate extracted PCD intensity values into BIN files
5. **Merging** - Merge processed point clouds from all LiDAR sensors into unified scenes
6. **Export** - Save as SemanticKITTI-compatible scenes

## Generate Intensity Extraction 

### Step To Run

#### Prerequisites 

Install the required dependencies:

```text
pip install -r requirements.txt
```

#### Overall LiDAR
```text
main.py --input-seq /path/to/sequence/dataset --input-pcd /path/to/pcd/dataset --output /path/to/output/folder
```
where:
- **`input-seq`** : path to the sequences dataset directory
- **`input-pcd`** : path to the pcd dataset directory
- **`output`** : path to the output directory

Example
```text
main.py --input-seq data/sequences --input-pcd data/pcd_data --output output
```

#### Specific LiDAR
```text
main.py --input-seq /path/to/sequence/dataset --input-pcd /path/to/pcd/dataset --output /path/to/output/folder --lidars "lidar_name" --filters lidar_ID
```

where:
- **`input-seq`** : path to the sequences dataset directory
- **`input-pcd`** : path to the pcd dataset directory
- **`output`** : path to the output directory
- **`lidars`** : name of the LiDAR sensor to process (comma-separated for multiple)
- **`filters`** : filter ID for the corresponding LiDAR sensor (comma-separated for multiple)

Example
- One specific LiDAR
```text
main.py --input-seq data/sequences --input-pcd data/pcd_data --output output --lidars "lidar_point_cloud_top_lidar" --filters 0
```
- Multiple LiDAR
```text
main.py --input-seq data/sequences --input-pcd data/pcd_data --output output --lidars "lidar_point_cloud_top_lidar,lidar_point_cloud_front_lidar" --filters "0,5"
```
