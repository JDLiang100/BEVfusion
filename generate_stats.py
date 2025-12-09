
import json
import numpy as np
import os

mappings = {
    'kitti': {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'},
    'nuscenes': {
        0: 'car', 1: 'truck', 2: 'construction_vehicle', 3: 'bus', 4: 'trailer',
        5: 'barrier', 6: 'motorcycle', 7: 'bicycle', 8: 'pedestrian', 9: 'traffic_cone'
    }
}

# Define the files we want to aggregate stats for. 
# We include all models we ran.
files = {
    'kitti_pointpillars': ('kitti', 'outputs/kitti_pointpillars/000008_predictions.json'),
    'kitti_pointpillars_3class': ('kitti', 'outputs/kitti_pointpillars_3class/000008_predictions.json'),
    'nuscenes_pointpillars': ('nuscenes', 'outputs/nuscenes_pointpillars/sample.pcd_predictions.json'),
    'bevfusion': ('nuscenes', 'outputs/bevfusion_lidar_fixed/sample.pcd_predictions.json'),
    'centerpoint': ('nuscenes', 'outputs/nuscenes_centerpoint_gpu/sample.pcd_predictions.json'),
    '3dssd': ('kitti', 'outputs/kitti_3dssd_gpu/000008_predictions.json')
}

aggregated = {}

for model_key, (dataset_name, path) in files.items():
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        continue
        
    try:
        data = json.load(open(path))
        scores = np.array(data.get('scores_3d', []), dtype=float)
        labels = data.get('labels_3d', [])
        
        # Use default mapping if dataset not found, though we should be careful
        class_map = mappings.get(dataset_name, {})
        
        counts = {}
        for lab in labels:
            cls = class_map.get(lab, str(lab))
            counts[cls] = counts.get(cls, 0) + 1
            
        aggregated[model_key] = {
            'detections': len(labels),
            'mean_score': float(scores.mean()) if scores.size else None,
            'score_std': float(scores.std()) if scores.size else None,
            'max_score': float(scores.max()) if scores.size else None,
            'min_score': float(scores.min()) if scores.size else None,
            'class_counts': counts
        }
    except Exception as e:
        print(f"Error processing {model_key}: {e}")

output_path = 'outputs/inference_stats.json'
with open(output_path, 'w') as f:
    json.dump(aggregated, f, indent=2)

print(f"Stats saved to {output_path}")
