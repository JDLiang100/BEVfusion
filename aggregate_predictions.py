
import json
import glob
import os

def aggregate_predictions(input_dir, output_file):
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    aggregated = {
        'scores_3d': [],
        'labels_3d': [],
        'bboxes_3d': []
    }

    print(f"Aggregating {len(json_files)} files...")
    
    for fpath in json_files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                aggregated['scores_3d'].extend(data.get('scores_3d', []))
                aggregated['labels_3d'].extend(data.get('labels_3d', []))
                aggregated['bboxes_3d'].extend(data.get('bboxes_3d', []))
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    with open(output_file, 'w') as f:
        json.dump(aggregated, f)
    
    print(f"Aggregated predictions saved to {output_file}")
    print(f"Total detections: {len(aggregated['scores_3d'])}")

if __name__ == "__main__":
    # Aggregating BEVFusion nuScenes Mini run
    aggregate_predictions("outputs/bevfusion_nuscenes_mini/preds", "outputs/bevfusion_nuscenes_mini/all_predictions.json")
