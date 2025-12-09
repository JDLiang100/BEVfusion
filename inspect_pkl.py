import pickle
import os

pkl_path = 'data/nuscenes/nuscenes_infos_train.pkl'
if not os.path.exists(pkl_path):
    print(f"{pkl_path} not found")
else:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Number of samples: {len(data['data_list'])}")
    sample = data['data_list'][0]
    if 'sweeps' in sample:
        print("Sweeps len:", len(sample['sweeps']))
        for i, s in enumerate(sample['sweeps'][:3]):
             print(f"Sweep {i} Path:", s['data_path'])
    if 'lidar_points' in sample:
        print("Main Lidar Path:", sample['lidar_points']['lidar_path'])
