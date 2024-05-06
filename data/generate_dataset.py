import os
import argparse
import pickle
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Generate IMU dataset')
    
    parser.add_argument(
        '-f', '--dataset_folder', required=True, type=str, help='dataset directory'
    )
    parser.add_argument("-n", "--dataset_name", type=str, default='imu_dataset')
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()
    
    dataset_types = ['train', 'eval']
    for tp in dataset_types:
        target_folder = os.path.join(args.dataset_folder, tp)
        files = os.listdir(target_folder)
        x = []
        y = []
        for file in files:
            data = pickle.load(open(os.path.join(target_folder, file), 'rb'))
            
            x.append(np.array([data['imu']['4'][10:-10], data['imu']['5'][10:-10]]).transpose((1,0,2)))
            y.append(np.array([data['imu']['C'][10:-10], data['imu']['D'][10:-10]]).transpose((1,0,2)))
        
        pickle.dump(
            {'x': x, 'y': y},
            open(f'{args.dataset_name}_{tp}.pkl', 'wb')
        )
    
    