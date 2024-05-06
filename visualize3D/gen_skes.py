import sys
import os
import os.path as osp
import argparse
import pickle
import numpy as np

sys.path.insert(0, osp.dirname(osp.realpath(__file__)))
import tools.visualization as visualization
sys.path.pop(0)

def arg_parse():
    parser = argparse.ArgumentParser('Generating skeleton demo.')
    parser.add_argument('--file', type=str, default='calib.pkl', help='path to the imu data file.')
    parser.add_argument('--out', type=str, default='test.mp4', help='output file name')
    args = parser.parse_args()

    return args

def main():
    os.makedirs('./output', exist_ok=True)
    args = arg_parse()
    
    # Loading csv
    data = pickle.load(open(args.file, 'rb'))['quat']
    quats = np.array([[data['3'], data['4'], data['5'], data['2'], data['1']]])
    quats = np.transpose(quats, (0,2,1,3))[:,:]
    
    # Loading bone info
    bone_info = np.load('data/bone_info.npz', allow_pickle=True)
    
    # Generating animation
    viz_output = './output/' + args.out
    print('Generating animation...')
    visualization.drawQuaternionTraj(
        quats, '', viz_output,
        bone_info['bone_len'], bone_info['rigid_torso'], azim=-90
    )
    
    
if __name__ == "__main__":
    main()