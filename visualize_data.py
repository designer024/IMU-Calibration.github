import os
import sys
import numpy as np
import pickle
import cv2
from tqdm import tqdm


def draw_quaternion(quat_C, quat_D, out_name='test'):
    from utils.quaternion_utils import quat_to_vec, plotVec
    
    res = []
    for i in range(len(quat_C)):
        res.append([
            [0,0,0],
            quat_to_vec(quat_C[i], np.array([0,0,-1])),
            quat_to_vec(quat_D[i], np.array([0,0,-1]))
        ])
    res = np.array(res)
    
    print("Drawing vectors...")
    frames = []
    for i in tqdm(range(len(res))):
        frame = plotVec(res[i], show=False)[:,:,:3]
        frame = cv2.cvtColor(frame.astype(np.float32)/255, cv2.COLOR_RGB2BGR) * 255
        frames.append(frame.astype(np.int8))
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(f'output/{out_name}.mp4', fourcc, 15, (width, height))
    print("Writing to video...")
    for i in range(len(frames)):
        video.write(frames[i])
    
    cv2.destroyAllWindows()
    video.release()

def draw_imu_curve(imu, sensor_names=['1','2','3','4','5','C','D']):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 3, figsize=(20,10), constrained_layout=True)
    
    titles = ['$a_x$', '$a_y$', '$a_z$', '$\\omega_x$', '$\\omega_y$', '$\\omega_z$', '$G_x$', '$G_y$', '$G_z$']
    for k in imu.keys():
        if k not in sensor_names:
            continue
        n = len(imu[k])
        for i in range(9):
            axes[i//3][i%3].plot([imu[k][t][i] for t in range(n)])
    
    for i in range(9):
        axes[i//3][i%3].set_title(titles[i])
    
    sensor_name_legend = [name for name in list(imu.keys()) if name in sensor_names]
    fig.legend(sensor_name_legend, loc='upper right')
    
    # plt.savefig('output/figures/a.png')
    plt.show()


if __name__ == '__main__':
    data_folder = 'data'
    
    files = [f for f in sorted(os.listdir(data_folder)) if f.endswith('pkl') and f.startswith('2024')]
    idx = -1
    print(files[idx])

    '''
    Check quaternion data
    '''
    # data = pickle.load(open(os.path.join(data_folder, files[idx]), 'rb'))
    # quat_C = data['quat']['C'] # lower arm
    # quat_D = data['quat']['D'] # upper arm
    # draw_quaternion(quat_C, quat_D, out_name='test')
    
    # data = pickle.load(open(os.path.join(data_folder, files[idx]), 'rb'))
    # quat_C = data['quat']['1'] # lower arm
    # quat_D = data['quat']['2'] # upper arm
    # draw_quaternion(quat_C, quat_D, out_name='test')
    
    '''
    Check imu data
    '''
    data = pickle.load(open(os.path.join(data_folder, files[idx]), 'rb'))
    draw_imu_curve(data['imu'], ['C', '4'])
    
    
    
    
    # for file in files:
        # data = pickle.load(open(os.path.join(data_folder, file), 'rb'))
        # imu_d = np.array(data['imu']['D'])
        # imu_4 = np.array(data['imu']['5'])
        
        # print(
            # file,
            # np.mean(imu_d[:,6] - imu_4[:,6]),
            # np.mean(imu_d[:,7] - imu_4[:,7]),
            # np.mean(imu_d[:,8] - imu_4[:,8])
        # )