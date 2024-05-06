import os
import sys
import argparse
import numpy as np
import pickle
import torch
from time import time
from tqdm import tqdm

from utils.imu_dataset import IMUDataset
from utils.model_rnn import IMUModel
# from utils.model_transformer import IMUModel


def parse_args():
    parser = argparse.ArgumentParser(description="transformer for imu calibration task")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint", help="path to save checkpoints")
    parser.add_argument("--ckpt_name", type=str, default="epoch_best.bin", help="checkpoint name")
    parser.add_argument("--imu_data", type=str, default="data/exp_2024_0217/run_north_3.pkl",
            help="path to the imu data")
    
    # model
    parser.add_argument("--num_joints", type=int, default=2)
    parser.add_argument("--in_features", type=int, default=9)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    
    # torch
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )

    args = parser.parse_args()
    return args

def load_imu_data(dataset_path, args):
    data = pickle.load(open(dataset_path, 'rb'))
    x = np.array([data['imu']['4'][10:522], data['imu']['5'][10:522]]).transpose((1,0,2))
    y = np.array([data['imu']['C'][10:522], data['imu']['D'][10:522]]).transpose((1,0,2))
    x = torch.tensor(x).unsqueeze(0).float() / 65536
    y = torch.tensor(y).unsqueeze(0).float() / 65536
    return {'input': x, 'gt': y}

def inference(model, imu_data, args):
    with torch.no_grad():
        model.eval()
        x = imu_data['input'].to(args.device)
        pred = model(x).cpu()
    return pred

def draw_imu_curve(data, name, legend=['pred', 'gt']):
    n = data[0].size(1)
    joint = 1

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(20,10), constrained_layout=True)
    titles = ['$a_x$', '$a_y$', '$a_z$', '$\\omega_x$', '$\\omega_y$', '$\\omega_z$', '$G_x$', '$G_y$', '$G_z$']
    colors = ['C0', 'C1', 'C2']
    linestyles = ['-','-','-']
    for k in range(len(data)):
        for i in range(9):
            axes[i//3][i%3].plot([data[k][0][t][joint][i] for t in range(n)], linestyles[k], color=colors[k])
            axes[i//3][i%3].set_title(titles[i])
    
    fig.legend(legend, loc='upper right')
    
    plt.savefig(f'output/figures/{name}.png')
    # plt.show()

def main(args):
    # load data
    imu_data = load_imu_data(args.imu_data, args)
    
    # init model
    model = IMUModel(args.num_joints, args.in_features, args.embed_dim, args.hidden_size, args.num_layers,
                    args.dropout, args.bidirectional).to(args.device)
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('model size:', model_params)
    
    # load model
    chk_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    checkpoint = torch.load(chk_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_pos'])
    
    # inference
    pred = inference(model, imu_data, args)
    # draw_imu_curve([pred, imu_data['gt']], 'pred_gt')
    # draw_imu_curve([imu_data['input'], imu_data['gt']], 'raw_gt', legend=['raw', 'gt'])
    draw_imu_curve([imu_data['input'], imu_data['gt'], pred], 'all', legend=['raw', 'gt', 'pred'])


if __name__ == '__main__':
    args = parse_args()
    main(args)