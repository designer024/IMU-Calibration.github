import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from tqdm import tqdm

class IMUDataset(Dataset):
    def __init__(self, dataset_path, input_length=256, overlap=32, shift=0):
        data = pickle.load(open(dataset_path, 'rb'))
        x_raw = data['x']
        y_raw = data['y']
        
        x = []
        y = []
        for i in range(len(x_raw)):
            t = input_length
            while (t < len(x_raw[i])):
                x.append(x_raw[i][t-input_length:t])
                y.append(y_raw[i][t-input_length:t+shift])
                if shift > 0:
                    y[-1][0] = x[-1][0]
                t += overlap
        
        self.x = torch.tensor(np.array(x)).float() / 65536
        self.y = torch.tensor(np.array(y)).float() / 65536

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
