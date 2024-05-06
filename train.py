import os
import sys
import argparse
import numpy as np
import pickle
import torch
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.nn.functional as F

from utils.imu_dataset import IMUDataset
from utils.model_rnn import IMUModel
# from utils.model_transformer import IMUModel


def parse_args():
    parser = argparse.ArgumentParser(description="transformer for imu calibration task")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint", help="path to save checkpoints")
    parser.add_argument("--dataset_dir", type=str, default="data/", help="path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, default="imu_dataset")
    
    # dataset
    parser.add_argument("--input_length", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    
    # model
    parser.add_argument("--num_joints", type=int, default=2)
    parser.add_argument("--in_features", type=int, default=9)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    
    # training
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-n", "--num_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lrd", type=float, default=0.99)
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_freq", type=int, default=40)
    parser.add_argument("--export_training_curves", action='store_true')

    args = parser.parse_args()
    return args

def train(model, dataloader, optimizer, loss_fn, args):
    h = model.init_hidden(args.batch_size)
    loss_train = 0
    N = 0
    for x, y in tqdm(dataloader):
        x = x.to(args.device)
        y = y.to(args.device)
        optimizer.zero_grad()
        
        # train
        if y.size(0) != args.batch_size:
            h = model.init_hidden(y.size(0))
        pred = model(x, h)
        
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        # metrics
        loss_train += loss.item()
        N += 1
    
    return loss_train / N

def evaluate(model_train_dict, model_eval, dataloader, loss_fn, args):
    loss_eval = 0
    N = 0
    with torch.no_grad():
        model_eval.load_state_dict(model_train_dict)
        model_eval.eval()
        for x, y in tqdm(dataloader):
            x = x.to(args.device)
            y = y.to(args.device)
            
            # train
            pred = model_eval(x)
            
            # metrics
            loss = loss_fn(pred, y)
            loss_eval += loss.item()
            N += 1
    
    return loss_eval / N


def main(args):
    # set random seed
    seed = 5731
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # setup dataset
    dataset_train = IMUDataset(
        os.path.join(args.dataset_dir, f'{args.dataset_name}_train.pkl'),
        input_length=args.input_length, overlap=args.overlap
    )
    dataloader_train = DataLoader(dataset_train, args.batch_size, shuffle=True)
    dataset_eval = IMUDataset(
        os.path.join(args.dataset_dir, f'{args.dataset_name}_eval.pkl'),
        input_length=args.input_length, overlap=args.overlap
    )
    dataloader_eval = DataLoader(dataset_eval, args.batch_size, shuffle=True)
    print('Preparing training data...')
    print('Training for', len(dataset_train), 'sequences')
    print('Evaluating for', len(dataset_eval), 'sequences')
    
    # init model
    model = IMUModel(args.num_joints, args.in_features, args.embed_dim, args.hidden_size, args.num_layers,
                    args.dropout, args.bidirectional, args.input_length).to(args.device)
    model_eval = IMUModel(args.num_joints, args.in_features, args.embed_dim, args.hidden_size, args.num_layers,
                    args.dropout, args.bidirectional, args.input_length).to(args.device)
    
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('model size:', model_params)

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # init loss function
    # loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    loss_fn = nn.L1Loss()

    # resume model
    start_epoch = 0
    lr = args.lr
    loss_history = []
    if args.resume:
        chk_path = os.path.join(args.ckpt_dir, args.resume)
        checkpoint = torch.load(chk_path, map_location=lambda storage, loc: storage)
        
        model.load_state_dict(checkpoint['model_pos'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        loss_history = pickle.load(open(os.path.join(args.ckpt_dir, 'loss.pkl'), 'rb'))
    
    # training
    loss_best = 1e9
    scheduler = ExponentialLR(optimizer, args.lrd)
    for epoch in range(start_epoch, args.num_epoch):
        start_time = time()
        model.train()
        
        loss_train = train(model, dataloader_train, optimizer, loss_fn, args)
        loss_eval = evaluate(model.state_dict(), model_eval, dataloader_eval, loss_fn, args)
        loss_history.append([loss_train, loss_eval])
        
        # saving data
        elapsed = (time() - start_time) / 60
        print('[%d] time %.2f lr %f train %f eval %f' % (
            epoch + 1, elapsed, lr, loss_train, loss_eval
        ))
        # if (epoch+1) % args.save_freq == 0:
            # chk_path = os.path.join(args.ckpt_dir, 'epoch_{}.bin'.format(epoch+1))
            # print('Saving checkpoint to', chk_path)
            # torch.save({
                # 'epoch': epoch+1,
                # 'lr': lr,
                # 'optimizer': optimizer.state_dict(),
                # 'model_pos': model.state_dict()
            # }, chk_path)
        
        if loss_eval < loss_best:
            loss_best = loss_eval
            chk_path = os.path.join(args.ckpt_dir, 'epoch_best.bin')
            print('Saving best checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model.state_dict()
            }, chk_path)
        
        # chk_path = os.path.join(args.ckpt_dir, 'final.bin')
        # torch.save({
            # 'epoch': epoch+1,
            # 'lr': lr,
            # 'optimizer': optimizer.state_dict(),
            # 'model_pos': model.state_dict()
        # }, chk_path)
        pickle.dump(loss_history, open(os.path.join(args.ckpt_dir, 'loss.pkl'), 'wb'))
        
        # update params
        if optimizer.param_groups[0]['lr'] >= 1e-6:
            scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        
        # plot loss curve
        if args.export_training_curves:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(len(loss_history)) + 1
            plt.plot(epoch_x, np.array(loss_history)[:,0], '--', color='C0')
            plt.plot(epoch_x, np.array(loss_history)[:,1], '--', color='C1')
            plt.legend(['train', 'eval'])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig(os.path.join(args.ckpt_dir, 'acc.png'))
            plt.close('all')


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)