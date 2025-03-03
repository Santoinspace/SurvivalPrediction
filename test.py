import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np

from config.config import get_args
from dataloader.dataset import MyDataset, split_samples, get_transforms
from models.model_multisurv import MultiSurv
from utils.loss import loss_nll
from utils.logging import AverageMeter, Logger


def main():
    """config"""
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.test_min_loss_model_path_reg = f'{args.model_path}/test_min_loss_epoch_*.pth'

    """data"""
    train_samples, val_samples, test_samples = split_samples(os.path.join(args.data_path, args.tabular_name), args.train_ratio, args.val_ratio, args.test_ratio)
    test_dataset = MyDataset(args.data_path, args.tabular_name, test_samples, args.intervals, args.mode, get_transforms(), args.seed)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=True, num_workers=args.num_workers, drop_last=True)

    """model"""
    model = MultiSurv(args.t_dim, args.interval_num).to(args.device)
    load_model(r'D:\AProjection\SurvivalPrediction\results\checkpoints\multisurv_new\best_model.pth', model)

    """criterion"""
    criterion = loss_nll

    loss_meter = AverageMeter()

    model.eval()
    """train"""
    for batch_idx, (pt, ct, tabular, time, event, surv_array, idx) in enumerate(test_dataloader):
        pt = pt.to(args.device)
        ct = ct.to(args.device)
        tabular = tabular.to(args.device)
        surv_array = surv_array.to(args.device)

        pred = model(pt, ct, tabular)
        loss = criterion(surv_array, pred, args.interval_num)
        loss_meter.update(loss.item())
        
        print(f'test Loss: {loss_meter.avg}')
        loss_meter.reset()           

def load_model(path, model):
    if not os.path.exists(path):
        return
    model.load_state_dict(torch.load(path))

if __name__ == '__main__':
    main()