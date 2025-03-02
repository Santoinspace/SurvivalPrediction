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
    min_test_loss = float('inf')

    logger = Logger(os.path.join(args.results_path, 'logs\\multisurv'))

    """data"""
    train_samples, val_samples, test_samples = split_samples(os.path.join(args.data_path, args.tabular_name), args.train_ratio, args.val_ratio, args.test_ratio)
    train_dataset = MyDataset(args.data_path, args.tabular_name, train_samples, args.intervals, args.mode, get_transforms(), args.seed)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    val_dataset = MyDataset(args.data_path, args.tabular_name, val_samples, args.intervals, args.mode, get_transforms(), args.seed)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers, drop_last=False)

    """model"""
    model = MultiSurv(args.t_dim, args.interval_num).to(args.device)

    """criterion"""
    criterion = loss_nll

    """optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1)

    """lr_scheduler"""
    def rule(epoch):
        if epoch < 20:
            lamb = 1e-4
        elif epoch < 40:
            lamb = 5e-5
        elif epoch < 60:
            lamb = 1e-5
        else:
            lamb = 1e-6
        return lamb

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)

    loss_meter = AverageMeter()

    """retrain"""
    start_epoch = 0

    if args.resume:
        load_model(f'{args.model_path}/epoch_10.pth', model)
        epoch, min_test_loss = load_training_state(f'{args.model_path}/training_state_epoch_10.pth', optimizer, lr_scheduler)
        start_epoch = epoch + 1
        print(f'Resume training from epoch {start_epoch}')

    for epoch in range(start_epoch, args.epoch_num):
        model.train()
        """train"""
        for batch_idx, (pt, ct, tabular, time, event, surv_array, idx) in enumerate(train_dataloader):
            pt = pt.to(args.device)
            ct = ct.to(args.device)
            tabular = tabular.to(args.device)
            surv_array = surv_array.to(args.device)

            pred = model(pt, ct, tabular)
            loss = criterion(surv_array, pred, args.interval_num)
            loss_meter.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        
        # print(f'Epoch: {epoch}, Loss: {loss_meter.avg}')
        logger.log('train_loss', loss_meter.avg, epoch)
        loss_meter.reset()

        """validation"""
        model.eval()
        with torch.no_grad():
            for batch_idx, (pt, ct, tabular, time, event, surv_array, idx) in enumerate(val_dataloader):
                pt = pt.to(args.device)
                ct = ct.to(args.device)
                tabular = tabular.to(args.device)
                surv_array = surv_array.to(args.device)

                pred = model(pt, ct, tabular)
                loss = criterion(surv_array, pred, args.interval_num)
                loss_meter.update(loss.item())

        # print(f'Epoch: {epoch}, valid Loss: {loss_meter.avg}')
        logger.log('valid_loss', loss_meter.avg, epoch)
        
        """save model frequently"""
        if epoch % args.epoch_save_model_interval == 0:
            save_model(model, epoch, args, is_best=False)
            save_training_state(f'{args.model_path}/training_state_epoch_{epoch}.pth', epoch, min_test_loss, optimizer, lr_scheduler)

        """save model with min valid loss"""
        if loss_meter.avg < min_test_loss:
            min_test_loss = loss_meter.avg
            save_model(model, epoch, args, is_best=True)
            save_training_state(f'{args.model_path}/training_best_state.pth', epoch, min_test_loss, optimizer, lr_scheduler)
        
        loss_meter.reset()
            
def save_model(model, epoch, args, is_best=False):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if is_best:
        model_path = f'{args.model_path}/best_model.pth'
        torch.save(model.state_dict(), model_path)
        return
    model_path = f'{args.model_path}/epoch_{epoch}.pth'
    torch.save(model.state_dict(), model_path)

def save_training_state(path, epoch, min_test_loss, optimizer, lr_scheduler):
    state = {
        'epoch': epoch,
        'min_test_loss': min_test_loss,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    torch.save(state, path)

def load_model(path, model):
    if not os.path.exists(path):
        return
    model.load_state_dict(torch.load(path))

def load_training_state(path, optimizer, lr_scheduler):
    if not os.path.exists(path):
        return
    state = torch.load(path)
    optimizer.load_state_dict(state['optimizer'])
    lr_scheduler.load_state_dict(state['lr_scheduler'])
    return state['epoch'], state['min_test_loss']

if __name__ == '__main__':
    main()