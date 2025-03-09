from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
import pandas as pd
import nibabel as nib
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

from config.config import get_args
from dataloader.dataset import MyDataset, get_transforms
from models.model_1 import MultiModalFusionModel as Model
# from models.model_multisurv import MultiSurv as Model
from utils.loss import loss_nll
from utils.logging import AverageMeter, Logger
from utils.metrics import get_brier_score, calculate_time

"""save single fold model"""
def save_model_single(model, epoch, args, is_best=False):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if is_best:
        model_path = f'{args.model_path}/best_model.pth'
        torch.save(model.state_dict(), model_path)
        return
    model_path = f'{args.model_path}/epoch_{epoch}.pth'
    torch.save(model.state_dict(), model_path)

"""save multi fold model"""
def save_model(model, k, epoch, args, is_best=False):
    # if not os.path.exists(args.model_path):
    #     os.makedirs(args.model_path)
    if not os.path.exists(f'{args.model_path}/{k}_fold'):
        os.makedirs(f'{args.model_path}/{k}_fold')
    if is_best:
        model_path = f'{args.model_path}/{k}_fold/best_model.pth'
        torch.save(model.state_dict(), model_path)
        return
    model_path = f'{args.model_path}/{k}_fold/epoch_{epoch}.pth'
    torch.save(model.state_dict(), model_path)

def save_training_state(path, epoch, min_test_loss, optimizer, lr_scheduler):
    state = {
        'epoch': epoch,
        'min_test_loss': min_test_loss,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    torch.save(state, path)

"""load pretrained single model"""
def load_pretrained_model(path, model):
    if not os.path.exists(path):
        print('Pretrained model not found.')
        return
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Pretrained model loaded.')

def load_training_state(path, optimizer, lr_scheduler):
    if not os.path.exists(path):
        return
    state = torch.load(path)
    optimizer.load_state_dict(state['optimizer'])
    lr_scheduler.load_state_dict(state['lr_scheduler'])
    return state['epoch'], state['min_test_loss']

def load_network(args, fold_k):
    """model"""
    # model = Model(args.t_dim, args.interval_num).to(args.device)
    model = Model(pet_in_channels=1, ct_in_channels=1, tabular_dim=args.t_dim,
                                  feature_dim=512, num_heads=8, transformer_layers=1,
                                  interval_num=args.interval_num).to(args.device)

    # TODO
    # """pretrained model"""

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

    return model, criterion, optimizer, lr_scheduler

def train_one_epoch(args, fold,  epoch, summary_writer, data_loader, model, criterion, optimizer, lr_scheduler):
    model.train()
    loss_meter = AverageMeter()
    for batch_idx, (pt, ct, tabular, time, event, surv_array, idx) in enumerate(data_loader):
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

    summary_writer.log('train_loss', loss_meter.avg, epoch)
    loss_meter.reset()

    return model, optimizer, lr_scheduler

def eval_one_epoch(args, fold, epoch, summary_writer, data_loader, model, criterion, optimizer, lr_scheduler):
    model.eval()
    loss_meter = AverageMeter()

    time_list = []
    event_list = []
    time_pred_list = []
    brier_score_list = []
    surv_pred_list = []
    for batch_idx, (pt, ct, tabular, time, event, surv_array, idx) in enumerate(data_loader):
        pt = pt.to(args.device)
        ct = ct.to(args.device)
        tabular = tabular.to(args.device)
        surv_array = surv_array.to(args.device)
        time = np.array(time)
        event = np.array(event)

        time_list.append(time)
        event_list.append(event)

        with torch.no_grad():
            pred = model(pt, ct, tabular)

        loss = criterion(surv_array, pred, args.interval_num)
        loss_meter.update(loss.item())

        """cumulate metrics"""
        surv_pred_list.append(pred.cpu().numpy())
        time_pred_list.append(calculate_time(pred.cpu(), args.intervals))
        
        brier_score = get_brier_score(pred.cpu(), surv_array.cpu(), time, event, args.intervals)
        if brier_score > 0:
            brier_score_list.append(brier_score)
        
    """calculate metrics"""
    label_time = np.array(time_list).squeeze()
    label_event = np.array(event_list).squeeze()
    scores = np.array(time_pred_list).squeeze()
    surv_preds = np.array(surv_pred_list).squeeze()
    ci = concordance_index(label_time, scores, label_event)
    bs = np.array(brier_score_list).mean()

    summary_writer.log('valid_loss', loss_meter.avg, epoch)
    summary_writer.log('valid_ci', ci, epoch)
    summary_writer.log('valid_bs', bs, epoch)

    loss_meter.reset()

    return ci, bs, label_time, label_event, scores, surv_preds

def main():
    """config"""
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    min_test_loss = float('inf')

    tabular_data = pd.read_csv(os.path.join(args.data_path, args.tabular_name), encoding='utf-8').iloc[:]
    samples = tabular_data['xing_ming'].tolist()
    events = tabular_data['censorship'].tolist()

    """average Ci across all folds"""
    ci_meter_total_average = AverageMeter()

    """total ci across all predictions for all folds"""
    all_label_time = []
    all_label_event = []
    all_scores = []

    """dataset split into k folds"""
    kf = StratifiedKFold(n_splits=args.fold_num, shuffle=True, random_state=args.seed)
    k = -1
    for train_index, eval_index in kf.split(samples, events):
        k += 1
        """data"""
        train_samples = [samples[i] for i in train_index]
        val_samples = [samples[i] for i in eval_index]
        train_dataset = MyDataset(args.data_path, args.tabular_name, train_samples, args.intervals, args.mode, get_transforms(), args.seed)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        val_dataset = MyDataset(args.data_path, args.tabular_name, val_samples, args.intervals, args.mode, get_transforms(), args.seed)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers, drop_last=False)
        
        print(f'Fold {k}')
        print(f'train:val={len(train_samples)}:{len(val_samples)}')

        """model"""
        model, criterion, optimizer, lr_scheduler = load_network(args=args,fold_k=k)
        total_comsuption = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        # 精确地计算：1MB=1024KB=1048576字节
        print(f'model name:{args.model_name}    Number of parameter: {(total_comsuption / 1024 / 1024):.4f}M')

        """train & eval"""
        start_epoch = 0
        ci_meter = AverageMeter()
        for epoch in range(start_epoch, args.epoch_num):
            train_logger = Logger(os.path.join(args.results_path, f'{args.model_name}\\logs\\train\\fold_{k}'))
            val_logger = Logger(os.path.join(args.results_path, f'{args.model_name}\\logs\\eval\\fold_{k}'))
            model, optimizer, lr_scheduler = train_one_epoch(
                args=args,
                fold=k,
                epoch=epoch,
                summary_writer=train_logger,
                data_loader=train_dataloader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )
            ci, bs, label_time, label_event, scores, surv_preds = eval_one_epoch(
                args=args,
                fold=k,
                epoch=epoch,
                summary_writer=val_logger,
                data_loader=val_dataloader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )
            # 新增数据收集
            all_label_time.extend(label_time)
            all_label_event.extend(label_event) 
            all_scores.extend(scores)
            
            """save model frequently"""
            if epoch % args.epoch_save_model_interval == 0 and epoch != 0:
                save_model(model, k, epoch, args, is_best=False)
                save_training_state(f'{args.model_path}/{k}_fold/training_state_epoch_{epoch}.pth', epoch, min_test_loss, optimizer, lr_scheduler)

            """save model with max valid ci"""
            if ci_meter.max < ci:
                save_model(model, k, epoch, args, is_best=True)
                save_training_state(f'{args.model_path}/{k}_fold/training_best_state.pth', epoch, min_test_loss, optimizer, lr_scheduler)

            ci_meter.update(ci)
        
        ci_meter_total_average.update(ci_meter.max)
        print(f'Fold {k} Ci: {ci_meter.max}')
        ci_meter.reset()

    # 输出所有fold的ci最大值的平均值
    print(f'all folds average Ci: {ci_meter_total_average.avg}')
    # 在fold循环外计算用全部数据及其预测值计算的CI
    overall_ci = concordance_index(np.array(all_label_time), 
                                  np.array(all_scores),
                                  np.array(all_label_event))
    print(f'Overall CI: {overall_ci:.4f}')

if __name__ == '__main__':
    main()