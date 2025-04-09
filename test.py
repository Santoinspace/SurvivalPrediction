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
import math

from config.config import get_args
from dataloader.dataset import MyDataset, get_transforms
from utils.loss import loss_nll
from utils.my_logging import AverageMeter, Logger
from utils.metrics import get_brier_score, calculate_time
from utils.util import setup_seed

# from models.model_1 import MultiModalFusionModel as Model
from models.model_multisurv import MultiSurv as Model
# from models.model_tmss_SurvPath import TMSS as Model

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
    # model = Model(pet_in_channels=1, ct_in_channels=1, tabular_dim=args.t_dim,
    #                               feature_dim=512, num_heads=8, transformer_layers=1,
    #                               interval_num=args.interval_num).to(args.device)

    # tmss_survpath
    model = Model(args.t_dim, args.interval_num).to(args.device)

    """load pretrained model"""
    if args.model_path:
        load_pretrained_model(os.path.join(args.model_path,str(fold_k) + '_fold','best_model.pth'), model)
    
    return model

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

    summary_writer.log('valid_ci', ci, epoch)
    summary_writer.log('valid_bs', bs, epoch)

    loss_meter.reset()

    return ci, bs, label_time, label_event, scores, surv_preds

def main():
    """config"""
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)

    """data"""
    tabular_data = pd.read_csv(os.path.join(args.data_path, args.tabular_name), encoding='utf-8').iloc[:]
    samples = tabular_data['xing_ming'].tolist()
    events = tabular_data['censorship'].tolist()

    """dataset split into k folds"""
    kf = StratifiedKFold(n_splits=args.fold_num, shuffle=True, random_state=args.seed)

    k = -1
    ci_of_all_folds = []
    bs_of_all_folds = []
    p_value_of_all_folds = []
    aucs_of_all_folds = [[], [], []]
    label_time_total = np.empty(0)
    label_event_total = np.empty(0)
    scores_total = np.empty(0)
    surv_preds_total = np.empty([0, 4])

    """logger"""
    val_logger = Logger(os.path.join(args.log_path, f'test\\fold_{k}'))

    for _, eval_index in kf.split(samples, events):
        k += 1
        """data"""
        val_samples = [samples[i] for i in eval_index]
        val_dataset = MyDataset(args.data_path, args.tabular_name, val_samples, args.intervals, args.mode, get_transforms(), args.seed)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers, drop_last=False)
        
        print(f'Fold {k}')
        print(f'val nums={len(val_samples)}')

        """model"""
        model = load_network(args=args,fold_k=k)

        """Test"""
        ci, bs, label_time, label_event, scores, surv_preds = eval_one_epoch(
            args=args,
            fold=k,
            epoch=0,
            summary_writer=val_logger,
            data_loader=val_dataloader,
            model=model,
            criterion=None,
            optimizer=None,
            lr_scheduler=None
        )

        ci_of_all_folds.append(ci)
        bs_of_all_folds.append(bs)
        label_time_total = np.concatenate((label_time_total, label_time))
        label_event_total = np.concatenate((label_event_total, label_event))
        scores_total = np.concatenate((scores_total, scores))
        surv_preds_total = np.concatenate((surv_preds_total, surv_preds))

    '''所有fold的平均ci和bs''' 
    ci_mean = np.mean(ci_of_all_folds)
    bs_mean = np.mean(bs_of_all_folds)
    ci_std = np.std(ci_of_all_folds)
    bs_std = np.std(bs_of_all_folds)

    print(f'avg_ci={ci_mean:.4f}±{ci_std:.4f}, avg_bs={bs_mean:.4f}±{bs_std:.4f}')

    '''所有预测的ci和bs'''
    ci_all = concordance_index(label_time_total, scores_total, label_event_total)
    # bs_all = get_brier_score(torch.from_numpy(surv_preds_total), torch.from_numpy(label_event_total), label_time_total, label_event_total, args.intervals)
    # print(f'ci_all={ci_all:.4f}, bs_all={bs_all:.4f}')
    print(f'ci_all={ci_all:.4f}')

if __name__ == '__main__':
    main()