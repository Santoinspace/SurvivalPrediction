"""
This file aims to
train the model for predicting survival.

Author: Han
"""
from pathlib import Path

import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from tensorboardX import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import get_args
from utils.util import setup_seed, plot_survival_curves, plot_tsne, coxph, get_surv_array

class TrainerCoxPH():
    """
    Functions:
        run
        train
        test
        load_data
        load_network
        train_one_epoch
        test_one_epoch
        save_model
    """

    def __init__(self, args):
        self.args = args
        self.dataset_path = r'D:\AProjection\SurvivalPrediction\data\preprocessed'
        self.result_path = r'D:\AProjection\SurvivalPrediction\results\CoxPH'
        self.surv_curve_path = f'{self.result_path}\\surv_curve'
        self.tsne_path = f'{self.result_path}\\tsne'

    def run(self):
        self.test()

    def test(self):
        setup_seed(self.args.seed)
        
        samples = pd.read_csv(Path(self.dataset_path) / self.args.tabular_name).iloc[:, 0].tolist()
        events = pd.read_csv(Path(self.dataset_path) / self.args.tabular_name).iloc[:, -2].tolist()
        
        if len(self.args.intervals) == 0:
            times = pd.read_csv(Path(self.dataset_path) / self.args.tabular_name).iloc[:, -1].tolist()
            self.args.intervals = np.linspace(0, max(times), num=self.args.interval_num + 1, dtype=int)

        kf = StratifiedKFold(n_splits=self.args.fold_num, shuffle=True, random_state=self.args.seed)
        
        k = -1
        ci_of_all_folds = []
        bs_of_all_folds = []
        dice_of_all_folds = []
        p_value_of_all_folds = []
        aucs_of_all_folds = [[], [], []]
        label_time_total = np.empty([0, 1])
        label_event_total = np.empty([0, 1])
        scores_total = np.empty(0)
        surv_preds_total = np.empty([0, 10])
        feats_total = np.empty([0, self.args.feat_num]) if self.args.feat_num > 0 else np.empty([0, len(self.args.feat_cols)])
        
        summary_writer = SummaryWriter(f'{self.result_path}/summary')
        
        for _, eval_index in kf.split(samples, events):
            """log"""
            k += 1
            logger.info(f'Fold: {k}')
            
            """data"""
            samples_eval = [samples[i] for i in eval_index]
            data = self.load_data(root=self.dataset_path, tabular_csv=self.args.tabular_name, 
                                  feat_num=self.args.feat_num, samples=samples_eval, feat_cols=self.args.feat_cols)
            
            """eval"""
            ci, bs, p_value, aucs, label_time, label_event, scores, feats = self.eval_one_epoch(
                fold=k,
                summary_writer=summary_writer,
                data=data,
            )
            
            ci_of_all_folds.append(ci)
        #     bs_of_all_folds.append(bs)
        #     dice_of_all_folds.append(dice)
            p_value_of_all_folds.append(p_value)
        #     for i in range(len(aucs)):
        #         aucs_of_all_folds[i].append(aucs[i])
            label_time_total = np.concatenate((label_time_total, label_time))
            label_event_total = np.concatenate((label_event_total, label_event))
            scores_total = np.concatenate((scores_total, scores))
            # surv_preds_total = np.concatenate((surv_preds_total, surv_preds))
            feats_total = np.concatenate((feats_total, feats))
        
        ci_mean = np.mean(ci_of_all_folds)
        # bs_mean = np.mean(bs_of_all_folds)
        # dice_mean = np.mean(dice_of_all_folds)
        p_value_mean = np.mean(p_value_of_all_folds)
        # auc_1_mean = np.mean(aucs_of_all_folds[0])
        # auc_3_mean = np.mean(aucs_of_all_folds[1])
        # auc_5_mean = np.mean(aucs_of_all_folds[2])
        
        ci_std = np.std(ci_of_all_folds)
        # bs_std = np.std(bs_of_all_folds)
        # dice_std = np.std(dice_of_all_folds)
        p_value_std = np.std(p_value_of_all_folds)
        # auc_1_std = np.std(aucs_of_all_folds[0])
        # auc_3_std = np.std(aucs_of_all_folds[1])
        # auc_5_std = np.std(aucs_of_all_folds[2])
        
        summary_writer.add_scalar('ci/mean', ci_mean)
        # summary_writer.add_scalar('bs/mean', bs_mean)
        # summary_writer.add_scalar('dice/mean', dice_mean)
        summary_writer.add_scalar('p value/mean', p_value_mean)
        # summary_writer.add_scalar('auc 1/mean', auc_1_mean)
        # summary_writer.add_scalar('auc 3/mean', auc_3_mean)
        # summary_writer.add_scalar('auc 5/mean', auc_5_mean)
        
        summary_writer.add_scalar('ci/std', ci_std)
        # summary_writer.add_scalar('bs/std', bs_std)
        # summary_writer.add_scalar('dice/std', dice_std)
        summary_writer.add_scalar('p value/std', p_value_std)
        # summary_writer.add_scalar('auc 1/std', auc_1_std)
        # summary_writer.add_scalar('auc 3/std', auc_3_std)
        # summary_writer.add_scalar('auc 5/std', auc_5_std)
        
        figure_surv_curve_total, p_value_total = plot_survival_curves(label_time_total, label_event_total, scores_total)
        # figure_roc_total, aucs_total = plot_roc_for_intervals(surv_preds_total, label_time_total, label_event_total, self.args.intervals, self.args.time_spots)
        figure_tsne_total = plot_tsne(feats_total, label_time_total, scores_total, 'CoxPH_tsne')
        
        summary_writer.add_scalar(f'p value/total', p_value_total)
        # summary_writer.add_scalar(f'auc_1/total', aucs_total[0])
        # summary_writer.add_scalar(f'auc_3/total', aucs_total[1])
        # summary_writer.add_scalar(f'auc_5/total', aucs_total[2])
        summary_writer.add_figure(f'surv curve/total', figure_surv_curve_total)
        # summary_writer.add_figure(f'roc/total', figure_roc_total)
        summary_writer.add_figure(f'tsne/total', figure_tsne_total)
        figure_surv_curve_total.savefig(f'{self.surv_curve_path}/surv_curv.svg', dpi=600)
        # figure_roc_total.savefig(f'{self.roc_path}/roc.svg', dpi=600)
        figure_tsne_total.savefig(f'{self.tsne_path}/tsne.svg', dpi=600)
        
        logger.info(f'''ci={ci_mean:.4f}±{ci_std:.4f},
                    p_value={p_value_mean:.4f}±{p_value_std:.4f},
                    p_value_total={p_value_total:.4f}''')
                
        summary_writer.close()
        
    def load_data(self, root, tabular_csv, feat_num, samples, feat_cols=[None]):
        df = pd.read_csv(f'{root}/{tabular_csv}')
        lines = df.loc[df.iloc[:, 0].isin(samples)]
        tabular = lines.iloc[:, 1:feat_num + 1].values if feat_num > 0 else lines.iloc[:, feat_cols].values
        time = lines.iloc[:, -2:-1].values
        event = lines.iloc[:, -1:].values
        surv_arrays = np.zeros((0, self.args.interval_num * 2))
        for t, e in zip(time, event):
            surv_array = get_surv_array(t, e, self.args.intervals)[np.newaxis, ...]
            surv_arrays = np.concatenate([surv_arrays, surv_array])
        
        return tabular, time, event, surv_arrays
        
    def eval_one_epoch(
        self,
        fold,
        summary_writer,
        data,
    ):
        tabular, time, event, surv_array = data
        
        """pred"""
        scores = coxph(tabular, time, event)
        
        """cumulate metrics"""
        # scores是返回的风险值，越大越危险，存活的时间应该越短，作为生存期预测值，需要取负号
        # for i in range(len(scores)):
            # print(f"{i} ", " time:", time[i], " -scores:", -scores[i], " event:", event[i])
        ci = concordance_index(time, -scores, event)
        bs = 0
        # bs = get_brier_score(surv_pred, surv_array, time, event, self.args.intervals)
        
        p_value, aucs = None, None
        """log"""
        figure_surv_curve, p_value = plot_survival_curves(time, event, scores)
        # figure_roc, aucs = plot_roc_for_intervals(surv_preds, time, event, self.args.intervals, self.args.time_spots)
        figure_tsne = plot_tsne(tabular, time, scores, "CoxPH")
        logger.info(f'ci={ci:.4f}, p_value={p_value:.4e}')
        summary_writer.add_scalar(f'ci/test', ci, fold + 1)
        # summary_writer.add_scalar(f'bs/test', bs, fold + 1)
        summary_writer.add_scalar(f'p value', p_value, fold + 1)
        # summary_writer.add_scalar(f'auc_1', aucs[0], fold + 1)
        # summary_writer.add_scalar(f'auc_3', aucs[1], fold + 1)
        # summary_writer.add_scalar(f'auc_5', aucs[2], fold + 1)
        summary_writer.add_figure(f'surv curve', figure_surv_curve, fold + 1)
        # summary_writer.add_figure(f'roc', figure_roc, fold + 1)
        summary_writer.add_figure(f'tsne', figure_tsne, fold + 1)
        figure_surv_curve.savefig(f'{self.surv_curve_path}\\fold_{fold}_surv_curv.svg', dpi=600)
        # figure_roc.savefig(f'{self.roc_path}/fold_{fold}_roc.svg', dpi=600)
        figure_tsne.savefig(f'{self.tsne_path}\\fold_{fold}_tsne.svg', dpi=600)
            
        return ci, bs, p_value, aucs, time, event, scores, tabular

if __name__ == '__main__':
    args = get_args()
    trainer = TrainerCoxPH(args)
    trainer.run()
