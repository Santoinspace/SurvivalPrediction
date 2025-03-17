import torch
import torch.nn as nn
import os
import random
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
plt.rcParams['font.family'] = 'Times New Roman'
figure_size = (10, 8)
font_size = 16

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def coxph(clinical_data, duration, event):
    df = pd.DataFrame(clinical_data, columns=[f"feature_{i}" for i in range(clinical_data.shape[1])])
    df['duration'] = duration
    df['event'] = event
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df, duration_col='duration', event_col='event')
    risk_scores = cph.predict_partial_hazard(df)
    return risk_scores

def plot_survival_curves(times, events, risks):
    risk_mean = risks.mean()
    idx_of_low_risk = np.argwhere(risks < risk_mean).squeeze()
    idx_of_high_risk = np.argwhere(risks >= risk_mean).squeeze()
    
    figure = plt.figure(figsize=figure_size)
    kmf = KaplanMeierFitter()
    kmf.fit(
        times[idx_of_low_risk],
        event_observed=events[idx_of_low_risk],
        label='Low Risk',
    )
    ax = kmf.plot_survival_function()
    kmf.fit(
        times[idx_of_high_risk],
        event_observed=events[idx_of_high_risk],
        label='High Risk',
    )
    kmf.plot_survival_function(ax=ax)
    ax.legend(loc="lower left", fontsize=font_size)
    # if ax.get_legend():
    #     ax.get_legend().remove()
    
    p_result = logrank_test(times[idx_of_low_risk],
                            times[idx_of_high_risk],
                            events[idx_of_low_risk],
                            events[idx_of_high_risk])
    p_value = p_result.p_value
    ax.set_title(f'p value={p_value:.4e}', fontsize=font_size)
    # plt.text(.04, .12, f'p_value={p_value:.4e}', fontsize=p_font_size, ha='left', va='top', transform=ax.transAxes)
    
    plt.xlabel('', fontsize=font_size)
    plt.xlabel('Timeline', fontsize=font_size)
    plt.ylabel('Survival Probability', fontsize=font_size)
    plt.tick_params(axis='x', labelsize=font_size)
    plt.tick_params(axis='y', labelsize=font_size)
    
    figure.canvas.draw()
    plt.close('all')
    
    return figure, p_value

def plot_tsne(feats, times, risks, title):
    risk_mean = risks.mean()
    idx_of_high_risk = np.argwhere(risks >= risk_mean)
    labels = np.zeros_like(times)
    labels[idx_of_high_risk] = 1

    targets = range(2)
    colors = ['blue', 'red']
    target_names = ["Low Risk", "High Risk"]
    
    tsne = TSNE(n_components=2, random_state=0, perplexity=10)
    X_tsne = tsne.fit_transform(feats)
    
    figure = plt.figure(figsize=figure_size)
    for target, color, label in zip(targets, colors, target_names):
        indices = [i for i, x in enumerate(labels) if x == target]
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=color, s=50, label=label, alpha=0.5)
    
    # plt.title(title)
    plt.legend(loc="upper right", fontsize=font_size)
    
    plt.tick_params(axis='x', labelsize=font_size)
    plt.tick_params(axis='y', labelsize=font_size)
    
    plt.close()
    
    return figure

def get_surv_array(time, event, intervals):
    """
    Transforms censored survival data into vector format that can be used in Keras.
    Args:
        time: Array of failure/censoring times.
        event: Array of censoring indicator. 1 if failed, 0 if censored.
        intervals: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Return:
        surv_array: Dimensions with (number of samples, number of time intervals*2)
    """
    
    breaks = np.array(intervals)
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    
    surv_array = np.zeros((n_intervals * 2))
    
    if event == 1:
        surv_array[0 : n_intervals] = 1.0 * (time >= breaks[1:]) 
        if time < breaks[-1]:
            surv_array[n_intervals + np.where(time < breaks[1:])[0][0]] = 1
    else:
        surv_array[0 : n_intervals] = 1.0 * (time >= breaks_midpoint)
    
    return surv_array