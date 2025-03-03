import numpy as np

def calculate_time(surv_pred, intervals):
    breaks = np.array(intervals)
    surv_pred = np.array(surv_pred)
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    surv_time = 0
    for i in range(n_intervals):
        cumulative_prob = np.prod(surv_pred[0:i+1])
        surv_time += cumulative_prob * timegap[i]
    return surv_time

def get_brier_score(surv_pred, surv_array, time, event, intervals):
    # 预处理阶段
    breaks = np.array(intervals)  # 将时间区间转换为numpy数组
    surv_pred = np.array(surv_pred.squeeze(0))  # 去除预测值的单维度
    surv_array = surv_array.squeeze(0)  # 去除真实值的单维度
    
    # 时间区间相关计算
    n_intervals=len(breaks)-1  # 实际间隔数
    timegap = breaks[1:] - breaks[:-1]  # 计算各时间区间长度
    breaks_midpoint = breaks[:-1] + 0.5 * timegap  # 计算各区间的中点
    
    # Brier Score计算核心逻辑
    brier_score = 0
    cnt = 0
    for i in range(n_intervals):
        # 处理删失数据：当无事件发生且时间早于当前区间中点时停止计算
        if event.item() == 0 and time.item() < breaks_midpoint[i]:
            break
        
        # 计算累积生存概率（预测值）
        cumulative_prob = np.prod(surv_pred[0:i+1]) 
        # 累加平方差（预测值与真实值）
        brier_score += (cumulative_prob - surv_array[i]) ** 2
        cnt += 1
    
    # 返回平均平方差（Brier Score）
    return (brier_score / cnt) if cnt != 0 else 0




