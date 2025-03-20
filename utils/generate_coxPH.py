import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter

if __name__ == '__main__':
    data_path = r'D:\AProjection\SurvivalPrediction\data\huaxi\clinal.csv'
    data = pd.read_csv(data_path)
    samples = data.iloc[:, 0].values
    feat = data.iloc[:, 1:-2].values
    time = data.iloc[:, -2].values
    event = data.iloc[:, -1].values

    df = pd.DataFrame(feat, columns=[f"feature_{i}" for i in range(feat.shape[1])])
    df['duration'] = time
    df['event'] = event

    # print(df.head())

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df, duration_col='duration', event_col='event')
    risk_scores = cph.predict_partial_hazard(df)
    for i in range(len(samples)):
        print(f"{samples[i]} :",'\n', f"{risk_scores[i]}")

    # 在data的倒数第三列添加风险分数
    # data['risk_score'] = risk_scores
    data.insert(data.shape[1]-2, 'risk_score', risk_scores.astype(np.float32))

    # 保存风险分数
    data.to_csv(r'D:\AProjection\SurvivalPrediction\data\huaxi\clinal_risk_score.csv', index=False)



