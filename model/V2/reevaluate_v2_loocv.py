"""
重新用 LOOCV 评估 V2.1（确保和 V3 评估方式一致）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from scipy.stats import spearmanr, kendalltau

DATA_PATH = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\final_data\cleaned_data.csv')
V2_DIR = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\model\V2')

# 加载数据
df = pd.read_csv(DATA_PATH)

# V2.1 的 118 特征
exclude_cols = ['player', 'year', 'rank', 'rating_2.0_both', 'rating_3.0_both']
features_118 = [c for c in df.columns if c not in exclude_cols]

df[features_118] = df[features_118].fillna(df[features_118].median())

# 准备 Pairwise 数据（差值）
def prepare_pairwise(df, features):
    X_list, y_list, pairs = [], [], []
    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year].reset_index(drop=True)
        df_year['score'] = -np.log(df_year['rank'])
        n = len(df_year)
        for i in range(n):
            for j in range(i+1, n):
                feat_A = df_year.loc[i, features].values
                feat_B = df_year.loc[j, features].values
                X_list.append(feat_A - feat_B)
                y_list.append(df_year.loc[i, 'score'] - df_year.loc[j, 'score'])
                pairs.append({
                    'year': year,
                    'player_A': df_year.loc[i, 'player'],
                    'player_B': df_year.loc[j, 'player'],
                    'rank_A': df_year.loc[i, 'rank'],
                    'rank_B': df_year.loc[j, 'rank']
                })
    return np.array(X_list), np.array(y_list), pd.DataFrame(pairs)

X_pair, y_pair, pairs_df = prepare_pairwise(df, features_118)

# LOOCV
years = sorted(df['year'].unique())
yearly_metrics = []

for test_year in years:
    print(f'评估 {test_year}年...')
    
    train_mask = pairs_df['year'] != test_year
    test_mask = pairs_df['year'] == test_year
    
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    model.fit(X_pair[train_mask], y_pair[train_mask])
    
    # 预测
    y_pred = model.predict(X_pair[test_mask])
    pairs_test = pairs_df[test_mask].reset_index(drop=True)
    
    # Ranking Reconstruction
    player_scores, player_counts = {}, {}
    for idx in range(len(pairs_test)):
        A, B, diff = pairs_test.loc[idx, 'player_A'], pairs_test.loc[idx, 'player_B'], y_pred[idx]
        if A not in player_scores: player_scores[A], player_counts[A] = 0, 0
        if B not in player_scores: player_scores[B], player_counts[B] = 0, 0
        player_scores[A] += diff
        player_counts[A] += 1
        player_scores[B] -= diff
        player_counts[B] += 1
    
    for p in player_scores:
        if player_counts[p] > 0: player_scores[p] /= player_counts[p]
    
    sorted_players = sorted(player_scores.items(), key=lambda x: -x[1])
    pred_rank = {p: r+1 for r, (p, _) in enumerate(sorted_players)}
    
    # 真实数据
    year_df = df[df['year'] == test_year]
    true_ranks = year_df['rank'].tolist()
    pred_ranks = [pred_rank.get(p, 10) for p in year_df['player']]
    
    # 指标
    spearman, _ = spearmanr(true_ranks, pred_ranks)
    kendall, _ = kendalltau(true_ranks, pred_ranks)
    top5_true = set(np.argsort(true_ranks)[:5])
    top5_pred = set(np.argsort(pred_ranks)[:5])
    top5_overlap = len(top5_true & top5_pred) / 5.0
    top1_acc = 1.0 if np.argmin(true_ranks) == np.argmin(pred_ranks) else 0.0
    
    yearly_metrics.append({
        'year': test_year,
        'spearman': spearman,
        'kendall_tau': kendall,
        'top5_overlap': top5_overlap,
        'top1_acc': top1_acc
    })
    
    print(f'    Spearman={spearman:.4f}, Kendall={kendall:.4f}, Top5={top5_overlap:.1%}, Top1={top1_acc:.0%}')

# 平均
avg_spearman = np.mean([m['spearman'] for m in yearly_metrics])
avg_kendall = np.mean([m['kendall_tau'] for m in yearly_metrics])
avg_top5 = np.mean([m['top5_overlap'] for m in yearly_metrics])
avg_top1 = np.mean([m['top1_acc'] for m in yearly_metrics])

print(f'\nV2.1 LOOCV 平均指标:')
print(f'    Spearman = {avg_spearman:.4f}')
print(f'    Kendall Tau = {avg_kendall:.4f}')
print(f'    Top5 Overlap = {avg_top5:.4f}')
print(f'    Top1 Accuracy = {avg_top1:.2%}')

# 保存
pd.DataFrame(yearly_metrics).to_csv(V2_DIR / 'yearly_metrics_v2_loocv.csv', index=False)
print(f'\n已保存：yearly_metrics_v2_loocv.csv')
