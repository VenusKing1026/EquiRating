"""
V3 每年预测 - 真实排名对比图
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置
DATA_PATH = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\final_data\cleaned_data.csv')
FORMAL_DIR = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\model\formal')
PLOTS_DIR = FORMAL_DIR / 'v3_yearly_plots'
PLOTS_DIR.mkdir(exist_ok=True)

# V3 特征
def get_v3_features():
    core_18 = [
        'kills_per_round_both', 'damage_per_round_both', 'damage_per_kill_both',
        'rounds_with_a_multi-kill_both', 'opening_kills_per_round_both',
        'opening_success_both', 'opening_deaths_per_round_both',
        'trade_kills_per_round_both', 'trade_kills_percentage_both',
        'traded_deaths_percentage_both', 'assists_per_round_both',
        'assisted_kills_percentage_both', 'utility_damage_per_round_both',
        'flash_assists_per_round_both', 'time_opponent_flashed_per_round_both',
        'clutch_points_per_round_both', '1on1_win_percentage_both',
        'time_alive_per_round_both',
    ]
    supplement_15 = [
        'attacks_per_round_both', 'flashes_thrown_per_round_both',
        'kills_per_round_win_both', 'last_alive_percentage_both',
        'opening_attempts_both', 'rounds_with_a_kill_both',
        'saved_by_teammate_per_round_both', 'saved_teammate_per_round_both',
        'saves_per_round_loss_both', 'support_rounds_both',
        'traded_deaths_per_round_both', 'utility_kills_per_100_rounds_both',
        'win%_after_opening_kill_both', 'damage_per_round_win_both',
    ]
    return list(dict.fromkeys(core_18 + supplement_15))

# Pairwise 数据
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

# Ranking Reconstruction
def reconstruct_ranking(pairs_test, y_pred):
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
    return {p: r+1 for r, (p, _) in enumerate(sorted_players)}

# 主程序
print('生成 V3 每年预测 - 真实排名对比图...')

# 加载数据
df = pd.read_csv(DATA_PATH)
df = df[df['year'] != 2016].reset_index(drop=True)
features = get_v3_features()
df[features] = df[features].fillna(df[features].median())

# 准备数据
X_pair, y_pair, pairs_df = prepare_pairwise(df, features)
years = sorted(df['year'].unique())

# 训练最终模型（用全部数据）
print('训练最终模型...')
final_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1
)
final_model.fit(X_pair, y_pair)

# 生成每年的图
print('生成每年的预测 - 真实对比图...')

for year in years:
    # LOOCV 预测
    train_mask = pairs_df['year'] != year
    test_mask = pairs_df['year'] == year
    
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    model.fit(X_pair[train_mask], y_pair[train_mask])
    y_pred = model.predict(X_pair[test_mask])
    
    pairs_test = pairs_df[test_mask].reset_index(drop=True)
    pred_rank = reconstruct_ranking(pairs_test, y_pred)
    
    # 真实数据
    year_df = df[df['year'] == year]
    true_ranks = year_df['rank'].tolist()
    pred_ranks = [pred_rank.get(p, 10) for p in year_df['player']]
    
    # 计算指标
    spearman, _ = spearmanr(true_ranks, pred_ranks)
    top5_true = set(np.argsort(true_ranks)[:5])
    top5_pred = set(np.argsort(pred_ranks)[:5])
    top5_overlap = len(top5_true & top5_pred) / 5.0
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(true_ranks, pred_ranks, alpha=0.7, s=80, color='#2E86AB', 
               edgecolors='white', linewidth=1)
    
    # 添加选手标签
    for i, player in enumerate(year_df['player']):
        ax.annotate(player, (true_ranks[i], pred_ranks[i]), 
                   fontsize=8, ha='center', va='bottom', alpha=0.7)
    
    ax.plot([1, 20], [1, 20], 'r--', linewidth=2, label='理想预测线')
    ax.set_xlabel('真实排名', fontsize=12, fontweight='bold')
    ax.set_ylabel('预测排名', fontsize=12, fontweight='bold')
    ax.set_title(f'V3 ({len(features)}特征) - {year}年\n'
                 f'Spearman = {spearman:.4f} | Top5 Overlap = {top5_overlap:.1%}',
                 fontsize=14, pad=20, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.5, 20.5)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'v3_{year}_pred_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {year}年 (Spearman={spearman:.4f}, Top5={top5_overlap:.1%})')

# 生成汇总图（所有年份拼在一起）
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()

for idx, year in enumerate(years):
    ax = axes[idx]
    
    # 重新加载该年数据
    train_mask = pairs_df['year'] != year
    test_mask = pairs_df['year'] == year
    
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    model.fit(X_pair[train_mask], y_pair[train_mask])
    y_pred = model.predict(X_pair[test_mask])
    
    pairs_test = pairs_df[test_mask].reset_index(drop=True)
    pred_rank = reconstruct_ranking(pairs_test, y_pred)
    
    year_df = df[df['year'] == year]
    true_ranks = year_df['rank'].tolist()
    pred_ranks = [pred_rank.get(p, 10) for p in year_df['player']]
    
    spearman, _ = spearmanr(true_ranks, pred_ranks)
    
    ax.scatter(true_ranks, pred_ranks, alpha=0.6, s=60, color='#2E86AB',
               edgecolors='white', linewidth=0.5)
    ax.plot([1, 20], [1, 20], 'r--', linewidth=2)
    ax.set_xlabel('真实排名', fontsize=9)
    ax.set_ylabel('预测排名', fontsize=9)
    ax.set_title(f'{year}年\nS={spearman:.3f}', fontsize=10, pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.5, 20.5)
    ax.invert_yaxis()
    ax.tick_params(labelsize=8)

plt.suptitle('V3 (32 特征) - 每年预测 - 真实排名对比\n(2017-2025 年，已剔除 2016 年)',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'v3_all_years_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\n  ✓ 汇总图：v3_all_years_summary.png')

print(f'\n完成！图表保存在：{PLOTS_DIR}')
