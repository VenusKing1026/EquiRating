"""
EquiRating V2 改进版 - Pairwise Ranking with Feature Difference

改进点：
1. Pairwise 特征从 [A_features, B_features] 改为 A_features - B_features（差值）
2. 新增 Ranking Reconstruction：从配对预测恢复年度排名
3. 新增评估指标：Top1 Accuracy, Top5 Overlap, Spearman, Kendall Tau
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, percentileofscore

# ============================================================================
# 配置
# ============================================================================
DATA_PATH = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\final_data\cleaned_data.csv')
V2_DIR = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\model\V2')
V2_DIR.mkdir(exist_ok=True)

# ============================================================================
# 数据准备：构建配对样本（差值特征）
# ============================================================================
def prepare_pairwise_data_diff(df, features):
    """
    构建 pairwise 配对数据（使用特征差值）
    
    输入：
        df: 原始数据
        features: 特征列列表
    
    输出：
        X_pair: 配对特征 [feat_A_1 - feat_B_1, ..., feat_A_n - feat_B_n]
        y_pair: 实力分数差（score_A - score_B）
        pairs_info: 配对信息
    """
    print('[1] 构建配对样本（差值特征）...')
    
    X_list = []
    y_list = []
    pairs_info = []
    
    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year].reset_index(drop=True)
        n_players = len(df_year)
        
        # 计算实力分数（-log(rank)）
        df_year['score'] = -np.log(df_year['rank'])
        
        for i in range(n_players):
            for j in range(i + 1, n_players):
                player_A = df_year.loc[i, 'player']
                player_B = df_year.loc[j, 'player']
                rank_A = df_year.loc[i, 'rank']
                rank_B = df_year.loc[j, 'rank']
                score_A = df_year.loc[i, 'score']
                score_B = df_year.loc[j, 'score']
                
                # 特征差值：A - B
                feat_A = df_year.loc[i, features].values
                feat_B = df_year.loc[j, features].values
                X_pair = feat_A - feat_B  # 关键改进：用差值而不是拼接
                
                # Label：实力分数差
                y_pair = score_A - score_B
                
                X_list.append(X_pair)
                y_list.append(y_pair)
                pairs_info.append({
                    'year': year,
                    'player_A': player_A,
                    'player_B': player_B,
                    'rank_A': rank_A,
                    'rank_B': rank_B,
                    'score_A': score_A,
                    'score_B': score_B,
                    'score_diff': y_pair
                })
    
    X_pair = np.array(X_list)
    y_pair = np.array(y_list)
    pairs_df = pd.DataFrame(pairs_info)
    
    print(f'    总配对数：{len(X_pair)}')
    print(f'    特征维度：{X_pair.shape[1]}（差值）')
    print(f'    每年配对数：{len(X_pair) // 10}')
    print()
    
    return X_pair, y_pair, pairs_df

# ============================================================================
# Ranking Reconstruction：从配对预测恢复年度排名
# ============================================================================
def reconstruct_ranking_from_pairs(predictions_df, year):
    """
    从配对预测反推该年度的选手排名
    
    方法：Bradley-Terry 模型简化版
    - 每个选手的分数 = 所有配对的预测分数差的平均值
    - 根据分数排序得到排名
    """
    year_pairs = predictions_df[predictions_df['year'] == year]
    
    player_scores = {}
    player_counts = {}
    
    for _, row in year_pairs.iterrows():
        player_A = row['player_A']
        player_B = row['player_B']
        pred_diff = row['pred_diff']
        
        if player_A not in player_scores:
            player_scores[player_A] = 0
            player_counts[player_A] = 0
        if player_B not in player_scores:
            player_scores[player_B] = 0
            player_counts[player_B] = 0
        
        # A 相对于 B 的分数差
        player_scores[player_A] += pred_diff
        player_counts[player_A] += 1
        player_scores[player_B] -= pred_diff
        player_counts[player_B] += 1
    
    # 计算平均分数
    for player in player_scores:
        if player_counts[player] > 0:
            player_scores[player] /= player_counts[player]
    
    # 根据分数排序得到预测排名（分数越高排名越靠前）
    sorted_players = sorted(player_scores.items(), key=lambda x: -x[1])
    player_pred_rank = {player: rank + 1 for rank, (player, _) in enumerate(sorted_players)}
    
    return player_scores, player_pred_rank

# ============================================================================
# 评估指标计算
# ============================================================================
def calculate_metrics(true_ranks, pred_ranks, true_scores, pred_scores):
    """
    计算所有评估指标
    
    返回：
        metrics: 包含所有指标的字典
    """
    true_ranks = np.array(true_ranks)
    pred_ranks = np.array(pred_ranks)
    
    # 1. 排名准确率
    acc_1 = (abs(true_ranks - pred_ranks) <= 1).mean()
    acc_3 = (abs(true_ranks - pred_ranks) <= 3).mean()
    acc_5 = (abs(true_ranks - pred_ranks) <= 5).mean()
    
    # 2. Top1 Accuracy（预测第一名是否是真实第一名）
    top1_true = np.argmin(true_ranks)  # rank=1 的索引
    top1_pred = np.argmin(pred_ranks)
    top1_acc = 1.0 if top1_true == top1_pred else 0.0
    
    # 3. Top5 Overlap（预测 Top5 和真实 Top5 的重合度）
    top5_true = set(np.argsort(true_ranks)[:5])
    top5_pred = set(np.argsort(pred_ranks)[:5])
    top5_overlap = len(top5_true & top5_pred) / 5.0
    
    # 4. Spearman 相关系数
    spearman_corr, _ = spearmanr(true_ranks, pred_ranks)
    
    # 5. Kendall Tau
    kendall_tau, _ = kendalltau(true_ranks, pred_ranks)
    
    # 6. R²（分数）
    r2 = r2_score(true_scores, pred_scores)
    
    # 7. MAE（排名）
    mae = mean_absolute_error(true_ranks, pred_ranks)
    
    return {
        'acc_1': acc_1,
        'acc_3': acc_3,
        'acc_5': acc_5,
        'top1_acc': top1_acc,
        'top5_overlap': top5_overlap,
        'spearman': spearman_corr,
        'kendall_tau': kendall_tau,
        'r2': r2,
        'mae': mae
    }

# ============================================================================
# 模型训练
# ============================================================================
def train_v2():
    print('=' * 80)
    print('EquiRating V2 改进版 - Pairwise Ranking with Feature Difference')
    print('=' * 80)
    print()
    
    # 1. 加载数据
    print('[0] 加载数据...')
    df = pd.read_csv(DATA_PATH)
    print(f'    数据：{len(df)}行 × {len(df.columns)}列')
    print(f'    年份：{df["year"].min()} - {df["year"].max()}')
    print()
    
    # 2. 定义特征集
    print('[1] 定义特征集...')
    exclude_cols = ['player', 'year', 'rank', 'rating_2.0_both', 'rating_3.0_both']
    features_all = [c for c in df.columns if c not in exclude_cols]
    
    feature_mapping = {
        '每回合击杀': 'kills_per_round_both',
        '胜利回合击杀': 'kills_per_round_win_both',
        '回均伤害': 'damage_per_round_both',
        '多杀回合占比': 'rounds_with_a_multi-kill_both',
        '助攻击杀占比': 'assisted_kills_percentage_both',
        '换命击杀占比': 'trade_kills_percentage_both',
        '回均击杀伤害': 'damage_per_kill_both',
        '回均拆火击杀': 'trade_kills_per_round_both',
        '场均首杀': 'opening_kills_per_round_both',
        '突破成功率': 'opening_success_both',
        '场均道具伤害': 'utility_damage_per_round_both',
        '有效闪光率': 'flash_assists_per_round_both',
        '1v1 胜率': '1on1_win_percentage_both',
        '回均存活时间': 'time_alive_per_round_both',
    }
    features_14 = [v for v in feature_mapping.values() if v is not None]
    
    print(f'    V2.1 全部特征：{len(features_all)}维')
    print(f'    V2.2 14 特征：{len(features_14)}维')
    print()
    
    # 3. 填充缺失值
    print('[2] 预处理...')
    df[features_all] = df[features_all].fillna(df[features_all].median())
    df[features_14] = df[features_14].fillna(df[features_14].median())
    print('    缺失值已填充')
    print()
    
    # 4. 构建配对数据（差值特征）
    print('=' * 80)
    print('实验 1: V2.1（全部特征 - 差值）')
    print('=' * 80)
    X_pair_118, y_pair_118, pairs_118 = prepare_pairwise_data_diff(df, features_all)
    
    print('=' * 80)
    print('实验 2: V2.2（14 特征 - 差值）')
    print('=' * 80)
    X_pair_14, y_pair_14, pairs_14 = prepare_pairwise_data_diff(df, features_14)
    print()
    
    # 5. 训练模型（Leave-One-Year-Out CV）
    print('[3] Leave-One-Year-Out CV 训练...')
    print()
    
    years = sorted(df['year'].unique())
    results = {}
    
    # V2.1: 118 特征（差值）
    print('训练 V2.1 (XGBoost + 118 特征差值)...')
    fold_predictions_118 = []
    yearly_metrics_118 = []
    
    for test_year in years:
        train_mask = pairs_118['year'] != test_year
        X_train = X_pair_118[train_mask]
        y_train = y_pair_118[train_mask]
        
        test_mask = pairs_118['year'] == test_year
        X_test = X_pair_118[test_mask]
        y_test = y_pair_118[test_mask]
        pairs_test = pairs_118[test_mask].reset_index(drop=True)
        
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 保存配对预测
        for idx in range(len(pairs_test)):
            fold_predictions_118.append({
                'year': test_year,
                'player_A': pairs_test.loc[idx, 'player_A'],
                'player_B': pairs_test.loc[idx, 'player_B'],
                'rank_A': pairs_test.loc[idx, 'rank_A'],
                'rank_B': pairs_test.loc[idx, 'rank_B'],
                'true_diff': pairs_test.loc[idx, 'score_diff'],
                'pred_diff': y_pred[idx]
            })
        
        # Ranking Reconstruction
        preds_df_year = pd.DataFrame([{
            'year': test_year,
            'player_A': pairs_test.loc[idx, 'player_A'],
            'player_B': pairs_test.loc[idx, 'player_B'],
            'rank_A': pairs_test.loc[idx, 'rank_A'],
            'rank_B': pairs_test.loc[idx, 'rank_B'],
            'true_diff': pairs_test.loc[idx, 'score_diff'],
            'pred_diff': y_pred[idx]
        } for idx in range(len(pairs_test))])
        
        player_scores, player_pred_rank = reconstruct_ranking_from_pairs(preds_df_year, test_year)
        
        # 获取该年真实数据
        year_df = df[df['year'] == test_year]
        true_ranks = year_df['rank'].tolist()
        pred_ranks = [player_pred_rank.get(p, 10) for p in year_df['player']]
        true_scores = (-np.log(year_df['rank'])).tolist()
        pred_scores = [-np.log(r) for r in pred_ranks]
        
        # 计算所有指标
        metrics = calculate_metrics(true_ranks, pred_ranks, true_scores, pred_scores)
        metrics['year'] = test_year
        yearly_metrics_118.append(metrics)
    
    # 平均指标
    avg_metrics_118 = {k: np.mean([m[k] for m in yearly_metrics_118]) for k in yearly_metrics_118[0].keys() if k != 'year'}
    
    # 配对预测指标
    preds_df_118 = pd.DataFrame(fold_predictions_118)
    pairwise_mae_118 = mean_absolute_error(preds_df_118['true_diff'], preds_df_118['pred_diff'])
    pairwise_r2_118 = r2_score(preds_df_118['true_diff'], preds_df_118['pred_diff'])
    pairwise_acc_118 = ((preds_df_118['true_diff'] > 0) == (preds_df_118['pred_diff'] > 0)).mean()
    
    results['V2.1'] = {
        'features': features_all,
        'n_features': len(features_all),
        'pairwise_mae': pairwise_mae_118,
        'pairwise_r2': pairwise_r2_118,
        'pairwise_accuracy': pairwise_acc_118,
        'yearly_metrics': avg_metrics_118,
        'yearly_metrics_all': yearly_metrics_118,
        'predictions': preds_df_118
    }
    
    print(f'    Pairwise MAE = {pairwise_mae_118:.4f}')
    print(f'    Pairwise R² = {pairwise_r2_118:.4f}')
    print(f'    Pairwise 准确率 = {pairwise_acc_118:.2%}')
    print(f'    年度平均 Spearman = {avg_metrics_118["spearman"]:.4f}')
    print(f'    年度平均 Kendall Tau = {avg_metrics_118["kendall_tau"]:.4f}')
    print(f'    年度平均 Top1 Accuracy = {avg_metrics_118["top1_acc"]:.2%}')
    print(f'    年度平均 Top5 Overlap = {avg_metrics_118["top5_overlap"]:.2%}')
    print()
    
    # V2.2: 14 特征（差值）
    print('训练 V2.2 (XGBoost + 14 特征差值)...')
    fold_predictions_14 = []
    yearly_metrics_14 = []
    
    for test_year in years:
        train_mask = pairs_14['year'] != test_year
        X_train = X_pair_14[train_mask]
        y_train = y_pair_14[train_mask]
        
        test_mask = pairs_14['year'] == test_year
        X_test = X_pair_14[test_mask]
        y_test = y_pair_14[test_mask]
        pairs_test = pairs_14[test_mask].reset_index(drop=True)
        
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        for idx in range(len(pairs_test)):
            fold_predictions_14.append({
                'year': test_year,
                'player_A': pairs_test.loc[idx, 'player_A'],
                'player_B': pairs_test.loc[idx, 'player_B'],
                'rank_A': pairs_test.loc[idx, 'rank_A'],
                'rank_B': pairs_test.loc[idx, 'rank_B'],
                'true_diff': pairs_test.loc[idx, 'score_diff'],
                'pred_diff': y_pred[idx]
            })
        
        preds_df_year = pd.DataFrame([{
            'year': test_year,
            'player_A': pairs_test.loc[idx, 'player_A'],
            'player_B': pairs_test.loc[idx, 'player_B'],
            'rank_A': pairs_test.loc[idx, 'rank_A'],
            'rank_B': pairs_test.loc[idx, 'rank_B'],
            'true_diff': pairs_test.loc[idx, 'score_diff'],
            'pred_diff': y_pred[idx]
        } for idx in range(len(pairs_test))])
        
        player_scores, player_pred_rank = reconstruct_ranking_from_pairs(preds_df_year, test_year)
        
        year_df = df[df['year'] == test_year]
        true_ranks = year_df['rank'].tolist()
        pred_ranks = [player_pred_rank.get(p, 10) for p in year_df['player']]
        true_scores = (-np.log(year_df['rank'])).tolist()
        pred_scores = [-np.log(r) for r in pred_ranks]
        
        metrics = calculate_metrics(true_ranks, pred_ranks, true_scores, pred_scores)
        metrics['year'] = test_year
        yearly_metrics_14.append(metrics)
    
    avg_metrics_14 = {k: np.mean([m[k] for m in yearly_metrics_14]) for k in yearly_metrics_14[0].keys() if k != 'year'}
    
    preds_df_14 = pd.DataFrame(fold_predictions_14)
    pairwise_mae_14 = mean_absolute_error(preds_df_14['true_diff'], preds_df_14['pred_diff'])
    pairwise_r2_14 = r2_score(preds_df_14['true_diff'], preds_df_14['pred_diff'])
    pairwise_acc_14 = ((preds_df_14['true_diff'] > 0) == (preds_df_14['pred_diff'] > 0)).mean()
    
    results['V2.2'] = {
        'features': features_14,
        'n_features': len(features_14),
        'pairwise_mae': pairwise_mae_14,
        'pairwise_r2': pairwise_r2_14,
        'pairwise_accuracy': pairwise_acc_14,
        'yearly_metrics': avg_metrics_14,
        'yearly_metrics_all': yearly_metrics_14,
        'predictions': preds_df_14
    }
    
    print(f'    Pairwise MAE = {pairwise_mae_14:.4f}')
    print(f'    Pairwise R² = {pairwise_r2_14:.4f}')
    print(f'    Pairwise 准确率 = {pairwise_acc_14:.2%}')
    print(f'    年度平均 Spearman = {avg_metrics_14["spearman"]:.4f}')
    print(f'    年度平均 Kendall Tau = {avg_metrics_14["kendall_tau"]:.4f}')
    print(f'    年度平均 Top1 Accuracy = {avg_metrics_14["top1_acc"]:.2%}')
    print(f'    年度平均 Top5 Overlap = {avg_metrics_14["top5_overlap"]:.2%}')
    print()
    
    # 6. 结果对比
    print('[4] 结果对比...')
    print()
    print(f'{"实验":<10} {"特征":<6} {"Pair R²":<10} {"Pair Acc":<10} {"Spearman":<10} {"Kendall":<10} {"Top1":<10} {"Top5":<10}')
    print('-' * 90)
    for exp_name, result in results.items():
        feat_count = result['n_features']
        print(f'{exp_name:<10} {feat_count:<6} {result["pairwise_r2"]:<10.4f} {result["pairwise_accuracy"]:<10.2%} '
              f'{result["yearly_metrics"]["spearman"]:<10.4f} {result["yearly_metrics"]["kendall_tau"]:<10.4f} '
              f'{result["yearly_metrics"]["top1_acc"]:<10.2%} {result["yearly_metrics"]["top5_overlap"]:<10.2%}')
    print()
    
    # 7. 训练最终模型
    print('[5] 训练最终模型...')
    
    best_exp = max(results.keys(), key=lambda x: results[x]['pairwise_r2'])
    best_result = results[best_exp]
    best_features = best_result['features']
    
    X_all, y_all, _ = prepare_pairwise_data_diff(df, best_features)
    
    final_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    final_model.fit(X_all, y_all)
    
    model_path = V2_DIR / 'pairwise_v2_diff.model'
    final_model.save_model(str(model_path))
    print(f'    模型已保存：{model_path.name}')
    
    # 特征重要性（差值特征）
    importance_df = pd.DataFrame({
        'feature': best_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(V2_DIR / 'feature_importance_v2_diff.csv', index=False, encoding='utf-8')
    print(f'    特征重要性：feature_importance_v2_diff.csv')
    print()
    
    # 8. 可视化
    print('[6] 可视化...')
    visualize_v2(results, importance_df, best_features, feature_mapping)
    print()
    
    # 9. 保存结果
    print('[7] 保存结果...')
    for exp_name, result in results.items():
        result['predictions'].to_csv(V2_DIR / f'predictions_{exp_name.lower()}_diff.csv', index=False, encoding='utf-8')
    print(f'    预测结果：predictions_v2.1_diff.csv, predictions_v2.2_diff.csv')
    
    # 保存年度指标
    yearly_summary = []
    for exp_name, result in results.items():
        for m in result['yearly_metrics_all']:
            row = {'experiment': exp_name, **m}
            yearly_summary.append(row)
    pd.DataFrame(yearly_summary).to_csv(V2_DIR / 'yearly_metrics_v2_diff.csv', index=False, encoding='utf-8')
    print(f'    年度指标：yearly_metrics_v2_diff.csv')
    
    # 保存 README
    save_readme_v2(results, best_exp)
    print(f'    README: README.md')
    print()
    
    print('=' * 80)
    print('V2 改进版训练完成！')
    print('=' * 80)

# ============================================================================
# 可视化
# ============================================================================
def visualize_v2(results, importance_df, features, feature_mapping):
    """生成 V2 可视化图"""
    
    features_set = set(features)
    features_14_set = set(feature_mapping.values())
    
    # 图 1: 综合指标对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    exp_names = list(results.keys())
    pairwise_r2 = [results[exp]['pairwise_r2'] for exp in exp_names]
    spearman = [results[exp]['yearly_metrics']['spearman'] for exp in exp_names]
    top1_acc = [results[exp]['yearly_metrics']['top1_acc'] for exp in exp_names]
    top5_overlap = [results[exp]['yearly_metrics']['top5_overlap'] for exp in exp_names]
    
    # Pairwise R²
    ax = axes[0, 0]
    bars = ax.bar(exp_names, pairwise_r2, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Pairwise R²', fontsize=12)
    ax.set_title('Pairwise R² 对比', fontsize=14, pad=20)
    ax.set_ylim(0, 0.7)
    for bar, r2 in zip(bars, pairwise_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{r2:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Spearman
    ax = axes[0, 1]
    bars = ax.bar(exp_names, spearman, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Spearman 相关系数', fontsize=12)
    ax.set_title('年度排名 Spearman 对比', fontsize=14, pad=20)
    ax.set_ylim(0, 1)
    for bar, sp in zip(bars, spearman):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{sp:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Top1 Accuracy
    ax = axes[1, 0]
    bars = ax.bar(exp_names, top1_acc, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Top1 Accuracy', fontsize=12)
    ax.set_title('Top1 预测准确率对比', fontsize=14, pad=20)
    ax.set_ylim(0, 0.5)
    for bar, acc in zip(bars, top1_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Top5 Overlap
    ax = axes[1, 1]
    bars = ax.bar(exp_names, top5_overlap, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Top5 Overlap', fontsize=12)
    ax.set_title('Top5 重合度对比', fontsize=14, pad=20)
    ax.set_ylim(0, 1)
    for bar, ov in zip(bars, top5_overlap):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{ov:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(V2_DIR / 'v2_diff_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    ✓ 对比图：v2_diff_comparison.png')
    
    # 图 2: 特征重要性 Top 20
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_20 = importance_df.head(20)
    top_20['in_14'] = top_20['feature'].isin(features_14_set)
    
    colors = ['#F18F01' if row['in_14'] else '#C73E1D' for _, row in top_20.iterrows()]
    y_pos = np.arange(len(top_20))
    
    ax.barh(y_pos, top_20['importance'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_20['feature'], fontsize=10)
    ax.set_xlabel('特征重要性（差值）', fontsize=12)
    ax.set_title('V2 改进版 Top 20 重要特征\n🔴 橙色=在 14 特征内', fontsize=14, pad=20)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F18F01', alpha=0.7, edgecolor='black', label='在 14 特征内'),
        Patch(facecolor='#C73E1D', alpha=0.7, edgecolor='black', label='不在 14 特征内')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(V2_DIR / 'v2_diff_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    ✓ 特征重要性图：v2_diff_feature_importance.png')

# ============================================================================
# 保存 README
# ============================================================================
def save_readme_v2(results, best_exp):
    with open(V2_DIR / 'README_v2_diff.md', 'w', encoding='utf-8') as f:
        f.write('# EquiRating V2 改进版 - Pairwise with Feature Difference\n\n')
        f.write('## 改进点\n\n')
        f.write('1. **特征表示**: 从 `[A_features, B_features]` 改为 `A_features - B_features`（差值）\n')
        f.write('2. **Ranking Reconstruction**: 从配对预测反推年度排名\n')
        f.write('3. **新增指标**: Top1 Accuracy, Top5 Overlap, Spearman, Kendall Tau\n\n')
        f.write('## 结果对比\n\n')
        f.write('| 实验 | 特征 | Pair R² | Pair Acc | Spearman | Kendall | Top1 | Top5 |\n')
        f.write('|------|------|---------|----------|----------|---------|------|------|\n')
        for exp_name, result in results.items():
            m = result['yearly_metrics']
            f.write(f'| {exp_name} | {result["n_features"]} | {result["pairwise_r2"]:.4f} | '
                    f'{result["pairwise_accuracy"]:.2%} | {m["spearman"]:.4f} | {m["kendall_tau"]:.4f} | '
                    f'{m["top1_acc"]:.2%} | {m["top5_overlap"]:.2%} |\n')
        f.write(f'\n**最佳模型**: {best_exp}\n\n')
        f.write('## 文件列表\n\n')
        f.write('- `pairwise_v2_diff.model`: 最佳模型\n')
        f.write('- `predictions_v2.1_diff.csv`, `predictions_v2.2_diff.csv`: 预测结果\n')
        f.write('- `feature_importance_v2_diff.csv`: 特征重要性\n')
        f.write('- `yearly_metrics_v2_diff.csv`: 年度指标\n')
        f.write('- `v2_diff_comparison.png`: 综合指标对比\n')
        f.write('- `v2_diff_feature_importance.png`: 特征重要性图\n')
        f.write('- `train_v2_pairwise_v2.py`: 训练脚本\n')

# ============================================================================
# 运行
# ============================================================================
if __name__ == '__main__':
    train_v2()
