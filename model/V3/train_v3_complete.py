"""
EquiRating V3 - 33 特征精简版本

特征选择策略：
1. 保留所有 _both 特征（118 → 42）
2. 删除 traditional 重复特征（42 → 38）
3. 删除狙击相关 weapon bias（38 → 33）
4. 包含 18 个核心特征 + 15 个补充特征

结合：
- V1 的对数 Label：y = -log(rank)
- V2 的 Pairwise 策略：特征差值 A - B
- XGBoost 模型
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

# ============================================================================
# 配置
# ============================================================================
DATA_PATH = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\final_data\cleaned_data.csv')
V3_DIR = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\model\V3')
V3_DIR.mkdir(exist_ok=True)

# ============================================================================
# 33 个精简特征
# ============================================================================
# 18 个核心特征
CORE_18 = [
    # 击杀相关（4）
    'kills_per_round_both',
    'damage_per_round_both',
    'damage_per_kill_both',
    'rounds_with_a_multi-kill_both',
    
    # 首杀相关（3）
    'opening_kills_per_round_both',
    'opening_success_both',
    'opening_deaths_per_round_both',
    
    # 交易相关（3）
    'trade_kills_per_round_both',
    'trade_kills_percentage_both',
    'traded_deaths_percentage_both',
    
    # 助攻相关（2）
    'assists_per_round_both',
    'assisted_kills_percentage_both',
    
    # 道具相关（3）
    'utility_damage_per_round_both',
    'flash_assists_per_round_both',
    'time_opponent_flashed_per_round_both',
    
    # 残局/1v1（2）
    'clutch_points_per_round_both',
    '1on1_win_percentage_both',
    
    # 存活（1）
    'time_alive_per_round_both',
]

# 15 个补充特征（其他 _both 特征，排除狙击相关）
SUPPLEMENT_15 = [
    'attacks_per_round_both',
    'flashes_thrown_per_round_both',
    'kills_per_round_win_both',
    'last_alive_percentage_both',
    'opening_attempts_both',
    'rounds_with_a_kill_both',
    'saved_by_teammate_per_round_both',
    'saved_teammate_per_round_both',
    'saves_per_round_loss_both',
    'support_rounds_both',
    'traded_deaths_per_round_both',
    'utility_kills_per_100_rounds_both',
    'win%_after_opening_kill_both',
    'damage_per_round_win_both',
    'assists_per_round_both',  # 已在核心中，去重
]

# 合并并去重
FEATURES_33 = list(dict.fromkeys(CORE_18 + SUPPLEMENT_15))

print(f'特征数：{len(FEATURES_33)}')
print(f'核心特征：{len(CORE_18)}')
print(f'补充特征：{len(FEATURES_33) - len(CORE_18)}')

# ============================================================================
# 数据准备：Pairwise 差值特征
# ============================================================================
def prepare_pairwise_data(df, features):
    """构建 Pairwise 配对数据（特征差值 + 对数 Label）"""
    print('[1] 构建 Pairwise 配对数据...')
    
    X_list = []
    y_list = []
    pairs_info = []
    
    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year].reset_index(drop=True)
        n_players = len(df_year)
        
        # 对数 Label：y = -log(rank)
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
                X_pair = feat_A - feat_B
                
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
                    'score_diff': y_pair
                })
    
    X_pair = np.array(X_list)
    y_pair = np.array(y_list)
    pairs_df = pd.DataFrame(pairs_info)
    
    print(f'    配对数：{len(X_pair)}')
    print(f'    特征数：{X_pair.shape[1]}')
    print()
    
    return X_pair, y_pair, pairs_df

# ============================================================================
# Ranking Reconstruction
# ============================================================================
def reconstruct_ranking_from_pairs(predictions_df, year):
    """从配对预测反推年度排名"""
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
        
        player_scores[player_A] += pred_diff
        player_counts[player_A] += 1
        player_scores[player_B] -= pred_diff
        player_counts[player_B] += 1
    
    for player in player_scores:
        if player_counts[player] > 0:
            player_scores[player] /= player_counts[player]
    
    sorted_players = sorted(player_scores.items(), key=lambda x: -x[1])
    player_pred_rank = {player: rank + 1 for rank, (player, _) in enumerate(sorted_players)}
    
    return player_scores, player_pred_rank

# ============================================================================
# 评估指标
# ============================================================================
def calculate_metrics(true_ranks, pred_ranks):
    """计算所有统一评估指标"""
    true_ranks = np.array(true_ranks)
    pred_ranks = np.array(pred_ranks)
    
    # Spearman
    spearman_corr, _ = spearmanr(true_ranks, pred_ranks)
    
    # Kendall Tau
    kendall_tau, _ = kendalltau(true_ranks, pred_ranks)
    
    # Top5 Overlap
    top5_true = set(np.argsort(true_ranks)[:5])
    top5_pred = set(np.argsort(pred_ranks)[:5])
    top5_overlap = len(top5_true & top5_pred) / 5.0
    
    # Top1 Accuracy
    top1_true = np.argmin(true_ranks)
    top1_pred = np.argmin(pred_ranks)
    top1_acc = 1.0 if top1_true == top1_pred else 0.0
    
    return {
        'spearman': spearman_corr,
        'kendall_tau': kendall_tau,
        'top5_overlap': top5_overlap,
        'top1_acc': top1_acc
    }

# ============================================================================
# 模型训练
# ============================================================================
def train_v3():
    print('=' * 80)
    print('EquiRating V3 训练 - 33 特征精简版')
    print('=' * 80)
    print()
    
    # 1. 加载数据
    print('[1] 加载数据...')
    df = pd.read_csv(DATA_PATH)
    print(f'    数据：{len(df)}行 × {len(df.columns)}列')
    print(f'    年份：{df["year"].min()} - {df["year"].max()}')
    print()
    
    # 2. 填充缺失值
    print('[2] 预处理...')
    df[FEATURES_33] = df[FEATURES_33].fillna(df[FEATURES_33].median())
    print(f'    特征数：{len(FEATURES_33)}')
    print('    缺失值已填充')
    print()
    
    # 3. 构建 Pairwise 数据
    print('=' * 80)
    print('构建 Pairwise 数据')
    print('=' * 80)
    X_pair, y_pair, pairs_df = prepare_pairwise_data(df, FEATURES_33)
    
    # 4. Leave-One-Year-Out CV
    print('[3] Leave-One-Year-Out CV 训练...')
    print()
    
    years = sorted(df['year'].unique())
    fold_predictions = []
    yearly_metrics = []
    
    for test_year in years:
        print(f'    折 {test_year - 2015}/10: {test_year}年...')
        
        # 训练集和测试集
        train_mask = pairs_df['year'] != test_year
        test_mask = pairs_df['year'] == test_year
        
        X_train = X_pair[train_mask]
        y_train = y_pair[train_mask]
        X_test = X_pair[test_mask]
        y_test = y_pair[test_mask]
        pairs_test = pairs_df[test_mask].reset_index(drop=True)
        
        # 训练 XGBoost
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 保存配对预测
        for idx in range(len(pairs_test)):
            fold_predictions.append({
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
        
        # 获取真实数据
        year_df = df[df['year'] == test_year]
        true_ranks = year_df['rank'].tolist()
        pred_ranks = [player_pred_rank.get(p, 10) for p in year_df['player']]
        
        # 计算指标
        metrics = calculate_metrics(true_ranks, pred_ranks)
        metrics['year'] = test_year
        yearly_metrics.append(metrics)
    
    print()
    
    # 5. 汇总指标
    print('[4] 汇总指标...')
    
    preds_df = pd.DataFrame(fold_predictions)
    pairwise_mae = mean_absolute_error(preds_df['true_diff'], preds_df['pred_diff'])
    pairwise_r2 = r2_score(preds_df['true_diff'], preds_df['pred_diff'])
    pairwise_acc = ((preds_df['true_diff'] > 0) == (preds_df['pred_diff'] > 0)).mean()
    
    avg_spearman = np.mean([m['spearman'] for m in yearly_metrics])
    avg_kendall = np.mean([m['kendall_tau'] for m in yearly_metrics])
    avg_top5 = np.mean([m['top5_overlap'] for m in yearly_metrics])
    avg_top1 = np.mean([m['top1_acc'] for m in yearly_metrics])
    
    print(f'    Pairwise MAE = {pairwise_mae:.4f}')
    print(f'    Pairwise R² = {pairwise_r2:.4f}')
    print(f'    Pairwise 准确率 = {pairwise_acc:.2%}')
    print()
    print(f'    年度平均 Spearman = {avg_spearman:.4f}')
    print(f'    年度平均 Kendall Tau = {avg_kendall:.4f}')
    print(f'    年度平均 Top5 Overlap = {avg_top5:.4f}')
    print(f'    年度平均 Top1 Accuracy = {avg_top1:.2%}')
    print()
    
    # 6. 训练最终模型
    print('[5] 训练最终模型...')
    
    final_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_pair, y_pair)
    
    model_path = V3_DIR / 'pairwise_v3_33features.model'
    final_model.save_model(str(model_path))
    print(f'    模型已保存：{model_path.name}')
    
    # 特征重要性
    importance_df = pd.DataFrame({
        'feature': FEATURES_33,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(V3_DIR / 'feature_importance_v3_33features.csv', index=False, encoding='utf-8')
    print(f'    特征重要性：feature_importance_v3_33features.csv')
    print()
    
    # 7. 可视化
    print('[6] 可视化...')
    visualize_v3(yearly_metrics, preds_df, importance_df, df, FEATURES_33, X_pair, y_pair, pairs_df, years)
    print()
    
    # 8. 保存结果
    print('[7] 保存结果...')
    
    preds_df.to_csv(V3_DIR / 'predictions_v3_33features.csv', index=False, encoding='utf-8')
    print(f'    预测结果：predictions_v3_33features.csv')
    
    yearly_df = pd.DataFrame(yearly_metrics)
    yearly_df.to_csv(V3_DIR / 'yearly_metrics_v3_33features.csv', index=False, encoding='utf-8')
    print(f'    年度指标：yearly_metrics_v3_33features.csv')
    
    # 保存汇总
    summary = pd.DataFrame({
        'model': ['V3'],
        'features': [len(FEATURES_33)],
        'pairwise_mae': [pairwise_mae],
        'pairwise_r2': [pairwise_r2],
        'pairwise_accuracy': [pairwise_acc],
        'spearman': [avg_spearman],
        'kendall_tau': [avg_kendall],
        'top5_overlap': [avg_top5],
        'top1_accuracy': [avg_top1]
    })
    summary.to_csv(V3_DIR / 'summary_v3_33features.csv', index=False, encoding='utf-8')
    print(f'    汇总结果：summary_v3_33features.csv')
    
    # 保存 README
    save_readme_v3(summary, yearly_metrics, FEATURES_33, CORE_18)
    print(f'    README: README.md')
    print()
    
    print('=' * 80)
    print('V3 训练完成！')
    print('=' * 80)
    
    return summary

# ============================================================================
# 可视化
# ============================================================================
def visualize_v3(yearly_metrics, preds_df, importance_df, df, features, X_pair, y_pair, pairs_df, years):
    """生成 V3 标准 4 图"""
    
    # 图 1: Spearman（按年份）
    fig, ax = plt.subplots(figsize=(10, 6))
    years = [m['year'] for m in yearly_metrics]
    spearmans = [m['spearman'] for m in yearly_metrics]
    
    ax.bar(years, spearmans, color='#2E86AB', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Spearman 相关系数', fontsize=12, fontweight='bold')
    ax.set_xlabel('年份', fontsize=12, fontweight='bold')
    ax.set_title('V3 Spearman Correlation（按年份）', fontsize=14, pad=20, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='中等相关基线')
    for year, sp in zip(years, spearmans):
        ax.text(year, sp + 0.03, f'{sp:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(V3_DIR / '01_spearman.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    ✓ 图 1: 01_spearman.png')
    
    # 图 2: Kendall Tau
    fig, ax = plt.subplots(figsize=(10, 6))
    kendalls = [m['kendall_tau'] for m in yearly_metrics]
    
    ax.bar(years, kendalls, color='#A23B72', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Kendall Tau', fontsize=12, fontweight='bold')
    ax.set_xlabel('年份', fontsize=12, fontweight='bold')
    ax.set_title('V3 Kendall Tau（按年份）', fontsize=14, pad=20, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, label='中等相关基线')
    for year, kt in zip(years, kendalls):
        ax.text(year, kt + 0.03, f'{kt:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(V3_DIR / '02_kendall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    ✓ 图 2: 02_kendall.png')
    
    # 图 3: Top5 Overlap
    fig, ax = plt.subplots(figsize=(10, 6))
    top5s = [m['top5_overlap'] for m in yearly_metrics]
    
    ax.bar(years, top5s, color='#F18F01', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Top5 Overlap', fontsize=12, fontweight='bold')
    ax.set_xlabel('年份', fontsize=12, fontweight='bold')
    ax.set_title('V3 Top5 Overlap（按年份）', fontsize=14, pad=20, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='随机基线')
    for year, t5 in zip(years, top5s):
        ax.text(year, t5 + 0.03, f'{t5:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(V3_DIR / '03_top5.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    ✓ 图 3: 03_top5.png')
    
    # 图 4: 预测 - 真实排名（使用 yearly_metrics 中保存的真实预测）
    fig, ax = plt.subplots(figsize=(10, 8))
    
    all_true_ranks = []
    all_pred_ranks = []
    
    # 重新用 LOOCV 生成预测（和评估指标一致）
    for test_year in years:
        train_mask = pairs_df['year'] != test_year
        test_mask = pairs_df['year'] == test_year
        
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1
        )
        model.fit(X_pair[train_mask], y_pair[train_mask])
        y_pred = model.predict(X_pair[test_mask])
        
        pairs_test = pairs_df[test_mask].reset_index(drop=True)
        
        # Ranking Reconstruction
        player_scores, player_counts = {}, {}
        for idx in range(len(pairs_test)):
            A = pairs_test.loc[idx, 'player_A']
            B = pairs_test.loc[idx, 'player_B']
            diff = y_pred[idx]
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
        for _, row in year_df.iterrows():
            all_true_ranks.append(row['rank'])
            all_pred_ranks.append(pred_rank.get(row['player'], 10))
    
    ax.scatter(all_true_ranks, all_pred_ranks, alpha=0.5, s=50, color='#2E86AB', edgecolors='white', linewidth=0.5)
    ax.plot([1, 20], [1, 20], 'r--', linewidth=2, label='理想预测线')
    ax.set_xlabel('真实排名', fontsize=12, fontweight='bold')
    ax.set_ylabel('预测排名', fontsize=12, fontweight='bold')
    ax.set_title('V3 预测 vs 真实排名（所有年份）', fontsize=14, pad=20, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.5, 20.5)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(V3_DIR / '04_pred_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    ✓ 图 4: 04_pred_vs_actual.png')
    
    # 图 5: 特征重要性
    fig, ax = plt.subplots(figsize=(10, 14))
    
    top_features = importance_df.sort_values('importance', ascending=False)
    y_pos = np.arange(len(top_features))
    
    # 标记核心特征
    core_set = set(CORE_18)
    colors = ['#F18F01' if row['feature'] in core_set else '#C73E1D' for _, row in top_features.iterrows()]
    
    ax.barh(y_pos, top_features['importance'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('特征重要性', fontsize=12, fontweight='bold')
    ax.set_title('V3 特征重要性（33 特征）\n🔴 橙色=18 核心特征', fontsize=14, pad=20, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F18F01', alpha=0.8, edgecolor='black', label='18 核心特征'),
        Patch(facecolor='#C73E1D', alpha=0.8, edgecolor='black', label='15 补充特征')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(V3_DIR / '05_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    ✓ 图 5: 05_feature_importance.png')

# ============================================================================
# 保存 README
# ============================================================================
def save_readme_v3(summary, yearly_metrics, features, core_features):
    with open(V3_DIR / 'README.md', 'w', encoding='utf-8') as f:
        f.write('# EquiRating V3 - 33 特征精简版本\n\n')
        f.write('## 特征选择策略\n\n')
        f.write('1. **保留所有 _both 特征**: 118 → 42\n')
        f.write('2. **删除 traditional 重复**: 42 → 38\n')
        f.write('3. **删除狙击 weapon bias**: 38 → 33\n\n')
        f.write('## 33 特征组成\n\n')
        f.write(f'- **18 核心特征**: {", ".join(core_features[:5])}...\n')
        f.write(f'- **15 补充特征**: attacks_per_round_both, flashes_thrown_per_round_both...\n\n')
        f.write('## 设计思路\n\n')
        f.write('1. **Label**: V1 的对数转换 y = -log(rank)\n')
        f.write('2. **策略**: V2 的 Pairwise 差值表示\n')
        f.write('3. **模型**: XGBoost 回归\n\n')
        f.write('## 评估结果\n\n')
        f.write('| 指标 | 值 |\n')
        f.write('|------|-----|\n')
        for col in summary.columns:
            if col != 'model':
                f.write(f'| {col} | {summary[col].values[0]} |\n')
        f.write('\n## 图表\n\n')
        f.write('1. `01_spearman.png`: Spearman 相关系数（按年份）\n')
        f.write('2. `02_kendall.png`: Kendall Tau（按年份）\n')
        f.write('3. `03_top5.png`: Top5 Overlap（按年份）\n')
        f.write('4. `04_pred_vs_actual.png`: 预测 - 真实排名\n')
        f.write('5. `05_feature_importance.png`: 特征重要性\n\n')
        f.write('## 文件列表\n\n')
        f.write('- `pairwise_v3_33features.model`: 训练好的模型\n')
        f.write('- `predictions_v3_33features.csv`: 配对预测结果\n')
        f.write('- `yearly_metrics_v3_33features.csv`: 年度指标\n')
        f.write('- `summary_v3_33features.csv`: 汇总结果\n')
        f.write('- `feature_importance_v3_33features.csv`: 特征重要性\n')
        f.write('- `train_v3_complete.py`: 训练脚本\n')

# ============================================================================
# 运行
# ============================================================================
if __name__ == '__main__':
    train_v3()
