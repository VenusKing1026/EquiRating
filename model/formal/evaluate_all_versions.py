"""
EquiRating 统一评估框架

功能：
1. 统一评估 V1/V2/V3 三个版本
2. 统一指标：Spearman, Kendall, Top5, Top1
3. 统一可视化（4 张标准图 + 中文支持）
4. 确保 LOOCV 评估方式一致

使用方法：
    python evaluate_all_versions.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['font.size'] = 10

import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
import xgboost as xgb

# ============================================================================
# 配置
# ============================================================================
DATA_PATH = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\final_data\cleaned_data.csv')
FORMAL_DIR = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\model\formal')
V1_DIR = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\model\V1')
V2_DIR = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\model\V2')
V3_DIR = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\model\V3')

FORMAL_DIR.mkdir(exist_ok=True)

# ============================================================================
# 特征定义
# ============================================================================
# V1: 118 特征
def get_v1_features(df):
    exclude_cols = ['player', 'year', 'rank', 'rating_2.0_both', 'rating_3.0_both']
    return [c for c in df.columns if c not in exclude_cols]

# V2: 118 特征差值
def get_v2_features(df):
    exclude_cols = ['player', 'year', 'rank', 'rating_2.0_both', 'rating_3.0_both']
    return [c for c in df.columns if c not in exclude_cols]

# V3: 32 特征
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

# ============================================================================
# Pairwise 数据准备
# ============================================================================
def prepare_pairwise_data(df, features):
    """构建 Pairwise 配对数据（特征差值 + 对数 Label）"""
    X_list, y_list, pairs_info = [], [], []
    
    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year].reset_index(drop=True)
        df_year['score'] = -np.log(df_year['rank'])
        n = len(df_year)
        
        for i in range(n):
            for j in range(i + 1, n):
                feat_A = df_year.loc[i, features].values
                feat_B = df_year.loc[j, features].values
                X_list.append(feat_A - feat_B)
                y_list.append(df_year.loc[i, 'score'] - df_year.loc[j, 'score'])
                pairs_info.append({
                    'year': year,
                    'player_A': df_year.loc[i, 'player'],
                    'player_B': df_year.loc[j, 'player'],
                    'rank_A': df_year.loc[i, 'rank'],
                    'rank_B': df_year.loc[j, 'rank']
                })
    
    return np.array(X_list), np.array(y_list), pd.DataFrame(pairs_info)

# ============================================================================
# Ranking Reconstruction
# ============================================================================
def reconstruct_ranking_from_predictions(year, pairs_test, y_pred):
    """从配对预测反推年度排名"""
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
        if player_counts[p] > 0:
            player_scores[p] /= player_counts[p]
    
    sorted_players = sorted(player_scores.items(), key=lambda x: -x[1])
    pred_rank = {player: rank + 1 for rank, (player, _) in enumerate(sorted_players)}
    
    return pred_rank

# ============================================================================
# 评估指标
# ============================================================================
def calculate_metrics(true_ranks, pred_ranks):
    """计算统一评估指标"""
    true_ranks = np.array(true_ranks)
    pred_ranks = np.array(pred_ranks)
    
    spearman_corr, _ = spearmanr(true_ranks, pred_ranks)
    kendall_tau, _ = kendalltau(true_ranks, pred_ranks)
    
    top5_true = set(np.argsort(true_ranks)[:5])
    top5_pred = set(np.argsort(pred_ranks)[:5])
    top5_overlap = len(top5_true & top5_pred) / 5.0
    
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
# V1 评估（直接预测 → 转换为 Pairwise 评估）
# ============================================================================
def evaluate_v1(df, features):
    """评估 V1 模型（使用 LOOCV + Pairwise 重建）"""
    print('评估 V1 (118 特征，直接预测)...')
    
    years = sorted(df['year'].unique())
    yearly_metrics = []
    all_true_ranks, all_pred_ranks = [], []
    
    for test_year in years:
        train_df = df[df['year'] != test_year]
        test_df = df[df['year'] == test_year].copy()
        
        X_train = train_df[features]
        y_train = -np.log(train_df['rank'])
        X_test = test_df[features]
        
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # 预测分数
        pred_scores = model.predict(X_test)
        
        # 转换为排名
        pred_ranks_temp = np.round(np.exp(-pred_scores)).astype(int)
        pred_ranks_temp = np.clip(pred_ranks_temp, 1, 20)
        
        # 处理并列排名
        score_to_rank = {}
        for rank, score in enumerate(sorted(set(pred_scores), reverse=True), 1):
            score_to_rank[score] = rank
        pred_ranks = [score_to_rank[s] for s in pred_scores]
        
        true_ranks = test_df['rank'].tolist()
        
        metrics = calculate_metrics(true_ranks, pred_ranks)
        metrics['year'] = test_year
        yearly_metrics.append(metrics)
        
        all_true_ranks.extend(true_ranks)
        all_pred_ranks.extend(pred_ranks)
    
    avg_metrics = {k: np.mean([m[k] for m in yearly_metrics]) for k in ['spearman', 'kendall_tau', 'top5_overlap', 'top1_acc']}
    
    return {
        'name': 'V1',
        'description': '118 特征，直接预测',
        'n_features': len(features),
        'yearly_metrics': yearly_metrics,
        'avg_metrics': avg_metrics,
        'all_true_ranks': all_true_ranks,
        'all_pred_ranks': all_pred_ranks
    }

# ============================================================================
# V2 评估（Pairwise 差值）
# ============================================================================
def evaluate_v2(df, features):
    """评估 V2 模型（118 特征差值）"""
    print('评估 V2 (118 特征差值)...')
    
    X_pair, y_pair, pairs_df = prepare_pairwise_data(df, features)
    years = sorted(df['year'].unique())
    yearly_metrics = []
    all_true_ranks, all_pred_ranks = [], []
    
    for test_year in years:
        train_mask = pairs_df['year'] != test_year
        test_mask = pairs_df['year'] == test_year
        
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1
        )
        model.fit(X_pair[train_mask], y_pair[train_mask])
        y_pred = model.predict(X_pair[test_mask])
        
        pairs_test = pairs_df[test_mask].reset_index(drop=True)
        pred_rank = reconstruct_ranking_from_predictions(test_year, pairs_test, y_pred)
        
        year_df = df[df['year'] == test_year]
        true_ranks = year_df['rank'].tolist()
        pred_ranks = [pred_rank.get(p, 10) for p in year_df['player']]
        
        metrics = calculate_metrics(true_ranks, pred_ranks)
        metrics['year'] = test_year
        yearly_metrics.append(metrics)
        
        all_true_ranks.extend(true_ranks)
        all_pred_ranks.extend(pred_ranks)
    
    avg_metrics = {k: np.mean([m[k] for m in yearly_metrics]) for k in ['spearman', 'kendall_tau', 'top5_overlap', 'top1_acc']}
    
    return {
        'name': 'V2',
        'description': '118 特征差值',
        'n_features': len(features),
        'yearly_metrics': yearly_metrics,
        'avg_metrics': avg_metrics,
        'all_true_ranks': all_true_ranks,
        'all_pred_ranks': all_pred_ranks
    }

# ============================================================================
# V3 评估（32 特征差值）
# ============================================================================
def evaluate_v3(df, features):
    """评估 V3 模型（32 特征差值）"""
    print('评估 V3 (32 特征差值)...')
    
    X_pair, y_pair, pairs_df = prepare_pairwise_data(df, features)
    years = sorted(df['year'].unique())
    yearly_metrics = []
    all_true_ranks, all_pred_ranks = [], []
    
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
        pred_rank = reconstruct_ranking_from_predictions(test_year, pairs_test, y_pred)
        
        year_df = df[df['year'] == test_year]
        true_ranks = year_df['rank'].tolist()
        pred_ranks = [pred_rank.get(p, 10) for p in year_df['player']]
        
        metrics = calculate_metrics(true_ranks, pred_ranks)
        metrics['year'] = test_year
        yearly_metrics.append(metrics)
        
        all_true_ranks.extend(true_ranks)
        all_pred_ranks.extend(pred_ranks)
    
    avg_metrics = {k: np.mean([m[k] for m in yearly_metrics]) for k in ['spearman', 'kendall_tau', 'top5_overlap', 'top1_acc']}
    
    return {
        'name': 'V3',
        'description': '32 特征差值',
        'n_features': len(features),
        'yearly_metrics': yearly_metrics,
        'avg_metrics': avg_metrics,
        'all_true_ranks': all_true_ranks,
        'all_pred_ranks': all_pred_ranks
    }

# ============================================================================
# 统一可视化
# ============================================================================
def visualize_comparison(results):
    """生成统一对比图表"""
    print('\n生成统一对比图表...')
    
    models = [r['name'] for r in results]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # 图 1: Spearman 对比
    fig, ax = plt.subplots(figsize=(12, 6))
    spearman_values = [r['avg_metrics']['spearman'] for r in results]
    bars = ax.bar(models, spearman_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Spearman 相关系数', fontsize=12, fontweight='bold')
    ax.set_title('图 1: Spearman Correlation 对比\n排名相关性越强越好', fontsize=14, pad=20, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='中等相关基线')
    for bar, val in zip(bars, spearman_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FORMAL_DIR / '01_spearman_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ 01_spearman_comparison.png')
    
    # 图 2: Kendall Tau 对比
    fig, ax = plt.subplots(figsize=(12, 6))
    kendall_values = [r['avg_metrics']['kendall_tau'] for r in results]
    bars = ax.bar(models, kendall_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Kendall Tau', fontsize=12, fontweight='bold')
    ax.set_title('图 2: Kendall Tau 对比\n排序一致性越强越好', fontsize=14, pad=20, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, label='中等相关基线')
    for bar, val in zip(bars, kendall_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FORMAL_DIR / '02_kendall_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ 02_kendall_comparison.png')
    
    # 图 3: Top5 Overlap 对比
    fig, ax = plt.subplots(figsize=(12, 6))
    top5_values = [r['avg_metrics']['top5_overlap'] for r in results]
    bars = ax.bar(models, top5_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Top5 Overlap', fontsize=12, fontweight='bold')
    ax.set_title('图 3: Top5 Overlap 对比\n预测 Top5 与真实 Top5 重合度', fontsize=14, pad=20, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='随机基线')
    for bar, val in zip(bars, top5_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FORMAL_DIR / '03_top5_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ 03_top5_comparison.png')
    
    # 图 4: 预测 - 真实排名对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        ax.scatter(result['all_true_ranks'], result['all_pred_ranks'], 
                   alpha=0.5, s=40, color=colors[idx], edgecolors='white', linewidth=0.5)
        ax.plot([1, 20], [1, 20], 'r--', linewidth=2, label='理想预测线')
        ax.set_xlabel('真实排名', fontsize=11, fontweight='bold')
        ax.set_ylabel('预测排名', fontsize=11, fontweight='bold')
        ax.set_title(f'{result["name"]}\n{result["description"]}\nSpearman={result["avg_metrics"]["spearman"]:.4f}', 
                     fontsize=12, pad=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(0.5, 20.5)
        ax.invert_yaxis()
    
    plt.suptitle('图 4: 预测 - 真实排名对比（所有年份）', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FORMAL_DIR / '04_pred_vs_actual_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ 04_pred_vs_actual_all.png')
    
    # 图 5: 按年份的 Spearman 对比
    fig, ax = plt.subplots(figsize=(12, 6))
    years = sorted(df['year'].unique())
    
    for idx, result in enumerate(results):
        spearmans = [m['spearman'] for m in result['yearly_metrics']]
        ax.plot(years, spearmans, marker='o', linewidth=2, markersize=8, 
                color=colors[idx], label=f'{result["name"]} ({result["n_features"]}特征)')
    
    ax.set_xlabel('年份', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spearman 相关系数', fontsize=12, fontweight='bold')
    ax.set_title('图 5: Spearman 按年份对比', fontsize=14, pad=20, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='中等相关基线')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FORMAL_DIR / '05_spearman_by_year.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ 05_spearman_by_year.png')
    
    print()

# ============================================================================
# 保存结果
# ============================================================================
def save_results(results):
    """保存汇总结果"""
    print('保存结果...')
    
    # 汇总表格
    summary_data = []
    for r in results:
        row = {
            '模型': r['name'],
            '描述': r['description'],
            '特征数': r['n_features'],
            'Spearman': r['avg_metrics']['spearman'],
            'Kendall Tau': r['avg_metrics']['kendall_tau'],
            'Top5 Overlap': r['avg_metrics']['top5_overlap'],
            'Top1 Accuracy': r['avg_metrics']['top1_acc']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(FORMAL_DIR / 'summary.csv', index=False, encoding='utf-8-sig')
    print('  ✓ summary.csv')
    
    # 年度指标
    yearly_data = []
    for r in results:
        for m in r['yearly_metrics']:
            row = {
                '模型': r['name'],
                '年份': m['year'],
                'Spearman': m['spearman'],
                'Kendall Tau': m['kendall_tau'],
                'Top5 Overlap': m['top5_overlap'],
                'Top1 Accuracy': m['top1_acc']
            }
            yearly_data.append(row)
    
    yearly_df = pd.DataFrame(yearly_data)
    yearly_df.to_csv(FORMAL_DIR / 'yearly_metrics.csv', index=False, encoding='utf-8-sig')
    print('  ✓ yearly_metrics.csv')
    
    # README
    with open(FORMAL_DIR / 'README.md', 'w', encoding='utf-8') as f:
        f.write('# EquiRating 统一评估结果\n\n')
        f.write('## 评估说明\n\n')
        f.write('- **评估方式**: Leave-One-Year-Out Cross-Validation (LOOCV)\n')
        f.write('- **统一指标**: Spearman, Kendall Tau, Top5 Overlap, Top1 Accuracy\n')
        f.write('- **所有版本使用相同的评估流程，确保可比性**\n\n')
        f.write('## 模型对比\n\n')
        f.write('| 模型 | 特征数 | Spearman | Kendall Tau | Top5 Overlap | Top1 Accuracy |\n')
        f.write('|------|--------|----------|-------------|--------------|---------------|\n')
        for r in results:
            f.write(f'| {r["name"]} | {r["n_features"]} | {r["avg_metrics"]["spearman"]:.4f} | '
                    f'{r["avg_metrics"]["kendall_tau"]:.4f} | {r["avg_metrics"]["top5_overlap"]:.1%} | '
                    f'{r["avg_metrics"]["top1_acc"]:.1%} |\n')
        f.write('\n## 图表\n\n')
        f.write('1. `01_spearman_comparison.png`: Spearman 对比\n')
        f.write('2. `02_kendall_comparison.png`: Kendall Tau 对比\n')
        f.write('3. `03_top5_comparison.png`: Top5 Overlap 对比\n')
        f.write('4. `04_pred_vs_actual_all.png`: 预测 - 真实排名对比\n')
        f.write('5. `05_spearman_by_year.png`: Spearman 按年份对比\n\n')
        f.write('## 数据文件\n\n')
        f.write('- `summary.csv`: 汇总结果\n')
        f.write('- `yearly_metrics.csv`: 年度详细指标\n\n')
        f.write('## 结论\n\n')
        best = max(results, key=lambda x: x['avg_metrics']['spearman'])
        f.write(f'**最佳模型**: {best["name"]} (Spearman = {best["avg_metrics"]["spearman"]:.4f})\n')
    
    print('  ✓ README.md')
    print()

# ============================================================================
# 主函数
# ============================================================================
if __name__ == '__main__':
    print('=' * 80)
    print('EquiRating 统一评估框架')
    print('=' * 80)
    print()
    
    # 加载数据
    print('[1] 加载数据...')
    df = pd.read_csv(DATA_PATH)
    
    # 剔除 2016 年数据
    df = df[df['year'] != 2016].reset_index(drop=True)
    
    print(f'    数据：{len(df)}行 × {len(df.columns)}列（已剔除 2016 年）')
    print(f'    年份：{sorted(df["year"].unique())}')
    
    # 填充缺失值
    print('\n[2] 预处理...')
    v1_features = get_v1_features(df)
    v2_features = get_v2_features(df)
    v3_features = get_v3_features()
    
    df[v1_features] = df[v1_features].fillna(df[v1_features].median())
    df[v3_features] = df[v3_features].fillna(df[v3_features].median())
    print(f'    V1 特征数：{len(v1_features)}')
    print(f'    V2 特征数：{len(v2_features)}')
    print(f'    V3 特征数：{len(v3_features)}')
    print('    缺失值已填充')
    
    # 评估各版本
    print('\n[3] 评估各版本...')
    results = []
    
    results.append(evaluate_v1(df, v1_features))
    results.append(evaluate_v2(df, v2_features))
    results.append(evaluate_v3(df, v3_features))
    
    # 打印结果
    print('\n[4] 结果汇总...')
    print()
    print(f'{"模型":<8} {"特征数":<8} {"Spearman":<12} {"Kendall":<12} {"Top5":<12} {"Top1":<12}')
    print('-' * 70)
    for r in results:
        print(f'{r["name"]:<8} {r["n_features"]:<8} {r["avg_metrics"]["spearman"]:<12.4f} '
              f'{r["avg_metrics"]["kendall_tau"]:<12.4f} {r["avg_metrics"]["top5_overlap"]:<12.1%} '
              f'{r["avg_metrics"]["top1_acc"]:<12.1%}')
    print()
    
    # 可视化
    visualize_comparison(results)
    
    # 保存结果
    save_results(results)
    
    print('=' * 80)
    print('统一评估完成！')
    print('=' * 80)
    print()
    print(f'输出目录：{FORMAL_DIR}')
    print()
