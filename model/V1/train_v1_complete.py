"""
EquiRating V1 完整训练脚本（单文件版）

实验设计（10 个模型）：
- V1.1.1: XGBoost + 118 特征 + y1 = 21-rank
- V1.1.2: XGBoost + 118 特征 + y2 = -log(rank)
- V1.1.3: XGBoost + 118 特征 + y3 = rank（原始）
- V1.2.1: XGBoost + 14 特征 + y1 = 21-rank
- V1.2.2: XGBoost + 14 特征 + y2 = -log(rank)
- V1.2.3: XGBoost + 14 特征 + y3 = rank（原始）
- V1.3.1: Ridge + 118 特征 + y1 = 21-rank
- V1.3.2: Ridge + 118 特征 + y2 = -log(rank)
- V1.3.3: Ridge + 14 特征 + y1 = 21-rank
- V1.3.4: Ridge + 14 特征 + y2 = -log(rank)

验证：Leave-One-Year-Out CV
输出：R² 对比图、预测 vs 真实图、特征重要性图
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

# ============================================================================
# 配置
# ============================================================================
DATA_PATH = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\final_data\cleaned_data.csv')
V1_DIR = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\model\V1')
V1_DIR.mkdir(exist_ok=True)

# ============================================================================
# Label 转换函数
# ============================================================================
def label_y1(rank):
    """y1 = 21 - rank（排名越高分数越高）"""
    return 21 - rank

def label_y2(rank):
    """y2 = -log(rank)（对数转换）"""
    return -np.log(rank)

def label_y3(rank):
    """y3 = rank（原始排名）"""
    return rank

def inverse_y1(score):
    return 21 - score

def inverse_y2(score):
    return np.round(np.exp(-score)).astype(int)

def inverse_y3(score):
    return np.round(score).astype(int)

# ============================================================================
# 主训练函数
# ============================================================================
def train_v1():
    print('=' * 80)
    print('EquiRating V1 完整训练')
    print('=' * 80)
    print()
    
    # 1. 加载数据
    print('[1] 加载数据...')
    df = pd.read_csv(DATA_PATH)
    print(f'    数据：{len(df)}行 × {len(df.columns)}列')
    print(f'    年份：{df["year"].min()} - {df["year"].max()}')
    print()
    
    # 2. 定义特征集
    print('[2] 定义特征集...')
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
    
    print(f'    全部特征：{len(features_all)}维')
    print(f'    14 特征：{len(features_14)}维')
    print()
    
    # 3. 填充缺失值
    print('[3] 预处理...')
    df[features_all] = df[features_all].fillna(df[features_all].median())
    df[features_14] = df[features_14].fillna(df[features_14].median())
    print('    缺失值已填充（中位数）')
    print()
    
    # 4. 实验配置（10 个模型）
    experiments = {
        'V1.1.1': {'features': features_all, 'label_func': label_y1, 'inverse_func': inverse_y1, 'label_name': 'y1', 'model_type': 'XGBoost'},
        'V1.1.2': {'features': features_all, 'label_func': label_y2, 'inverse_func': inverse_y2, 'label_name': 'y2', 'model_type': 'XGBoost'},
        'V1.1.3': {'features': features_all, 'label_func': label_y3, 'inverse_func': inverse_y3, 'label_name': 'y3', 'model_type': 'XGBoost'},
        'V1.2.1': {'features': features_14, 'label_func': label_y1, 'inverse_func': inverse_y1, 'label_name': 'y1', 'model_type': 'XGBoost'},
        'V1.2.2': {'features': features_14, 'label_func': label_y2, 'inverse_func': inverse_y2, 'label_name': 'y2', 'model_type': 'XGBoost'},
        'V1.2.3': {'features': features_14, 'label_func': label_y3, 'inverse_func': inverse_y3, 'label_name': 'y3', 'model_type': 'XGBoost'},
        'V1.3.1': {'features': features_all, 'label_func': label_y1, 'inverse_func': inverse_y1, 'label_name': 'y1', 'model_type': 'Ridge'},
        'V1.3.2': {'features': features_all, 'label_func': label_y2, 'inverse_func': inverse_y2, 'label_name': 'y2', 'model_type': 'Ridge'},
        'V1.3.3': {'features': features_14, 'label_func': label_y1, 'inverse_func': inverse_y1, 'label_name': 'y1', 'model_type': 'Ridge'},
        'V1.3.4': {'features': features_14, 'label_func': label_y2, 'inverse_func': inverse_y2, 'label_name': 'y2', 'model_type': 'Ridge'},
    }
    
    # 5. Leave-One-Year-Out CV
    print('[4] Leave-One-Year-Out CV...')
    print()
    
    years = sorted(df['year'].unique())
    results = {}
    
    for exp_name, config in experiments.items():
        print(f'训练 {exp_name} ({config["model_type"]} + {len(config["features"])}特征 + {config["label_name"]})...')
        
        features = config['features']
        label_func = config['label_func']
        inverse_func = config['inverse_func']
        model_type = config['model_type']
        
        fold_predictions = []
        fold_models = []
        
        for test_year in years:
            train_df = df[df['year'] != test_year]
            test_df = df[df['year'] == test_year]
            
            X_train = train_df[features]
            X_test = test_df[features]
            y_train = label_func(train_df['rank'])
            y_test = label_func(test_df['rank'])
            
            if model_type == 'XGBoost':
                model = xgb.XGBRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
                )
            else:
                model = Ridge(alpha=1.0)
            
            model.fit(X_train, y_train)
            fold_models.append(model)
            y_pred = model.predict(X_test)
            
            fold_predictions.append({
                'year': test_year,
                'true_rank': test_df['rank'].tolist(),
                'true_label': y_test.tolist(),
                'pred_label': y_pred.tolist()
            })
        
        # 汇总结果
        fold_true_ranks = []
        fold_true_labels = []
        fold_pred_labels = []
        
        for fold in fold_predictions:
            fold_true_ranks.extend(fold['true_rank'])
            fold_true_labels.extend(fold['true_label'])
            fold_pred_labels.extend(fold['pred_label'])
        
        fold_true_ranks = np.array(fold_true_ranks)
        fold_true_labels = np.array(fold_true_labels)
        fold_pred_labels = np.array(fold_pred_labels)
        
        # 计算指标
        mae = mean_absolute_error(fold_true_labels, fold_pred_labels)
        rmse = np.sqrt(mean_squared_error(fold_true_labels, fold_pred_labels))
        r2 = r2_score(fold_true_labels, fold_pred_labels)
        
        # 转回 rank
        pred_rank = inverse_func(fold_pred_labels)
        pred_rank = np.clip(pred_rank, 1, 20)
        
        acc_1 = (abs(fold_true_ranks - pred_rank) <= 1).mean()
        acc_3 = (abs(fold_true_ranks - pred_rank) <= 3).mean()
        
        # 特征重要性
        if model_type == 'XGBoost':
            coef_importance = fold_models[0].feature_importances_
        else:
            coef_importance = np.mean([np.abs(m.coef_) for m in fold_models], axis=0)
        
        results[exp_name] = {
            'features': features,
            'label_func': label_func,
            'label_name': config['label_name'],
            'model_type': model_type,
            'r2': r2,
            'mae': mae,
            'acc_1': acc_1,
            'acc_3': acc_3,
            'true_ranks': fold_true_ranks,
            'pred_ranks': pred_rank,
            'coef_importance': coef_importance,
        }
        
        marker = '✓' if r2 > 0 else '✗'
        print(f'    {marker} R² = {r2:.4f}, MAE = {mae:.4f}, ±1 位={acc_1:.2%}')
    
    print()
    
    # 6. 结果表格
    print('[5] 结果对比...')
    print()
    print(f'{"实验":<10} {"模型":<10} {"特征":<6} {"Label":<6} {"R²":<10} {"MAE":<8} {"±1 位":<8}')
    print('-' * 70)
    for exp_name, result in results.items():
        feat_count = len(result['features'])
        marker = '✓' if result['r2'] > 0 else '✗'
        print(f'{marker} {exp_name:<8} {result["model_type"]:<10} {feat_count:<6} {result["label_name"]:<6} '
              f'{result["r2"]:<10.4f} {result["mae"]:<8.4f} {result["acc_1"]:<8.2%}')
    print()
    
    # 7. 可视化
    print('[6] 可视化...')
    visualize_results(results, features_14)
    print()
    
    # 8. 保存最佳模型
    print('[7] 保存模型...')
    valid_models = {k: v for k, v in results.items() if v['r2'] > 0}
    best_exp = max(valid_models.keys(), key=lambda x: valid_models[x]['r2'])
    best_result = results[best_exp]
    
    X_all = df[best_result['features']]
    y_all = best_result['label_func'](df['rank'])
    
    if best_result['model_type'] == 'XGBoost':
        final_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
        )
        final_model.fit(X_all, y_all)
        final_model.save_model(str(V1_DIR / 'teacher_v1.model'))
    else:
        final_model = Ridge(alpha=1.0)
        final_model.fit(X_all, y_all)
        joblib.dump(final_model, str(V1_DIR / 'teacher_v1.pkl'))
    
    # 预测和百分位数映射
    raw_scores = final_model.predict(X_all)
    percentile_scores = [percentileofscore(raw_scores, s) for s in raw_scores]
    df['equi_rating'] = percentile_scores
    df['raw_score'] = raw_scores
    df.to_csv(V1_DIR / 'predictions_v1.csv', index=False, encoding='utf-8')
    
    print(f'    最佳模型：{best_exp} (R² = {best_result["r2"]:.4f})')
    print(f'    模型已保存：teacher_v1.model')
    print(f'    预测已保存：predictions_v1.csv')
    print()
    
    # 9. 保存 README
    save_readme(results, best_exp, best_result)
    
    print('=' * 80)
    print('训练完成！')
    print('=' * 80)

# ============================================================================
# 可视化函数
# ============================================================================
def visualize_results(results, features_14):
    """生成 3 张可视化图"""
    
    # 过滤 R² > 0 的模型
    valid_models = {k: v for k, v in results.items() if v['r2'] > 0}
    
    # 图 1: R² 柱状图（只显示 R² > 0）
    fig, ax = plt.subplots(figsize=(12, 6))
    exp_names = list(valid_models.keys())
    r2_values = [valid_models[exp]['r2'] for exp in exp_names]
    colors = ['#2E86AB' if valid_models[exp]['model_type'] == 'XGBoost' else '#A23B72' for exp in exp_names]
    
    bars = ax.bar(exp_names, r2_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_xlabel('模型', fontsize=12)
    ax.set_title('所有模型 R² 对比（Leave-One-Year-Out CV）\n仅显示 R² > 0 的模型', fontsize=14, pad=20)
    ax.set_ylim(0, 0.7)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='R²=0.5 基准线')
    
    for bar, r2 in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, 
                f'{r2:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.legend(handles=[
        plt.Line2D([0], [0], color='#2E86AB', lw=4, alpha=0.8, label='XGBoost'),
        plt.Line2D([0], [0], color='#A23B72', lw=4, alpha=0.8, label='Ridge')
    ], loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(V1_DIR / 'r2_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    ✓ R² 对比图：r2_comparison.png')
    
    # 图 2: 所有模型的预测 vs 真实图
    n_models = len(results)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, (exp_name, result) in enumerate(results.items()):
        ax = axes[idx]
        color = '#2E86AB' if result['model_type'] == 'XGBoost' else '#A23B72'
        
        ax.scatter(result['true_ranks'], result['pred_ranks'], 
                   alpha=0.6, s=60, color=color, edgecolors='white', linewidth=0.5)
        ax.plot([1, 20], [1, 20], 'r--', linewidth=2, label='理想预测线')
        
        ax.set_xlabel('实际排名', fontsize=11)
        ax.set_ylabel('预测排名', fontsize=11)
        marker = '✓' if result['r2'] > 0 else '✗'
        ax.set_title(f'{exp_name} ({result["model_type"]})\nR² = {result["r2"]:.4f}, ±1 位={result["acc_1"]:.1%}', 
                     fontsize=11, pad=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(0.5, 20.5)
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(V1_DIR / 'pred_vs_actual_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    ✓ 预测 vs 真实图：pred_vs_actual_all.png')
    
    # 图 3: 特征重要性（4 个代表模型）
    model_list = ['V1.1.2', 'V1.2.2', 'V1.1.3', 'V1.2.3']
    titles = [
        'V1.1.2: XGBoost + 118 特征 + y2 (最佳)',
        'V1.2.2: XGBoost + 14 特征 + y2',
        'V1.1.3: XGBoost + 118 特征 + y3 (原始)',
        'V1.2.3: XGBoost + 14 特征 + y3 (原始)'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, (exp_name, title) in enumerate(zip(model_list, titles)):
        ax = axes[idx]
        result = results[exp_name]
        
        importance_df = pd.DataFrame({
            'feature': result['features'],
            'importance': result['coef_importance']
        }).sort_values('importance', ascending=False).head(20)
        
        importance_df['in_14'] = importance_df['feature'].isin(features_14)
        colors = ['#F18F01' if row['in_14'] else '#C73E1D' for _, row in importance_df.iterrows()]
        y_pos = np.arange(len(importance_df))
        
        ax.barh(y_pos, importance_df['importance'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['feature'], fontsize=9)
        ax.set_xlabel('特征重要性', fontsize=11)
        ax.set_title(title + f'\nR² = {result["r2"]:.4f}', fontsize=12, pad=15)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#F18F01', alpha=0.7, edgecolor='black', label='在 14 特征内'),
            Patch(facecolor='#C73E1D', alpha=0.7, edgecolor='black', label='不在 14 特征内')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(V1_DIR / 'feature_importance_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    ✓ 特征重要性图：feature_importance_all.png')

# ============================================================================
# 保存 README
# ============================================================================
def save_readme(results, best_exp, best_result):
    with open(V1_DIR / 'README.md', 'w', encoding='utf-8') as f:
        f.write('# EquiRating V1 最终模型\n\n')
        f.write('## 实验设计（10 个模型）\n\n')
        f.write('| 实验 | 模型 | 特征数 | Label | 说明 |\n')
        f.write('|------|------|--------|-------|------|\n')
        f.write('| V1.1.1 | XGBoost | 118 | y1 | 21-rank |\n')
        f.write('| V1.1.2 | XGBoost | 118 | y2 | -log(rank) |\n')
        f.write('| V1.1.3 | XGBoost | 118 | y3 | rank（原始） |\n')
        f.write('| V1.2.1 | XGBoost | 14 | y1 | 21-rank |\n')
        f.write('| V1.2.2 | XGBoost | 14 | y2 | -log(rank) |\n')
        f.write('| V1.2.3 | XGBoost | 14 | y3 | rank（原始） |\n')
        f.write('| V1.3.1 | Ridge | 118 | y1 | 21-rank |\n')
        f.write('| V1.3.2 | Ridge | 118 | y2 | -log(rank) |\n')
        f.write('| V1.3.3 | Ridge | 14 | y1 | 21-rank |\n')
        f.write('| V1.3.4 | Ridge | 14 | y2 | -log(rank) |\n')
        f.write('\n## 结果对比\n\n')
        f.write('| 实验 | 模型 | R² | MAE | ±1 位 | ±3 位 |\n')
        f.write('|------|------|-----|-----|-------|-------|\n')
        for exp_name, result in results.items():
            marker = '✓' if result['r2'] > 0 else '✗'
            f.write(f'| {marker} {exp_name} | {result["model_type"]} | {result["r2"]:.4f} | {result["mae"]:.4f} | {result["acc_1"]:.2%} | {result["acc_3"]:.2%} |\n')
        f.write(f'\n**最佳模型**: {best_exp} (R² = {best_result["r2"]:.4f})\n\n')
        f.write('## 文件列表\n\n')
        f.write('- `teacher_v1.model`: 最佳模型\n')
        f.write('- `predictions_v1.csv`: 预测结果（含 EquiRating 0-100 分）\n')
        f.write('- `r2_comparison.png`: R² 对比图\n')
        f.write('- `pred_vs_actual_all.png`: 所有模型预测 vs 真实图\n')
        f.write('- `feature_importance_all.png`: 特征重要性对比图\n')
        f.write('- `train_v1_complete.py`: 训练脚本（单文件）\n')
    print(f'    ✓ README: README.md')

# ============================================================================
# 运行
# ============================================================================
if __name__ == '__main__':
    train_v1()
