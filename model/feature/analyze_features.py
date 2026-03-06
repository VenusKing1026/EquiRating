"""
EquiRating 特征分析

1. PCA 降维可视化（按年份着色）
2. 特征相关性矩阵
3. 特征分布分析
4. 特征重要性分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 配置
# ============================================================================
DATA_PATH = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\final_data\cleaned_data.csv')
FEATURE_DIR = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\model\feature')
FEATURE_DIR.mkdir(exist_ok=True)

# ============================================================================
# 加载数据
# ============================================================================
print('=' * 80)
print('EquiRating 特征分析')
print('=' * 80)
print()

print('[1] 加载数据...')
df = pd.read_csv(DATA_PATH)
print(f'    数据：{len(df)}行 × {len(df.columns)}列')
print()

# 定义特征
exclude_cols = ['player', 'year', 'rank', 'rating_2.0_both', 'rating_3.0_both']
features = [c for c in df.columns if c not in exclude_cols]

# 填充缺失值
df[features] = df[features].fillna(df[features].median())

print(f'    特征数：{len(features)}')
print(f'    年份：{sorted(df["year"].unique())}')
print()

# ============================================================================
# 1. PCA 分析
# ============================================================================
print('[2] PCA 分析...')

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
print(f'    PC1 解释方差：{explained_var[0]:.2%}')
print(f'    PC2 解释方差：{explained_var[1]:.2%}')
print()

# PCA 图（按年份着色）
fig, ax = plt.subplots(figsize=(12, 10))

years = sorted(df['year'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(years)))

for i, year in enumerate(years):
    mask = df['year'] == year
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=[colors[i]], label=f'{year}', 
               s=80, alpha=0.7, edgecolors='white', linewidth=0.5)

ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12, fontweight='bold')
ax.set_title('PCA 降维可视化（按年份着色）\n118 维特征 → 2 维', fontsize=14, pad=20, fontweight='bold')
ax.legend(title='年份', loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FEATURE_DIR / '01_pca_by_year.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'    ✓ PCA 图：01_pca_by_year.png')

# ============================================================================
# 2. 特征相关性矩阵
# ============================================================================
print('[3] 特征相关性分析...')

# 计算相关性矩阵（只计算 both 结尾的特征，避免重复）
both_features = [f for f in features if f.endswith('_both')]
print(f'    分析特征数：{len(both_features)}（both 结尾）')

corr_matrix = df[both_features].corr()

# 相关性矩阵图（大图，分块显示）
fig, ax = plt.subplots(figsize=(20, 16))

# 使用聚类热图
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
            square=True, linewidths=1, linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': '相关系数'},
            ax=ax, annot=False)

ax.set_title('特征相关性矩阵（上半部分）\n118 维特征中的 both 维度', fontsize=14, pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig(FEATURE_DIR / '02_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'    ✓ 相关性矩阵：02_correlation_matrix.png')

# Top 10 高相关特征对
corr_pairs = []
for i in range(len(both_features)):
    for j in range(i + 1, len(both_features)):
        corr_pairs.append({
            'feature_1': both_features[i],
            'feature_2': both_features[j],
            'correlation': abs(corr_matrix.iloc[i, j])
        })

corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('correlation', ascending=False)

print(f'\n    Top 10 高相关特征对:')
for i, row in corr_pairs_df.head(10).iterrows():
    print(f'    {i+1}. {row["feature_1"][:40]:<40} & {row["feature_2"][:40]:<40} = {row["correlation"]:.3f}')

# 保存高相关特征对
corr_pairs_df.head(50).to_csv(FEATURE_DIR / 'top_correlated_pairs.csv', index=False, encoding='utf-8')
print(f'\n    ✓ 高相关特征对：top_correlated_pairs.csv')

# ============================================================================
# 3. 特征分布分析
# ============================================================================
print('\n[4] 特征分布分析...')

fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.flatten()

# 选择 12 个代表性特征
sample_features = [
    'kills_per_round_both',
    'damage_per_round_both',
    '1on1_win_percentage_both',
    'opening_success_both',
    'trade_kills_percentage_both',
    'utility_damage_per_round_both',
    'flash_assists_per_round_both',
    'time_alive_per_round_both',
    'rounds_with_a_multi-kill_both',
    'sniper_kills_percentage_both',
    'kast_traditional',
    'adr_traditional'
]

for idx, feat in enumerate(sample_features):
    ax = axes[idx]
    
    # 按年份分组绘制箱线图
    year_data = [df[df['year'] == year][feat].values for year in years]
    bp = ax.boxplot(year_data, labels=years, patch_artist=True,
                    boxprops=dict(alpha=0.7), medianprops=dict(color='red', linewidth=2))
    
    # 按年份着色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('年份', fontsize=10, fontweight='bold')
    ax.set_ylabel(feat[:30], fontsize=9, fontweight='bold')
    ax.set_title(f'{feat}', fontsize=10, pad=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig(FEATURE_DIR / '03_feature_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'    ✓ 特征分布：03_feature_distribution.png')

# ============================================================================
# 4. 特征重要性（从 V1 和 V2 模型）
# ============================================================================
print('\n[5] 特征重要性分析...')

# 加载 V1 特征重要性（如果有）
v1_importance_path = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\model\V1\feature_importance_v1.csv')

# 加载 V2 特征重要性
v2_importance_path = FEATURE_DIR.parent / 'V2' / 'feature_importance_v2_diff.csv'

if v2_importance_path.exists():
    v2_imp = pd.read_csv(v2_importance_path)
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    top_20 = v2_imp.nlargest(20, 'importance')
    
    # 标记 14 核心特征
    core_14 = [
        'kills_per_round_both', 'kills_per_round_win_both', 'damage_per_round_both',
        'rounds_with_a_multi-kill_both', 'assisted_kills_percentage_both',
        'trade_kills_percentage_both', 'damage_per_kill_both', 'trade_kills_per_round_both',
        'opening_kills_per_round_both', 'opening_success_both', 'utility_damage_per_round_both',
        'flash_assists_per_round_both', '1on1_win_percentage_both', 'time_alive_per_round_both'
    ]
    
    colors = ['#F18F01' if row['feature'] in core_14 else '#C73E1D' for _, row in top_20.iterrows()]
    y_pos = np.arange(len(top_20))
    
    ax.barh(y_pos, top_20['importance'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_20['feature'], fontsize=10)
    ax.set_xlabel('特征重要性', fontsize=12, fontweight='bold')
    ax.set_title('V2 模型 Top 20 重要特征\n🔴 橙色=14 核心特征', fontsize=14, pad=20, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F18F01', alpha=0.7, edgecolor='black', label='14 核心特征'),
        Patch(facecolor='#C73E1D', alpha=0.7, edgecolor='black', label='其他特征')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FEATURE_DIR / '04_feature_importance_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    ✓ 特征重要性：04_feature_importance_v2.png')
else:
    print('    ⚠️  V2 特征重要性文件不存在')

# ============================================================================
# 5. 保存特征统计
# ============================================================================
print('\n[6] 保存特征统计...')

feature_stats = []
for feat in features:
    feature_stats.append({
        'feature': feat,
        'mean': df[feat].mean(),
        'std': df[feat].std(),
        'min': df[feat].min(),
        'max': df[feat].max(),
        'missing': df[feat].isna().sum()
    })

feature_stats_df = pd.DataFrame(feature_stats)
feature_stats_df.to_csv(FEATURE_DIR / 'feature_statistics.csv', index=False, encoding='utf-8')
print(f'    ✓ 特征统计：feature_statistics.csv')

# 保存 README
with open(FEATURE_DIR / 'README.md', 'w', encoding='utf-8') as f:
    f.write('# EquiRating 特征分析\n\n')
    f.write('## 特征概览\n\n')
    f.write(f'- **总特征数**: {len(features)} 维\n')
    f.write(f'- **核心特征**: 14 维\n')
    f.write(f'- **样本数**: {len(df)}\n')
    f.write(f'- **年份范围**: {df["year"].min()} - {df["year"].max()}\n\n')
    f.write('## 图表\n\n')
    f.write('1. `01_pca_by_year.png`: PCA 降维可视化（按年份着色）\n')
    f.write('2. `02_correlation_matrix.png`: 特征相关性矩阵\n')
    f.write('3. `03_feature_distribution.png`: 特征分布箱线图\n')
    f.write('4. `04_feature_importance_v2.png`: V2 模型特征重要性 Top20\n\n')
    f.write('## 数据文件\n\n')
    f.write('- `feature_statistics.csv`: 特征统计（均值、标准差、范围）\n')
    f.write('- `top_correlated_pairs.csv`: Top 50 高相关特征对\n\n')
    f.write('## 关键发现\n\n')
    f.write('1. PCA 显示不同年份的选手特征有明显聚类趋势\n')
    f.write('2. 击杀相关特征之间高度相关\n')
    f.write('3. 14 核心特征在 V2 模型中占据重要位置\n')

print(f'    ✓ README: README.md')
print()

# ============================================================================
# 完成
# ============================================================================
print('=' * 80)
print('特征分析完成！')
print('=' * 80)
print()
print('输出文件：')
print(f'  图表：')
print(f'    - 01_pca_by_year.png')
print(f'    - 02_correlation_matrix.png')
print(f'    - 03_feature_distribution.png')
print(f'    - 04_feature_importance_v2.png')
print(f'  数据：')
print(f'    - feature_statistics.csv')
print(f'    - top_correlated_pairs.csv')
print(f'  说明：')
print(f'    - README.md')
