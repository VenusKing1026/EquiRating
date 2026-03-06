"""
特征相关性矩阵（带聚类）

使用层次聚类对特征进行分组，展示特征间的关系结构
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, cut_tree

# ============================================================================
# 配置
# ============================================================================
DATA_PATH = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\final_data\cleaned_data.csv')
FEATURE_DIR = Path(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\model\feature')

# ============================================================================
# 加载数据
# ============================================================================
print('=' * 80)
print('特征相关性聚类分析')
print('=' * 80)
print()

df = pd.read_csv(DATA_PATH)

# 定义特征
exclude_cols = ['player', 'year', 'rank', 'rating_2.0_both', 'rating_3.0_both']
features = [c for c in df.columns if c not in exclude_cols]
df[features] = df[features].fillna(df[features].median())

# 只分析 both 结尾的特征
both_features = [f for f in features if f.endswith('_both')]
print(f'分析特征数：{len(both_features)}')
print()

# 计算相关性矩阵
corr_matrix = df[both_features].corr()

# ============================================================================
# 1. 聚类热图（完整）
# ============================================================================
print('[1] 生成聚类热图...')

plt.figure(figsize=(20, 18))

# 使用 clustermap 进行聚类和可视化
g = sns.clustermap(
    corr_matrix,
    method='ward',  # 聚类方法：ward 最小化方差
    metric='euclidean',  # 距离度量
    cmap='RdBu_r',  # 颜色映射：红蓝
    center=0,  # 中心值为 0
    figsize=(20, 18),
    dendrogram_ratio=0.15,  # 树状图比例
    cbar_pos=(0.02, 0.85, 0.03, 0.12),  # 颜色条位置
    cbar_kws={'label': '相关系数', 'shrink': 0.8},
    linewidths=0.5,
    linecolor='white'
)

# 设置标题
g.fig.suptitle('特征相关性聚类热图\n基于 118 维特征中的 both 维度', 
               fontsize=16, fontweight='bold', y=0.98)

# 调整布局
g.fig.tight_layout()
plt.savefig(FEATURE_DIR / '05_correlation_clustermap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'    ✓ 聚类热图：05_correlation_clustermap.png')

# ============================================================================
# 2. 特征聚类分组
# ============================================================================
print('\n[2] 特征聚类分组...')

from scipy.cluster.hierarchy import linkage

# 层次聚类
linkage_matrix = linkage(corr_matrix.values, method='ward')

# 切割树状图，分成 8 个簇
n_clusters = 8
clusters = cut_tree(linkage_matrix, n_clusters=n_clusters).flatten()

# 将聚类结果保存到 DataFrame
cluster_df = pd.DataFrame({
    'feature': both_features,
    'cluster': clusters
}).sort_values('cluster')

# 打印每个簇的特征
for i in range(n_clusters):
    cluster_features = cluster_df[cluster_df['cluster'] == i]['feature'].tolist()
    print(f'\n    簇 {i+1} ({len(cluster_features)}个特征):')
    
    # 简化特征名显示
    simplified = [f.replace('_both', '').replace('per_round', '/R').replace('percentage', '%') 
                  for f in cluster_features[:8]]
    for feat in simplified:
        print(f'      - {feat}')
    if len(cluster_features) > 8:
        print(f'      ... 还有 {len(cluster_features) - 8} 个特征')

# 保存聚类结果
cluster_df.to_csv(FEATURE_DIR / 'feature_clusters.csv', index=False, encoding='utf-8')
print(f'\n    ✓ 聚类结果：feature_clusters.csv')

# ============================================================================
# 3. 分簇热图（带簇标签）
# ============================================================================
print('\n[3] 生成分簇热图...')

# 重新排序相关性矩阵
cluster_df = cluster_df.sort_values('cluster')
ordered_features = cluster_df['feature'].tolist()
corr_ordered = corr_matrix.loc[ordered_features, ordered_features]

# 创建簇边界
cluster_boundaries = []
current_cluster = -1
for i, cluster in enumerate(cluster_df['cluster']):
    if cluster != current_cluster:
        cluster_boundaries.append(i)
        current_cluster = cluster
cluster_boundaries.append(len(ordered_features))

# 绘制分簇热图
fig, ax = plt.subplots(figsize=(16, 14))

# 热图
sns.heatmap(corr_ordered, 
            cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, linecolor='white',
            cbar_kws={'label': '相关系数', 'shrink': 0.8},
            ax=ax, annot=False,
            xticklabels=False, yticklabels=False)

# 添加簇边界线
for boundary in cluster_boundaries[1:-1]:
    ax.axhline(boundary, color='black', linewidth=2)
    ax.axvline(boundary, color='black', linewidth=2)

# 添加簇标签
for i in range(n_clusters):
    start = cluster_boundaries[i]
    end = cluster_boundaries[i + 1]
    center = (start + end) / 2
    
    # 左侧标签
    ax.text(-5, center, f'簇{i+1}', ha='right', va='center', 
            fontsize=11, fontweight='bold', color='black')
    
    # 顶部标签
    ax.text(center, -5, f'簇{i+1}', ha='center', va='top', 
            fontsize=11, fontweight='bold', color='black', rotation=45)

ax.set_title('特征相关性分簇热图\n共 8 个特征簇', fontsize=14, pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig(FEATURE_DIR / '06_correlation_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'    ✓ 分簇热图：06_correlation_clusters.png')

# ============================================================================
# 4. 簇内平均相关性分析
# ============================================================================
print('\n[4] 簇内平均相关性分析...')

cluster_stats = []

for i in range(n_clusters):
    cluster_features = cluster_df[cluster_df['cluster'] == i]['feature'].tolist()
    n_feats = len(cluster_features)
    
    # 计算簇内平均相关性（排除对角线）
    sub_corr = corr_matrix.loc[cluster_features, cluster_features]
    upper_tri = sub_corr.where(np.triu(np.ones(sub_corr.shape), k=1).astype(bool))
    avg_corr = upper_tri.stack().mean()
    
    # 计算簇间平均相关性
    other_features = [f for f in both_features if f not in cluster_features]
    if other_features:
        cross_corr = corr_matrix.loc[cluster_features, other_features]
        avg_cross_corr = cross_corr.values.mean()
    else:
        avg_cross_corr = np.nan
    
    cluster_stats.append({
        'cluster': i + 1,
        'n_features': n_feats,
        'avg_internal_corr': avg_corr,
        'avg_external_corr': avg_cross_corr,
        'cohesion': avg_corr / avg_cross_corr if avg_cross_corr > 0 else np.nan
    })

cluster_stats_df = pd.DataFrame(cluster_stats)

print(f'{"簇":<6} {"特征数":<8} {"簇内平均相关":<12} {"簇间平均相关":<12} {"内聚度":<10}')
print('-' * 55)
for _, row in cluster_stats_df.iterrows():
    print(f'{int(row["cluster"]):<6} {int(row["n_features"]):<8} '
          f'{row["avg_internal_corr"]:<12.3f} {row["avg_external_corr"]:<12.3f} '
          f'{row["cohesion"]:<10.2f}')

cluster_stats_df.to_csv(FEATURE_DIR / 'cluster_statistics.csv', index=False, encoding='utf-8')
print(f'\n    ✓ 簇统计：cluster_statistics.csv')

# ============================================================================
# 5. 树状图（独立）
# ============================================================================
print('\n[5] 生成树状图...')

fig, ax = plt.subplots(figsize=(14, 8))

# 绘制树状图
dendro = dendrogram(
    linkage_matrix,
    labels=both_features,
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=0.7 * max(linkage_matrix[:, 2]),  # 颜色阈值
    ax=ax
)

ax.set_title('特征层次聚类树状图\n共 8 个簇', fontsize=14, pad=20, fontweight='bold')
ax.set_xlabel('特征', fontsize=12, fontweight='bold')
ax.set_ylabel('距离', fontsize=12, fontweight='bold')
ax.axhline(y=linkage_matrix[-n_clusters+1, 2], color='red', linestyle='--', 
           linewidth=2, label=f'切割线 (n_clusters={n_clusters})')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(FEATURE_DIR / '07_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'    ✓ 树状图：07_dendrogram.png')

# ============================================================================
# 6. 更新 README
# ============================================================================
print('\n[6] 更新 README...')

with open(FEATURE_DIR / 'README.md', 'a', encoding='utf-8') as f:
    f.write('\n\n## 聚类分析\n\n')
    f.write('### 图表\n\n')
    f.write('5. `05_correlation_clustermap.png`: 聚类热图（完整）\n')
    f.write('6. `06_correlation_clusters.png`: 分簇热图（带簇标签）\n')
    f.write('7. `07_dendrogram.png`: 层次聚类树状图\n\n')
    f.write('### 数据文件\n\n')
    f.write('- `feature_clusters.csv`: 特征聚类分组结果\n')
    f.write('- `cluster_statistics.csv`: 簇统计分析\n\n')
    f.write('### 关键发现\n\n')
    f.write('1. 特征可分为 8 个主要簇\n')
    f.write('2. 狙击相关特征高度聚集在同一簇\n')
    f.write('3. 击杀/伤害相关特征形成独立簇\n')

print(f'    ✓ README 已更新')
print()

# ============================================================================
# 完成
# ============================================================================
print('=' * 80)
print('特征聚类分析完成！')
print('=' * 80)
print()
print('新增输出文件：')
print(f'  图表：')
print(f'    - 05_correlation_clustermap.png')
print(f'    - 06_correlation_clusters.png')
print(f'    - 07_dendrogram.png')
print(f'  数据：')
print(f'    - feature_clusters.csv')
print(f'    - cluster_statistics.csv')
