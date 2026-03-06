# EquiRating 特征分析

## 特征概览

- **总特征数**: 118 维
- **核心特征**: 14 维
- **样本数**: 200
- **年份范围**: 2016 - 2025

## 图表

1. `01_pca_by_year.png`: PCA 降维可视化（按年份着色）
2. `02_correlation_matrix.png`: 特征相关性矩阵
3. `03_feature_distribution.png`: 特征分布箱线图
4. `04_feature_importance_v2.png`: V2 模型特征重要性 Top20

## 数据文件

- `feature_statistics.csv`: 特征统计（均值、标准差、范围）
- `top_correlated_pairs.csv`: Top 50 高相关特征对

## 关键发现

1. PCA 显示不同年份的选手特征有明显聚类趋势
2. 击杀相关特征之间高度相关
3. 14 核心特征在 V2 模型中占据重要位置


## 聚类分析

### 图表

5. `05_correlation_clustermap.png`: 聚类热图（完整）
6. `06_correlation_clusters.png`: 分簇热图（带簇标签）
7. `07_dendrogram.png`: 层次聚类树状图

### 数据文件

- `feature_clusters.csv`: 特征聚类分组结果
- `cluster_statistics.csv`: 簇统计分析

### 关键发现

1. 特征可分为 8 个主要簇
2. 狙击相关特征高度聚集在同一簇
3. 击杀/伤害相关特征形成独立簇
