# EquiRating V2 改进版 - Pairwise with Feature Difference

## 改进点

1. **特征表示**: 从 `[A_features, B_features]` 改为 `A_features - B_features`（差值）
2. **Ranking Reconstruction**: 从配对预测反推年度排名
3. **新增指标**: Top1 Accuracy, Top5 Overlap, Spearman, Kendall Tau

## 结果对比

| 实验 | 特征 | Pair R² | Pair Acc | Spearman | Kendall | Top1 | Top5 |
|------|------|---------|----------|----------|---------|------|------|
| V2.1 | 118 | 0.6074 | 73.95% | 0.6558 | 0.5053 | 60.00% | 74.00% |
| V2.2 | 14 | 0.5323 | 73.32% | 0.6335 | 0.4874 | 60.00% | 74.00% |

**最佳模型**: V2.1

## 文件列表

- `pairwise_v2_diff.model`: 最佳模型
- `predictions_v2.1_diff.csv`, `predictions_v2.2_diff.csv`: 预测结果
- `feature_importance_v2_diff.csv`: 特征重要性
- `yearly_metrics_v2_diff.csv`: 年度指标
- `v2_diff_comparison.png`: 综合指标对比
- `v2_diff_feature_importance.png`: 特征重要性图
- `train_v2_pairwise_v2.py`: 训练脚本
