# EquiRating V3 - 33 特征精简版本

## 特征选择策略

1. **保留所有 _both 特征**: 118 → 42
2. **删除 traditional 重复**: 42 → 38
3. **删除狙击 weapon bias**: 38 → 33

## 33 特征组成

- **18 核心特征**: kills_per_round_both, damage_per_round_both, damage_per_kill_both, rounds_with_a_multi-kill_both, opening_kills_per_round_both...
- **15 补充特征**: attacks_per_round_both, flashes_thrown_per_round_both...

## 设计思路

1. **Label**: V1 的对数转换 y = -log(rank)
2. **策略**: V2 的 Pairwise 差值表示
3. **模型**: XGBoost 回归

## 评估结果

| 指标 | 值 |
|------|-----|
| features | 32 |
| pairwise_mae | 0.5881832208841795 |
| pairwise_r2 | 0.5692670958804836 |
| pairwise_accuracy | 0.7357894736842105 |
| spearman | 0.6657142857142856 |
| kendall_tau | 0.5189473684210526 |
| top5_overlap | 0.72 |
| top1_accuracy | 0.6 |

## 图表

1. `01_spearman.png`: Spearman 相关系数（按年份）
2. `02_kendall.png`: Kendall Tau（按年份）
3. `03_top5.png`: Top5 Overlap（按年份）
4. `04_pred_vs_actual.png`: 预测 - 真实排名
5. `05_feature_importance.png`: 特征重要性

## 文件列表

- `pairwise_v3_33features.model`: 训练好的模型
- `predictions_v3_33features.csv`: 配对预测结果
- `yearly_metrics_v3_33features.csv`: 年度指标
- `summary_v3_33features.csv`: 汇总结果
- `feature_importance_v3_33features.csv`: 特征重要性
- `train_v3_complete.py`: 训练脚本
