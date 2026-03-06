# EquiRating 统一评估结果

## 评估说明

- **评估方式**: Leave-One-Year-Out Cross-Validation (LOOCV)
- **统一指标**: Spearman, Kendall Tau, Top5 Overlap, Top1 Accuracy
- **所有版本使用相同的评估流程，确保可比性**

## 模型对比

| 模型 | 特征数 | Spearman | Kendall Tau | Top5 Overlap | Top1 Accuracy |
|------|--------|----------|-------------|--------------|---------------|
| V1 | 118 | 0.5708 | 0.4211 | 71.1% | 44.4% |
| V2 | 118 | 0.7109 | 0.5591 | 77.8% | 55.6% |
| V3 | 32 | 0.7263 | 0.5789 | 77.8% | 55.6% |

## 图表

1. `01_spearman_comparison.png`: Spearman 对比
2. `02_kendall_comparison.png`: Kendall Tau 对比
3. `03_top5_comparison.png`: Top5 Overlap 对比
4. `04_pred_vs_actual_all.png`: 预测 - 真实排名对比
5. `05_spearman_by_year.png`: Spearman 按年份对比

## 数据文件

- `summary.csv`: 汇总结果
- `yearly_metrics.csv`: 年度详细指标

## 结论

**最佳模型**: V3 (Spearman = 0.7263)
