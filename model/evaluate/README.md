# EquiRating 统一评估结果

## 主指标

1. **Spearman Correlation**: 排名相关性（越强越好）
2. **Kendall Tau**: 排序一致性（越强越好）
3. **Top5 Overlap**: Top5 重合度（越高越好）

## 对比结果

| 模型 | 特征 | Spearman | Kendall Tau | Top5 Overlap |
|------|------|----------|-------------|--------------|
| **V1.1.2** | 118 | 0.5613 | 0.4281 | 66.0% |
| **V2.1** | 118 (差值) | **0.6558** | **0.5053** | **74.0%** |
| V2.2 | 14 (差值) | 0.6335 | 0.4874 | 74.0% |

## 结论

- **最佳模型**: V2.1（XGBoost + 118 特征差值）
- **V2 优于 V1**: 所有三个主指标都更高
- **14 特征表现接近**: V2.2 在 Top5 Overlap 上与 V2.1 持平

## 图表

1. `01_spearman_comparison.png`: Spearman 对比
2. `02_kendall_comparison.png`: Kendall Tau 对比
3. `03_top5_overlap_comparison.png`: Top5 Overlap 对比
4. `04_pred_vs_actual_all.png`: 预测 - 真实排名对比

## 文件列表

- `summary.csv`: 汇总对比表
- `01_spearman_comparison.png` ~ `04_pred_vs_actual_all.png`: 4 个标准图表
- `evaluate_all.py`: 统一评估脚本
