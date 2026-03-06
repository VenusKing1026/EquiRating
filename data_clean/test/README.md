# 数据处理测试结果

## 完成任务

### 1. 缺失数据统计
- 已统计每个选手的缺失数据个数
- 结果保存在: `missing_data_statistics.csv`
- 本次测试：2025年20个选手，所有数据完整，无缺失值

### 2. 数据合并
- 已将每个选手的 `role_stats.csv` 和 `right_bottom.csv` 合并为一个向量（一行）
- 已将所有年份的数据合并到一个CSV文件
- 结果保存在: `merged_all_players_all_years.csv`

## 数据处理规则

### role_stats.csv 处理
- 每一行指标拆分为 `Both`、`CT`、`T` 三列
- 列名格式: `metric_side`（如 `rating_3.0_both`, `rating_3.0_ct`, `rating_3.0_t`）
- **Time alive per round** 转换为秒数（如 `1m 9s` → `69.0`）
- 所有数值转换为 float

### right_bottom.csv 处理
- 第一列（metric）和第二列（variant）拼接成列名
- 列名格式: `metric_variant`（如 `dpr_traditional`, `kast_ecoadjusted`）
- 第三列（value）作为数值
- 第四列如果是 `%`，自动移除百分号并转换
- 自动处理 `+` 号（如 `+3.40%` → `3.4`）

## 输出文件

### merged_all_players_all_years.csv
- 每行代表一个选手某一年的完整数据
- 共 133 列（player + year + 131个指标）
- 当前包含：20个选手 × 1年 = 20行数据

### missing_data_statistics.csv
- 包含每个选手每年的缺失数据统计
- 列：player, year, total_fields, missing_fields, missing_rate

## 测试数据
- 测试选手: donk
- 测试年份: 2025
- 数据来源: `data_processor_formal/data/2025/processed_data/donk`

## 使用方法

运行完整处理脚本：
```bash
python process_all_data.py
```

运行单个选手测试：
```bash
python data_merger.py
```
