import pandas as pd
import os

# 创建目录
output_dir = r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\final_data'
os.makedirs(output_dir, exist_ok=True)

# 读取原始数据（带所有 Rating 字段）
df_raw = pd.read_csv(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\data_clean\test\merged_all_players_all_years.csv')

# 保存完整数据
all_data_path = os.path.join(output_dir, 'all_data.csv')
df_raw.to_csv(all_data_path, index=False, encoding='utf-8')
print(f'[1] 完整数据已保存：{all_data_path}')
print(f'    数据：{len(df_raw)}行 × {len(df_raw.columns)}列')
print()

# 要删除的字段（ecoAdjusted 和多余的 Rating）
eco_cols = ['dpr_ecoadjusted', 'kast_ecoadjusted', 'multi-kill_traditional', 
            'mk_rating_ecoadjusted', 'adr_ecoadjusted', 'kpr_ecoadjusted', 
            'round_swing_single']

# 删除多余的 Rating 字段（保留 rating_2.0_both 和 rating_3.0_both）
rating_cols_to_drop = [c for c in df_raw.columns if 'rating' in c.lower()]
rating_cols_to_keep = ['rating_2.0_both', 'rating_3.0_both']
rating_cols_to_drop = [c for c in rating_cols_to_drop if c not in rating_cols_to_keep]

cols_to_drop = rating_cols_to_drop + eco_cols

# 清理后的数据
df_clean = df_raw.drop(columns=cols_to_drop)

# 保存清理后的数据
cleaned_data_path = os.path.join(output_dir, 'cleaned_data.csv')
df_clean.to_csv(cleaned_data_path, index=False, encoding='utf-8')
print(f'[2] 清理后数据已保存：{cleaned_data_path}')
print(f'    数据：{len(df_clean)}行 × {len(df_clean.columns)}列')
print()

# 统计信息
print('=' * 80)
print('数据摘要')
print('=' * 80)
print(f'删除字段数：{len(cols_to_drop)}')
print(f'  - Rating 字段：{len(rating_cols_to_drop)}')
print(f'  - ecoAdjusted 字段：{len(eco_cols)}')
print()
print(f'保留的 Rating 字段：{rating_cols_to_keep}')
print()
print(f'最终特征数：{len(df_clean.columns)}')
print(f'  - 基础信息：player, year, rank')
print(f'  - 保留 Rating：{len(rating_cols_to_keep)}')
print(f'  - 其他特征：{len(df_clean.columns) - 3 - len(rating_cols_to_keep)}')
print()

# 验证数据
print('=' * 80)
print('数据验证')
print('=' * 80)
print(f'选手数：{df_clean["player"].nunique()}')
print(f'年份范围：{df_clean["year"].min()} - {df_clean["year"].max()}')
print(f'Rank 缺失值：{df_clean["rank"].isna().sum()}')
print()
print('保存完成！')
