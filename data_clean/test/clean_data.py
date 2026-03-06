import pandas as pd
import numpy as np

# 读取数据（已更新 rank 的版本）
df = pd.read_csv(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\data_clean\test\merged_all_players_all_years.csv')

print('=' * 80)
print('原始数据统计')
print('=' * 80)
print(f'行数：{len(df)}')
print(f'列数：{len(df.columns)}')
print()

# 要删除的字段
rating_cols = [c for c in df.columns if 'rating' in c.lower()]
eco_cols = ['dpr_ecoadjusted', 'kast_ecoadjusted', 'multi-kill_traditional', 
            'mk_rating_ecoadjusted', 'adr_ecoadjusted', 'kpr_ecoadjusted', 
            'round_swing_single']

cols_to_drop = rating_cols + eco_cols
print(f'【要删除的字段】共 {len(cols_to_drop)} 个')
print(f'  Rating 字段：{len(rating_cols)}')
print(f'  ecoAdjusted 字段：{len(eco_cols)}')

# 删除字段
df_clean = df.drop(columns=cols_to_drop)
print(f'\n删除后列数：{len(df_clean.columns)}')
print()

# 检查缺失值
missing_rank = df_clean['rank'].isna().sum()
print(f'【Rank 缺失检查】')
if missing_rank > 0:
    print(f'  [WARN] 仍有 {missing_rank} 个 rank 缺失')
    missing_rows = df_clean[df_clean['rank'].isna()]
    print(f'  缺失行：')
    print(missing_rows[['player', 'year', 'rank']].to_string(index=False))
else:
    print(f'  [OK] 所有 rank 已填充')
print()

# 保存清理后的数据
output_path = r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\data_clean\test\merged_cleaned.csv'
df_clean.to_csv(output_path, index=False, encoding='utf-8')

print(f'【保存清理后的数据】')
print(f'  路径：{output_path}')
print(f'  数据：{len(df_clean)}行 × {len(df_clean.columns)}列')
print()

# 最终统计
print('=' * 80)
print('最终数据统计')
print('=' * 80)
print(f'选手数：{df_clean["player"].nunique()}')
print(f'年份范围：{df_clean["year"].min()} - {df_clean["year"].max()}')
print(f'每年选手数：')
for year in sorted(df_clean['year'].unique()):
    count = len(df_clean[df_clean['year'] == year])
    print(f'  {year}年：{count}人')

print('\n清理完成！')
