import pandas as pd

df = pd.read_csv(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\data_clean\test\merged_all_players_all_years.csv')
df_2024 = df[df['year'] == 2024]

print('=== 2024 年 Rating 对比 ===')
print(f'选手数：{len(df_2024)}')
print()

print('=== 所有列名 ===')
rating_cols = [c for c in df_2024.columns if 'rating' in c.lower()]
print(rating_cols)
print()

cols = ['rating_2.0_both', 'rating_3.0_both', 'rating_2.0_ct', 'rating_3.0_ct', 'rating_2.0_t', 'rating_3.0_t']
for col in cols:
    if col in df_2024.columns:
        mean_val = df_2024[col].mean()
        median_val = df_2024[col].median()
        print(f'{col}: 均值={mean_val:.3f}, 中位数={median_val:.3f}')

print()
print('=== 前 5 名选手对比 ===')
print(df_2024[['player', 'rank', 'rating_2.0_both', 'rating_3.0_both']].head())

print()
print('=== 相关性 ===')
corr = df_2024['rating_2.0_both'].corr(df_2024['rating_3.0_both'])
print(f'2.0 vs 3.0 both: {corr:.4f}')

print()
print('=== 2024 rating_1.0 数据 ===')
cols_1_0 = ['rating_1.0_both', 'rating_1.0_ct', 'rating_1.0_t']
for col in cols_1_0:
    if col in df_2024.columns:
        mean_val = df_2024[col].mean()
        median_val = df_2024[col].median()
        print(f'{col}: 均值={mean_val:.3f}, 中位数={median_val:.3f}')

print()
print('=== 检查所有年份 rating_1.0 ===')
for year in sorted(df['year'].unique()):
    df_year = df[df['year'] == year]
    if 'rating_1.0_both' in df_year.columns:
        mean_val = df_year['rating_1.0_both'].mean()
        print(f'{year}年：{len(df_year)}人，rating_1.0_both 均值={mean_val:.3f}')
    else:
        print(f'{year}年：{len(df_year)}人，无 rating_1.0_both 字段')
