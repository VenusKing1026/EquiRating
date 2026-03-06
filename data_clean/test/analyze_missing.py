import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\data_clean\test\merged_all_players_all_years.csv')

print('=' * 80)
print('原始数据统计')
print('=' * 80)
print(f'行数：{len(df)}')
print(f'列数：{len(df.columns)}')
print()

# ============================================================================
# 1. 要删除的字段
# ============================================================================
rating_cols = [c for c in df.columns if 'rating' in c.lower()]
eco_cols = ['dpr_ecoadjusted', 'kast_ecoadjusted', 'multi-kill_traditional', 
            'mk_rating_ecoadjusted', 'adr_ecoadjusted', 'kpr_ecoadjusted', 
            'round_swing_single']

cols_to_drop = rating_cols + eco_cols
print(f'【要删除的字段】共 {len(cols_to_drop)} 个')
print('Rating 字段:', len(rating_cols))
print('ecoAdjusted 字段:', len(eco_cols))
print()

# 删除字段
df_clean = df.drop(columns=cols_to_drop)
print(f'删除后列数：{len(df_clean.columns)}')
print()

# ============================================================================
# 2. 缺失值分析 - 列维度
# ============================================================================
print('=' * 80)
print('【缺失值分析 - 列维度】')
print('=' * 80)

missing_cols = df_clean.isna().sum()
missing_pct = (df_clean.isna().sum() / len(df_clean) * 100).round(2)
missing_df = pd.DataFrame({
    'missing_count': missing_cols,
    'missing_pct': missing_pct
}).sort_values('missing_count', ascending=False)

print(f'\n缺失值最多的前 20 列：')
print(missing_df.head(20).to_string())

print(f'\n缺失值统计：')
print(f'  有缺失的列数：{(missing_cols > 0).sum()}')
print(f'  无缺失的列数：{(missing_cols == 0).sum()}')
print(f'  最大缺失率：{missing_pct.max():.2f}%')
print(f'  平均缺失率：{missing_pct[missing_cols > 0].mean():.2f}%')

# 缺失严重的列（>5%）
high_missing = missing_df[missing_df['missing_pct'] > 5]
if len(high_missing) > 0:
    print(f'\n⚠️  缺失率 > 5% 的列 ({len(high_missing)}个):')
    print(high_missing.to_string())

# ============================================================================
# 3. 缺失值分析 - 行维度
# ============================================================================
print('\n' + '=' * 80)
print('【缺失值分析 - 行维度】')
print('=' * 80)

missing_rows = df_clean.isna().sum(axis=1)
missing_rows_pct = (df_clean.isna().sum(axis=1) / len(df_clean.columns) * 100).round(2)

print(f'\n缺失值最多的前 10 行：')
top_rows = pd.DataFrame({
    'player': df_clean['player'].values,
    'year': df_clean['year'].values,
    'rank': df_clean['rank'].values,
    'missing_count': missing_rows.values,
    'missing_pct': missing_rows_pct.values
}).sort_values('missing_count', ascending=False).head(10)
print(top_rows.to_string(index=False))

print(f'\n行缺失统计：')
print(f'  有缺失的行数：{(missing_rows > 0).sum()}')
print(f'  无缺失的行数：{(missing_rows == 0).sum()}')
print(f'  最大缺失率：{missing_rows_pct.max():.2f}%')

# ============================================================================
# 4. 异常值分析 - 全 0 列
# ============================================================================
print('\n' + '=' * 80)
print('【异常值分析 - 全 0 列】')
print('=' * 80)

# 数值列
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
all_zero_cols = []
near_zero_cols = []

for col in numeric_cols:
    if col in ['player', 'year', 'rank']:
        continue
    col_data = df_clean[col].dropna()
    if len(col_data) > 0:
        zero_count = (col_data == 0).sum()
        zero_pct = zero_count / len(col_data) * 100
        
        if zero_count == len(col_data):
            all_zero_cols.append(col)
        elif zero_pct > 90:
            near_zero_cols.append((col, zero_pct))

print(f'\n全 0 列 ({len(all_zero_cols)}个):')
if all_zero_cols:
    for col in all_zero_cols:
        print(f'  - {col}')
else:
    print('  无')

print(f'\n接近全 0 列 (>90% 为 0, {len(near_zero_cols)}个):')
if near_zero_cols:
    for col, pct in sorted(near_zero_cols, key=lambda x: -x[1])[:20]:
        print(f'  - {col}: {pct:.1f}%')
else:
    print('  无')

# ============================================================================
# 5. 异常值分析 - 全 0 行
# ============================================================================
print('\n' + '=' * 80)
print('【异常值分析 - 全 0 行】')
print('=' * 80)

# 检查每行的 0 值比例
numeric_data = df_clean[numeric_cols].drop(columns=['year', 'rank'], errors='ignore')
row_zero_counts = (numeric_data == 0).sum(axis=1)
row_zero_pcts = (row_zero_counts / len(numeric_cols) * 100).round(2)

print(f'\n0 值最多的前 10 行：')
top_zero_rows = pd.DataFrame({
    'player': df_clean['player'].values,
    'year': df_clean['year'].values,
    'rank': df_clean['rank'].values,
    'zero_count': row_zero_counts.values,
    'zero_pct': row_zero_pcts.values
}).sort_values('zero_count', ascending=False).head(10)
print(top_zero_rows.to_string(index=False))

# ============================================================================
# 6. 数值分布统计
# ============================================================================
print('\n' + '=' * 80)
print('【数值分布统计 - 关键指标】')
print('=' * 80)

key_metrics = [
    'kills_per_round_both', 'damage_per_round_both', 'adr_traditional',
    'kast_traditional', 'dpr_traditional', 'kpr_traditional',
    'opening_kills_per_round_both', '1on1_win_percentage_both'
]

print(f'\n关键指标统计：')
stats = df_clean[key_metrics].describe()
print(stats.round(3).to_string())

# ============================================================================
# 7. 保存清理后的数据
# ============================================================================
print('\n' + '=' * 80)
print('【保存清理后的数据】')
print('=' * 80)

output_path = r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\data_clean\test\merged_cleaned.csv'
df_clean.to_csv(output_path, index=False, encoding='utf-8')
print(f'已保存到：{output_path}')
print(f'最终数据：{len(df_clean)}行 × {len(df_clean.columns)}列')

print('\n' + '=' * 80)
print('分析完成！')
print('=' * 80)
