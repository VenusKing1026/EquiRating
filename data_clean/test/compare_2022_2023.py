import pandas as pd

# 读取合并后的数据
df = pd.read_csv(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\data_clean\test\merged_all_players_all_years.csv')

# 筛选 s1mple 的 2022 和 2023 年数据
s1mple_2022 = df[(df['player'] == 's1mple') & (df['year'] == 2022)].iloc[0]
s1mple_2023 = df[(df['player'] == 's1mple') & (df['year'] == 2023)].iloc[0]

# 对比关键指标
metrics = [
    'rating_2.0_both', 'kpr_traditional', 'adr_traditional', 
    'kast_traditional', 'dpr_traditional', 'kills_per_round_both',
    'damage_per_round_both', 'opening_kills_per_round_both'
]

print("=" * 60)
print("s1mple 2022 vs 2023 数据对比")
print("=" * 60)
print(f"{'指标':<35} {'2022':<12} {'2023':<12} {'变化':<12}")
print("-" * 60)

for metric in metrics:
    val_2022 = s1mple_2022[metric] if metric in s1mple_2022 else 'N/A'
    val_2023 = s1mple_2023[metric] if metric in s1mple_2023 else 'N/A'
    
    if val_2022 != 'N/A' and val_2023 != 'N/A' and pd.notna(val_2022) and pd.notna(val_2023):
        change = ((val_2023 - val_2022) / val_2022 * 100) if val_2022 != 0 else 0
        change_str = f"{change:+.2f}%"
    else:
        change_str = 'N/A'
    
    print(f"{metric:<35} {str(val_2022):<12} {str(val_2023):<12} {change_str:<12}")

print("\n" + "=" * 60)
print("列数统计:")
print(f"2022 年总列数：{len(s1mple_2022)}")
print(f"2023 年总列数：{len(s1mple_2023)}")

# 检查缺失值
missing_2022 = s1mple_2022.isna().sum()
missing_2023 = s1mple_2023.isna().sum()
print(f"\n2022 年缺失值：{missing_2022}")
print(f"2023 年缺失值：{missing_2023}")

# 找出缺失的字段
missing_cols_2022 = [col for col in s1mple_2022.index if pd.isna(s1mple_2022[col])]
missing_cols_2023 = [col for col in s1mple_2023.index if pd.isna(s1mple_2023[col])]

print(f"\n2022 年缺失字段：{missing_cols_2022}")
print(f"2023 年缺失字段：{missing_cols_2023}")
