import pandas as pd

df = pd.read_csv(r'E:\OneDrive - International Campus, Zhejiang University\EquiRating\data_clean\test\merged_all_players_all_years.csv')

# 基础信息列
base_cols = ['player', 'year', 'rank']

# 所有 rating 相关列（要删除的）
rating_cols = [c for c in df.columns if 'rating' in c.lower()]

# 所有列
all_cols = list(df.columns)

# 删除 rating 后的特征
feature_cols = [c for c in all_cols if c not in rating_cols]

print('=' * 80)
print(f'总列数：{len(all_cols)}')
print(f'基础信息列：{len(base_cols)}')
print(f'Rating 相关列（要删除）：{len(rating_cols)}')
print(f'删除后特征数：{len(feature_cols)}')
print('=' * 80)

print('\n【要删除的 Rating 字段】')
for col in sorted(rating_cols):
    print(f'  - {col}')

print('\n【保留的特征字段】')
for i, col in enumerate(feature_cols, 1):
    marker = '← 基础信息' if col in base_cols else ''
    print(f'{i:3d}. {col} {marker}')

print('\n' + '=' * 80)
print('特征分类统计：')
print('=' * 80)

# 按前缀分类
prefixes = {}
for col in feature_cols:
    if col in base_cols:
        continue
    # 提取前缀
    parts = col.rsplit('_', 1)
    if len(parts) == 2:
        prefix = parts[0]
    else:
        prefix = col
    prefixes[prefix] = prefixes.get(prefix, 0) + 1

print(f'\n特征前缀统计（按指标类型）：')
for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1])[:30]:
    print(f'  {prefix}: {count}个')
