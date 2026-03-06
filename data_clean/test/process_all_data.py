import pandas as pd
import numpy as np
import re
import json
from pathlib import Path


def time_to_seconds(time_str):
    """将时间字符串（如'1m 9s'）转换为秒数"""
    if pd.isna(time_str) or time_str == '':
        return np.nan
    
    time_str = str(time_str).strip()
    seconds = 0
    
    # 匹配分钟
    m_match = re.search(r'(\d+)m', time_str)
    if m_match:
        seconds += int(m_match.group(1)) * 60
    
    # 匹配秒
    s_match = re.search(r'(\d+)s', time_str)
    if s_match:
        seconds += int(s_match.group(1))
    
    return float(seconds)


def convert_to_float(value):
    """将值转换为float，处理百分号"""
    if pd.isna(value) or value == '':
        return np.nan
    
    value_str = str(value).strip()
    
    # 移除百分号
    if '%' in value_str:
        value_str = value_str.replace('%', '')
    
    # 移除+号
    if '+' in value_str:
        value_str = value_str.replace('+', '')
    
    try:
        return float(value_str)
    except:
        return np.nan


def process_role_stats(role_stats_path):
    """
    处理role_stats.csv，将每一行拆成both、ct、t三种
    返回一个字典，键为"metric_side"格式
    """
    df = pd.read_csv(role_stats_path)
    result = {}
    
    for _, row in df.iterrows():
        metric = row['metric']
        
        for side in ['Both', 'CT', 'T']:
            value = row[side]
            
            # 特殊处理Time alive per round
            if 'time alive per round' in metric.lower():
                value = time_to_seconds(value)
            else:
                value = convert_to_float(value)
            
            # 列名格式：metric_side（转为小写并替换空格为下划线）
            col_name = f"{metric}_{side}".replace(' ', '_').lower()
            result[col_name] = value
    
    return result


def process_right_bottom(right_bottom_path):
    """
    处理right_bottom.csv
    将第一列和第二列拼接作为列名，第三列作为值
    如果第四列是%，说明数据是百分比形式
    """
    df = pd.read_csv(right_bottom_path)
    result = {}
    
    for _, row in df.iterrows():
        metric = str(row['metric'])
        variant = str(row['variant']) if pd.notna(row['variant']) else ''
        value = row['value']
        unit = str(row['unit']) if pd.notna(row['unit']) else ''
        
        # 拼接列名：metric_variant
        if variant and variant != 'nan':
            col_name = f"{metric}_{variant}".replace(' ', '_').lower()
        else:
            col_name = metric.replace(' ', '_').lower()
        
        # 转换值
        value = convert_to_float(value)
        
        result[col_name] = value
    
    return result


def count_missing_values(data_dict):
    """统计缺失数据的个数"""
    missing_count = sum(1 for v in data_dict.values() if pd.isna(v))
    total_count = len(data_dict)
    return missing_count, total_count


def merge_player_data(player_folder_path, year):
    """
    合并一个选手的数据
    返回：合并后的数据字典和缺失统计
    """
    player_folder = Path(player_folder_path)
    player_name = player_folder.name
    
    # 读取meta.json
    meta_path = player_folder / 'meta.json'
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # 处理role_stats.csv
    role_stats_path = player_folder / 'role_stats.csv'
    role_stats_data = process_role_stats(role_stats_path)
    
    # 处理right_bottom.csv
    right_bottom_path = player_folder / 'right_bottom.csv'
    right_bottom_data = process_right_bottom(right_bottom_path)
    
    # 合并所有数据
    merged_data = {
        'player': player_name,
        'year': year,
        **role_stats_data,
        **right_bottom_data
    }
    
    # 统计缺失值
    missing_count, total_count = count_missing_values(merged_data)
    
    return merged_data, missing_count, total_count


def process_all_years(base_path):
    """
    处理所有年份的所有选手数据
    """
    base_path = Path(base_path)
    all_data = []
    missing_stats = []
    
    # 遍历所有年份文件夹
    for year_folder in sorted(base_path.iterdir()):
        if not year_folder.is_dir():
            continue
        
        year = year_folder.name
        processed_data_path = year_folder / 'processed_data'
        
        if not processed_data_path.exists():
            continue
        
        print(f"\n处理 {year} 年数据...")
        print("=" * 60)
        
        # 遍历所有选手文件夹
        for player_folder in sorted(processed_data_path.iterdir()):
            if not player_folder.is_dir():
                continue
            
            player_name = player_folder.name
            
            # 检查是否有必要的文件
            if not (player_folder / 'role_stats.csv').exists() or \
               not (player_folder / 'right_bottom.csv').exists():
                print(f"  跳过 {player_name}: 缺少必要文件")
                continue
            
            try:
                # 合并数据
                merged_data, missing_count, total_count = merge_player_data(player_folder, year)
                all_data.append(merged_data)
                
                # 记录缺失统计
                missing_stats.append({
                    'player': player_name,
                    'year': year,
                    'total_fields': total_count,
                    'missing_fields': missing_count,
                    'missing_rate': f"{missing_count/total_count*100:.2f}%"
                })
                
                print(f"  ✓ {player_name}: {total_count}个字段, {missing_count}个缺失 ({missing_count/total_count*100:.2f}%)")
            
            except Exception as e:
                print(f"  ✗ {player_name}: 处理失败 - {e}")
    
    return all_data, missing_stats


def main():
    # 数据路径
    data_base_path = r"E:\OneDrive - International Campus, Zhejiang University\EquiRating\data_processor_formal\data"
    output_folder = Path(__file__).parent
    
    print("开始处理所有选手数据...")
    print("=" * 60)
    
    # 处理所有数据
    all_data, missing_stats = process_all_years(data_base_path)
    
    if not all_data:
        print("\n没有找到可处理的数据！")
        return
    
    # 保存合并后的所有数据
    df_all = pd.DataFrame(all_data)
    output_all_path = output_folder / 'merged_all_players_all_years.csv'
    df_all.to_csv(output_all_path, index=False, encoding='utf-8')
    
    print(f"\n" + "=" * 60)
    print(f"所有数据合并完成！")
    print(f"  总共处理: {len(all_data)} 条记录")
    print(f"  总列数: {len(df_all.columns)}")
    print(f"  保存到: {output_all_path}")
    
    # 保存缺失统计
    df_missing = pd.DataFrame(missing_stats)
    output_missing_path = output_folder / 'missing_data_statistics.csv'
    df_missing.to_csv(output_missing_path, index=False, encoding='utf-8')
    
    print(f"\n缺失数据统计已保存到: {output_missing_path}")
    print(f"\n缺失数据汇总:")
    print(df_missing.to_string(index=False))
    
    # 打印按选手汇总的统计
    print(f"\n按选手统计:")
    player_stats = df_missing.groupby('player').agg({
        'missing_fields': 'sum',
        'total_fields': 'sum'
    })
    player_stats['missing_rate'] = (player_stats['missing_fields'] / player_stats['total_fields'] * 100).round(2).astype(str) + '%'
    print(player_stats.to_string())


if __name__ == "__main__":
    main()
