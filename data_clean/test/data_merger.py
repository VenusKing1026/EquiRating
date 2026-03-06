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


def merge_player_data(player_folder_path):
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
        **role_stats_data,
        **right_bottom_data
    }
    
    # 统计缺失值
    missing_count, total_count = count_missing_values(merged_data)
    
    return merged_data, missing_count, total_count


def main():
    # 测试路径
    test_player_path = r"E:\OneDrive - International Campus, Zhejiang University\EquiRating\data_processor_formal\data\2025\processed_data\donk"
    
    print(f"处理玩家数据: {test_player_path}")
    print("=" * 60)
    
    # 合并数据
    merged_data, missing_count, total_count = merge_player_data(test_player_path)
    
    # 打印缺失统计
    print(f"\n缺失数据统计:")
    print(f"  总字段数: {total_count}")
    print(f"  缺失字段数: {missing_count}")
    print(f"  缺失率: {missing_count/total_count*100:.2f}%")
    
    # 将数据转换为DataFrame
    df = pd.DataFrame([merged_data])
    
    # 保存到test文件夹
    output_path = Path(__file__).parent / 'merged_donk_2025.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n合并后的数据已保存到: {output_path}")
    print(f"合并后共有 {len(df.columns)} 列")
    
    # 打印前几列预览
    print(f"\n前10列预览:")
    print(df.iloc[:, :10].to_string())
    
    # 检查哪些字段有缺失值
    missing_fields = [col for col in df.columns if pd.isna(df[col].iloc[0])]
    if missing_fields:
        print(f"\n缺失值字段列表:")
        for field in missing_fields[:20]:  # 只显示前20个
            print(f"  - {field}")
        if len(missing_fields) > 20:
            print(f"  ... 还有 {len(missing_fields) - 20} 个字段")


if __name__ == "__main__":
    main()
