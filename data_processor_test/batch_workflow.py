import os
import time
import pandas as pd
from glob import glob

# 1. 批量自动打开所有选手页面

def open_all_players(csv_path, sleep_sec=2):
    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        print(f"Opening {row['name']} ...")
        os.system(f'start msedge "{row["url"]}"')
        time.sleep(sleep_sec)
    print("请在浏览器中用插件手动保存所有页面 body HTML 到同一文件夹，如 E:/下载")

# 2. 批量处理所有 HTML 文件

def process_all_html(html_dir, processor_script="data_processor.py"):
    html_files = glob(os.path.join(html_dir, "*.html"))
    for html_file in html_files:
        print(f"Processing {html_file} ...")
        os.system(f'python {processor_script} "{html_file}"')
    print("全部处理完成！")

if __name__ == "__main__":
    # 步骤1：批量打开页面
    open_all_players("players.csv", sleep_sec=2)
    input("\n请用插件手动保存所有页面 body HTML 后，按回车继续...\n")
    # 步骤2：批量处理 HTML
    process_all_html(r"E:/下载")
