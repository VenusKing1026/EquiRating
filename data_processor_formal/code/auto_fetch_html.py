"""
自动抓取 HLTV 选手页面 HTML（无需插件）
使用 Playwright 模拟插件行为：保存 document.body.outerHTML
"""

import subprocess
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def build_parser():
    parser = argparse.ArgumentParser(description="Auto fetch HLTV player HTML using Playwright")
    parser.add_argument("--year", default=str(datetime.now().year), help="Data year, e.g. 2021")
    parser.add_argument("--sleep-sec", type=int, default=3, help="Sleep seconds between pages")
    return parser


def get_year_paths(year: str):
    year_dir = DATA_DIR / str(year)
    return {
        "year_dir": year_dir,
        "raw_html_dir": year_dir / "raw_html",
        "players_csv": year_dir / "players.csv",
    }


def fetch_all_players(csv_path, raw_html_dir, sleep_sec=3):
    """用 Playwright 打开每个选手页面并保存 HTML"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("错误：需要安装 playwright")
        print("运行：pip install playwright")
        print("然后：playwright install")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    with sync_playwright() as p:
        # 启动 Edge 浏览器
        browser = p.chromium.launch(
            headless=False,
            channel="msedge"  # 使用 Edge
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        
        for i, row in df.iterrows():
            if pd.isna(row.get("url")) or str(row.get("url", "")).strip() == "":
                print(f"Skipping {row['name']} (url 为空)")
                continue
            
            url = row["url"]
            player_name = row["name"]
            
            print(f"\n[{i+1}/{len(df)}] Fetching {player_name}...")
            
            try:
                # 打开页面
                page.goto(url, wait_until="networkidle", timeout=30000)
                
                # 等待页面加载
                time.sleep(2)
                
                # 获取 body HTML（和插件一样）
                body_html = page.evaluate("() => document.body.outerHTML")
                
                # 生成文件名（和插件逻辑一致）
                url_path = url.split("/")
                filename_base = url_path[-1] if url_path[-1] else url_path[-2]
                # 清理文件名
                safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename_base)
                filename = f"{safe_name}.html"
                
                # 保存文件
                output_path = Path(raw_html_dir) / filename
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(body_html)
                
                print(f"  ✓ Saved: {output_path.name}")
                
                # 等待一下
                if i < len(df) - 1:
                    time.sleep(sleep_sec)
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        browser.close()
    
    print(f"\n✅ 所有页面已保存到：{raw_html_dir}")


def main():
    args = build_parser().parse_args()
    paths = get_year_paths(args.year)
    
    # 检查 players.csv 是否存在
    if not paths["players_csv"].exists():
        print(f"错误：{paths['players_csv']} 不存在")
        print("请先运行 top_workflow.py 生成选手名单")
        sys.exit(1)
    
    # 创建 raw_html 目录
    paths["raw_html_dir"].mkdir(parents=True, exist_ok=True)
    
    print(f"开始抓取 {args.year} 年选手页面...")
    print(f"选手名单：{paths['players_csv']}")
    print(f"保存目录：{paths['raw_html_dir']}")
    print("=" * 60)
    
    fetch_all_players(paths["players_csv"], paths["raw_html_dir"], sleep_sec=args.sleep_sec)


if __name__ == "__main__":
    main()
