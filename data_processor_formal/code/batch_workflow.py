import subprocess
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

# 1. 批量自动打开所有选手页面

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSOR_SCRIPT = Path(__file__).resolve().parent / "data_processor.py"
TOP_WORKFLOW_SCRIPT = Path(__file__).resolve().parent / "top_workflow.py"


def build_parser():
    parser = argparse.ArgumentParser(description="Batch open players and process HTML files")
    parser.add_argument("--year", default=str(datetime.now().year), help="Data year, e.g. 2026")
    parser.add_argument(
        "--prepare-players",
        action="store_true",
        help="Run top_workflow (TOPmapping.html + TOP20.html) before opening pages",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only process existing HTML files, do not open browser pages",
    )
    return parser


def get_year_paths(year: str):
    year_dir = DATA_DIR / str(year)
    return {
        "year_dir": year_dir,
        "raw_html_dir": year_dir / "raw_html",
        "players_csv": year_dir / "players.csv",
    }

def open_all_players(csv_path, raw_html_dir, sleep_sec=2):
    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        if pd.isna(row.get("url")) or str(row.get("url", "")).strip() == "":
            print(f"Skipping {row['name']} (url 为空)")
            continue
        print(f"Opening {row['name']} ...")
        subprocess.run(["cmd", "/c", "start", "msedge", row["url"]], check=False)
        time.sleep(sleep_sec)
    print(f"请在浏览器中用插件手动保存所有页面 body HTML 到目录: {raw_html_dir}")


def prepare_players(year):
    subprocess.run(
        [sys.executable, str(TOP_WORKFLOW_SCRIPT), "--year", str(year)],
        check=True,
    )

# 2. 批量处理所有 HTML 文件

def process_all_html(html_dir, year, processor_script=PROCESSOR_SCRIPT):
    html_files = sorted(Path(html_dir).glob("*.html"))
    for html_file in html_files:
        if html_file.name in {"TOP20.html", "TOPmapping.html"}:
            continue
        print(f"Processing {html_file} ...")
        subprocess.run(
            [
                sys.executable,
                str(processor_script),
                "--year",
                str(year),
                "--input-html",
                str(html_file),
            ],
            check=False,
        )
    print("全部处理完成！")

if __name__ == "__main__":
    args = build_parser().parse_args()
    paths = get_year_paths(args.year)

    paths["year_dir"].mkdir(parents=True, exist_ok=True)
    paths["raw_html_dir"].mkdir(parents=True, exist_ok=True)

    if args.prepare_players or not paths["players_csv"].exists():
        prepare_players(args.year)

    if args.process_only:
        process_all_html(paths["raw_html_dir"], args.year)
        sys.exit(0)

    # 步骤1：批量打开页面
    open_all_players(paths["players_csv"], paths["raw_html_dir"], sleep_sec=2)
    input("\n请用插件手动保存所有页面 body HTML 后，按回车继续...\n")
    # 步骤2：批量处理 HTML
    process_all_html(paths["raw_html_dir"], args.year)
