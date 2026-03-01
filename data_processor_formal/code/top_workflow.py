import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MAPPING_SCRIPT = BASE_DIR / "get_top_mapping.py"
RANK_SCRIPT = BASE_DIR / "get_top_rank.py"


def build_parser():
    parser = argparse.ArgumentParser(description="Run mapping + rank workflow to build players.csv")
    parser.add_argument("--year", default=str(datetime.now().year), help="Data year, e.g. 2026")
    return parser


def run_step(script_path: Path, year: str):
    command = [sys.executable, str(script_path), "--year", str(year)]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    args = build_parser().parse_args()

    print(f"▶ Step 1/2: 生成映射表 ({MAPPING_SCRIPT.name})")
    run_step(MAPPING_SCRIPT, args.year)

    print(f"▶ Step 2/2: 生成年度榜单 ({RANK_SCRIPT.name})")
    run_step(RANK_SCRIPT, args.year)

    print("✅ 工作流完成：players.csv 已生成")


if __name__ == "__main__":
    main()
