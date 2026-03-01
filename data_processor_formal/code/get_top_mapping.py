from bs4 import BeautifulSoup
import csv
import argparse
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BASE_URL = "https://www.hltv.org"


def build_parser():
    parser = argparse.ArgumentParser(description="Build players_mapping.csv from TOPmapping HTML")
    parser.add_argument("--year", default=str(datetime.now().year), help="Data year, e.g. 2026")
    return parser


def get_year_paths(year: str):
    year_dir = DATA_DIR / str(year)
    return {
        "year_dir": year_dir,
        "html_file": year_dir / "raw_html" / "TOPmapping.html",
        "mapping_csv": year_dir / "players_mapping.csv",
    }


def parse_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    players = []
    rows = soup.find_all("tr")

    order = 1
    for row in rows:
        player_cell = row.find("td", class_="playerCol")
        if not player_cell:
            continue

        link = player_cell.find("a", href=True)
        if not link:
            continue

        name = link.text.strip()
        relative_url = link["href"]
        full_url = BASE_URL + relative_url

        players.append({
            "order": order,
            "name": name,
            "url": full_url,
        })
        order += 1

    return players


def save_to_csv(players, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["order", "name", "url"])
        writer.writeheader()
        writer.writerows(players)

    print(f"✅ 映射表解析成功 {len(players)} 名选手")
    print(f"📁 已保存为 {filename}")


def main():
    args = build_parser().parse_args()
    paths = get_year_paths(args.year)

    if not paths["html_file"].exists():
        print(f"❌ 未找到 mapping HTML: {paths['html_file']}")
        sys.exit(1)

    players = parse_html(paths["html_file"])

    if not players:
        print("❌ 未从 TOPmapping.html 解析到选手")
        sys.exit(1)
    else:
        save_to_csv(players, paths["mapping_csv"])


if __name__ == "__main__":
    main()
