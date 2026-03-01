from bs4 import BeautifulSoup
import csv
import argparse
from datetime import datetime
from pathlib import Path

# ==========================
# 1️⃣ 修改为你的 HTML 文件路径
# ==========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def build_parser():
    parser = argparse.ArgumentParser(description="Build ranked players.csv from TOP20 and mapping CSV")
    parser.add_argument("--year", default=str(datetime.now().year), help="Data year, e.g. 2026")
    parser.add_argument(
        "--mapping-csv",
        default=None,
        help="Path to name->url mapping CSV. Default: data/<year>/players_mapping.csv",
    )
    return parser


def get_year_paths(year: str):
    year_dir = DATA_DIR / str(year)
    return {
        "year_dir": year_dir,
        "html_file": year_dir / "raw_html" / "TOP20.html",
        "mapping_csv": year_dir / "players_mapping.csv",
        "players_csv": year_dir / "players.csv",
    }


def normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def load_url_map(csv_path):
    url_map = {}
    p = Path(csv_path)
    if not p.exists():
        print(f"⚠️ 未找到旧映射表: {p}，url 将留空")
        return url_map

    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = normalize_name(row.get("name", ""))
            url = (row.get("url", "") or "").strip()
            if name and url and name not in url_map:
                url_map[name] = url

    print(f"ℹ️ 已加载旧映射 {len(url_map)} 条: {p}")
    return url_map


def parse_html(file_path, url_map):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    players = []
    rows = soup.find_all("tr")

    order = 1

    for row in rows:
        # 新结构：top20-year-*（当年榜单顺序）
        nick_el = row.select_one("b.top20-year-playernick")
        link = row.select_one("a.top20-year-playername-wrapper[href]")
        if not nick_el or not link:
            continue

        name = nick_el.get_text(strip=True)
        mapped_url = url_map.get(normalize_name(name), "")

        pos_el = row.select_one("td.top20-th-position b")
        if pos_el:
            pos_text = pos_el.get_text(strip=True).replace("#", "")
            if pos_text.isdigit():
                order = int(pos_text)

        players.append({
            "order": order,
            "name": name,
            "url": mapped_url
        })
        order += 1

    empty_cnt = sum(1 for p in players if not p.get("url"))
    print(f"ℹ️ 按映射回填 url 完成，留空 {empty_cnt} 条")

    return players


def save_to_csv(players, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["order", "name", "url"])
        writer.writeheader()
        writer.writerows(players)

    print(f"✅ 成功解析 {len(players)} 名选手")
    print(f"📁 已保存为 {filename}")


def main():
    args = build_parser().parse_args()
    paths = get_year_paths(args.year)

    mapping_csv = args.mapping_csv if args.mapping_csv else str(paths["mapping_csv"])
    url_map = load_url_map(mapping_csv)
    players = parse_html(paths["html_file"], url_map)

    if not players:
        print("❌ 未找到选手")
    else:
        save_to_csv(players, paths["players_csv"])


if __name__ == "__main__":
    main()