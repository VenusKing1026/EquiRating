import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from lxml import html

# =========================
# 用户参数
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def get_year_paths(year: str):
    year_dir = DATA_DIR / str(year)
    return {
        "year_dir": year_dir,
        "raw_html_dir": year_dir / "raw_html",
        "output_base_dir": year_dir / "processed_data",
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Parse HLTV player HTML into CSV/JSON")
    parser.add_argument(
        "--year",
        default=str(datetime.now().year),
        help="Data year, e.g. 2026",
    )
    parser.add_argument(
        "--input-html",
        default=None,
        help="Path to player HTML file",
    )
    parser.add_argument(
        "--output-base-dir",
        default=None,
        help="Base output directory for processed data",
    )
    return parser

# =========================
# 自动生成选手文件夹
# =========================

args = build_parser().parse_args()
paths = get_year_paths(args.year)
input_html = os.path.abspath(
    args.input_html
    if args.input_html
    else str(paths["raw_html_dir"] / "donk_startDate_2025-11-28_endDate_2026-02-28_rankingFilter_Top20.html")
)
output_base_dir = os.path.abspath(
    args.output_base_dir if args.output_base_dir else str(paths["output_base_dir"])
)

raw_player_name = os.path.splitext(os.path.basename(input_html))[0]
player_name = raw_player_name.split("_startDate", 1)[0]
player_dir = os.path.join(output_base_dir, player_name)

os.makedirs(player_dir, exist_ok=True)

print("Player:", player_name)
print("Output folder:", player_dir)

# =========================
# 读取 DOM
# =========================

s = open(input_html, "r", encoding="utf-8", errors="ignore").read().strip()
root = html.fromstring(s)

def cls(token: str) -> str:
    return f'contains(concat(" ", normalize-space(@class), " "), " {token} ")'

def direct_text(el):
    if el is None:
        return None
    parts = [t.strip() for t in el.xpath("./text()") if t and t.strip()]
    return " ".join(parts) if parts else None

def first(nodes):
    return nodes[0] if nodes else None

def parse_value_text(el):
    if el is None:
        return (None, None)
    txt = re.sub(r"\s+", " ", el.text_content().strip())
    unit = "%" if "%" in txt else None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", txt)
    val = m.group(0) if m else txt
    return (val, unit)

# =========================
# 1️⃣ 解析 role-stats
# =========================

role_rows = root.xpath(f'//*[ {cls("role-stats-row")} ]')
role_out = []

for r in role_rows:
    c = r.get("class", "")

    if "stats-side-combined" in c:
        side = "Both"
    elif "stats-side-ct" in c:
        side = "CT"
    elif "stats-side-t" in c:
        side = "T"
    else:
        continue

    title_el = first(r.xpath(f'.//*[ {cls("role-stats-title")} ]'))
    metric = title_el.text_content().strip() if title_el is not None else None

    val = r.get("data-original-value")
    if val is None:
        data_el = first(r.xpath(f'.//*[ {cls("role-stats-data")} ]'))
        val = data_el.text_content().strip() if data_el is not None else None

    if metric and val:
        role_out.append({"metric": metric, "side": side, "value": val})

df_role = pd.DataFrame(role_out)

if not df_role.empty:
    role_wide = df_role.pivot_table(
        index="metric", columns="side", values="value", aggfunc="first"
    ).reset_index()

    role_wide.to_csv(
        os.path.join(player_dir, "role_stats.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("✔ role_stats.csv saved")

# =========================
# 2️⃣ 解析 right-bottom
# =========================

rb = root.xpath(f'//*[ {cls("player-summary-stat-box-right-bottom")} ]')

right_out = []

if rb:
    rb = rb[0]
    boxes = rb.xpath(f'.//*[ {cls("player-summary-stat-box-data-wrapper")} ]')

    for b in boxes:
        grade_el = first(b.xpath(f'.//*[ {cls("player-summary-stat-box-breakdown-description")} ]'))
        grade = grade_el.text_content().strip() if grade_el is not None else None

        # 单值型
        plain_data = first(b.xpath(
            f'.//*[ {cls("player-summary-stat-box-data")} and '
            f'not({cls("traditionalData")}) and not({cls("ecoAdjustedData")}) ]'
        ))

        if plain_data is not None:
            val, unit = parse_value_text(plain_data)
            text_el = first(b.xpath(f'.//*[ {cls("player-summary-stat-box-data-text")} ]'))
            metric = direct_text(text_el)

            if metric:
                right_out.append({
                    "metric": metric,
                    "variant": "single",
                    "value": val,
                    "unit": unit,
                    "grade": grade
                })
            continue

        # 双值型
        trad_data = first(b.xpath(f'.//*[ {cls("player-summary-stat-box-data")} and {cls("traditionalData")} ]'))
        eco_data  = first(b.xpath(f'.//*[ {cls("player-summary-stat-box-data")} and {cls("ecoAdjustedData")} ]'))

        trad_val, trad_unit = parse_value_text(trad_data)
        eco_val, eco_unit   = parse_value_text(eco_data)

        trad_text = first(b.xpath(f'.//*[ {cls("player-summary-stat-box-data-text")} and {cls("traditionalData")} ]'))
        eco_text  = first(b.xpath(f'.//*[ {cls("player-summary-stat-box-data-text")} and {cls("ecoAdjustedData")} ]'))

        trad_metric = direct_text(trad_text)
        eco_metric  = direct_text(eco_text) or trad_metric

        if trad_metric:
            right_out.append({
                "metric": trad_metric,
                "variant": "traditional",
                "value": trad_val,
                "unit": trad_unit,
                "grade": grade
            })

        if eco_metric:
            right_out.append({
                "metric": eco_metric,
                "variant": "ecoAdjusted",
                "value": eco_val,
                "unit": eco_unit,
                "grade": grade
            })

if right_out:
    df_right = pd.DataFrame(right_out)
    df_right.to_csv(
        os.path.join(player_dir, "right_bottom.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    print("✔ right_bottom.csv saved")

# =========================
# 3️⃣ 可选：保存基础信息
# =========================

meta = {
    "player": player_name,
    "source_html": input_html
}

with open(os.path.join(player_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("✔ meta.json saved")
print("DONE.")