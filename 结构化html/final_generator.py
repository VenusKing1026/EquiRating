import os
import re
import pandas as pd
from lxml import html

IN_HTML = r"E:\OneDrive - International Campus, Zhejiang University\EquiRating\结构化html\donk2.html"
OUT_ROLE = r"E:\OneDrive - International Campus, Zhejiang University\EquiRating\结构化html\donk_role_stats.csv"
OUT_RIGHT = r"E:\OneDrive - International Campus, Zhejiang University\EquiRating\结构化html\donk_right_bottom.csv"

assert os.path.exists(IN_HTML), IN_HTML
s = open(IN_HTML, "r", encoding="utf-8", errors="ignore").read()
root = html.fromstring(s)

def cls(token: str) -> str:
    return f'contains(concat(" ", normalize-space(@class), " "), " {token} ")'

def direct_text(el):
    """只取直系文本，避免 tooltip 混入"""
    if el is None:
        return None
    parts = [t.strip() for t in el.xpath("./text()") if t and t.strip()]
    return " ".join(parts) if parts else None

def first(nodes):
    return nodes[0] if nodes else None

def parse_value_text(el):
    """解析 <div ...>77.6<span>%</span></div> -> ('77.6','%')"""
    if el is None:
        return (None, None)
    txt = re.sub(r"\s+", " ", el.text_content().strip())
    unit = "%" if "%" in txt else None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", txt)
    val = m.group(0) if m else txt
    return (val, unit)

# =========================
# 1) 输出 role-stats（Both/CT/T）
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

    # value：优先 data-original-value，否则取 role-stats-data 文本（适配 Rating 那种）
    val = r.get("data-original-value")
    if val is None:
        data_el = first(r.xpath(f'.//*[ {cls("role-stats-data")} ]'))
        val = data_el.text_content().strip() if data_el is not None else None

    if metric and val:
        role_out.append({"metric": metric, "side": side, "value": val})

df_role = pd.DataFrame(role_out)
if df_role.empty:
    print("WARN: role-stats not found. Is this really the rendered DOM (outerHTML)?")
else:
    # 宽表：metric | Both | CT | T
    wide_role = df_role.pivot_table(index="metric", columns="side", values="value", aggfunc="first").reset_index()
    wide_role.to_csv(OUT_ROLE, index=False, encoding="utf-8-sig")
    print("saved:", OUT_ROLE)

# =========================
# 2) 输出 right-bottom（single vs traditional/eco）
# =========================
rb = root.xpath(f'//*[ {cls("player-summary-stat-box-right-bottom")} ]')
if not rb:
    print("WARN: right-bottom not found. Is this really the rendered DOM (outerHTML)?")
else:
    rb = rb[0]
    boxes = rb.xpath(f'.//*[ {cls("player-summary-stat-box-data-wrapper")} ]')

    right_out = []
    for b in boxes:
        grade_el = first(b.xpath(f'.//*[ {cls("player-summary-stat-box-breakdown-description")} ]'))
        grade = grade_el.text_content().strip() if grade_el is not None else None

        # 单值
        plain_data = first(b.xpath(
            f'.//*[ {cls("player-summary-stat-box-data")} and not({cls("traditionalData")}) and not({cls("ecoAdjustedData")}) ]'
        ))
        if plain_data is not None:
            val, unit = parse_value_text(plain_data)
            text_el = first(b.xpath(f'.//*[ {cls("player-summary-stat-box-data-text")} ]'))
            metric = direct_text(text_el)
            if metric and val:
                right_out.append({
                    "metric": metric, "variant": "single",
                    "value": val, "unit": unit, "grade": grade
                })
            continue

        # 双值（traditional + ecoAdjusted）
        trad_data = first(b.xpath(f'.//*[ {cls("player-summary-stat-box-data")} and {cls("traditionalData")} ]'))
        eco_data  = first(b.xpath(f'.//*[ {cls("player-summary-stat-box-data")} and {cls("ecoAdjustedData")} ]'))
        trad_val, trad_unit = parse_value_text(trad_data)
        eco_val, eco_unit   = parse_value_text(eco_data)

        trad_text = first(b.xpath(f'.//*[ {cls("player-summary-stat-box-data-text")} and {cls("traditionalData")} ]'))
        eco_text  = first(b.xpath(f'.//*[ {cls("player-summary-stat-box-data-text")} and {cls("ecoAdjustedData")} ]'))

        trad_metric = direct_text(trad_text)
        eco_metric  = direct_text(eco_text) or trad_metric

        if trad_metric and trad_val:
            right_out.append({
                "metric": trad_metric, "variant": "traditional",
                "value": trad_val, "unit": trad_unit, "grade": grade
            })
        if eco_metric and eco_val:
            right_out.append({
                "metric": eco_metric, "variant": "ecoAdjusted",
                "value": eco_val, "unit": eco_unit, "grade": grade
            })

    df_right = pd.DataFrame(right_out)
    if df_right.empty:
        print("WARN: extracted 0 right-bottom stats.")
    else:
        df_right.to_csv(OUT_RIGHT, index=False, encoding="utf-8-sig")
        print("saved:", OUT_RIGHT)