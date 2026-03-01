import pandas as pd
from lxml import html
import re

PATH = r"E:\OneDrive - International Campus, Zhejiang University\EquiRating\结构化html\right_bottom.html"
OUT_CSV = r"E:\OneDrive - International Campus, Zhejiang University\EquiRating\结构化html\right_bottom_structured.csv"
OUT_JSON = r"E:\OneDrive - International Campus, Zhejiang University\EquiRating\结构化html\right_bottom_structured.json"

s = open(PATH, "r", encoding="utf-8", errors="ignore").read().strip()
root = html.fromstring(s)

def cls(token: str) -> str:
    return f'contains(concat(" ", normalize-space(@class), " "), " {token} ")'

def direct_text(el):
    """只取直系文本（避免把 tooltip 的解释一起拿进来）"""
    if el is None:
        return None
    parts = [t.strip() for t in el.xpath("./text()") if t and t.strip()]
    return " ".join(parts) if parts else None

def full_text(el):
    return el.text_content().strip() if el is not None else None

def parse_value(data_el):
    """
    data_el: <div class="player-summary-stat-box-data ..."> 可能包含 <span class="...percentage">%</span>
    取出主数值 + 单位（%）
    """
    if data_el is None:
        return (None, None)
    txt = data_el.text_content().strip()
    # 把连续空白压掉
    txt = re.sub(r"\s+", " ", txt)
    unit = "%" if "%" in txt else None
    # 提取数值部分（允许 +3.40, 77.6, 0.66）
    m = re.search(r"[-+]?\d+(?:\.\d+)?", txt)
    val = m.group(0) if m else txt
    return (val, unit)

# 一个 box = 一个 player-summary-stat-box-data-wrapper
boxes = root.xpath(f'.//*[ {cls("player-summary-stat-box-data-wrapper")} ]')

rows = []
for b in boxes:
    grade_el = b.xpath(f'.//*[ {cls("player-summary-stat-box-breakdown-description")} ]')
    grade = full_text(grade_el[0]) if grade_el else None

    # --- 单值型：只有一个 data，没有 traditional/eco class ---
    plain_data = b.xpath(
        f'.//*[ {cls("player-summary-stat-box-data")} '
        f'and not({cls("traditionalData")}) and not({cls("ecoAdjustedData")}) ]'
    )
    if plain_data:
        data_el = plain_data[0]
        val, unit = parse_value(data_el)

        # label 在 data-text 里（也可能只有一个）
        text_el = b.xpath(f'.//*[ {cls("player-summary-stat-box-data-text")} ]')
        label = direct_text(text_el[0]) if text_el else None

        rows.append({
            "metric": label,
            "variant": "single",
            "value": val,
            "unit": unit,
            "grade": grade,
            "tooltip_title": None,
            "tooltip_desc": None,
        })
        continue

    # --- 双值型：traditional + ecoAdjusted ---
    # 传统值
    trad_data_el = b.xpath(f'.//*[ {cls("player-summary-stat-box-data")} and {cls("traditionalData")} ]')
    eco_data_el  = b.xpath(f'.//*[ {cls("player-summary-stat-box-data")} and {cls("ecoAdjustedData")} ]')
    trad_val, trad_unit = parse_value(trad_data_el[0]) if trad_data_el else (None, None)
    eco_val, eco_unit   = parse_value(eco_data_el[0]) if eco_data_el else (None, None)

    # label：注意 ecoAdjusted 的 label 可能不同（Multi-kill vs MK rating）
    trad_label_el = b.xpath(f'.//*[ {cls("player-summary-stat-box-data-text")} and {cls("traditionalData")} ]')
    eco_label_el  = b.xpath(f'.//*[ {cls("player-summary-stat-box-data-text")} and {cls("ecoAdjustedData")} ]')

    trad_label = direct_text(trad_label_el[0]) if trad_label_el else None
    eco_label  = direct_text(eco_label_el[0]) if eco_label_el else None

    # tooltip：可选，取 b 内对应 variant 的 tooltip
    def parse_tooltip(text_el):
        if text_el is None:
            return (None, None)
        tip = text_el.xpath(f'.//*[ {cls("player-summary-tooltip")} ]')
        if not tip:
            return (None, None)
        tip = tip[0]
        # <b>Title</b><br>Desc...
        title_el = tip.xpath(".//b")
        title = full_text(title_el[0]) if title_el else None
        # 去掉 title 后剩余描述：用整个 text_content 再减去 title（简单处理）
        desc = tip.text_content().strip()
        if title and desc.startswith(title):
            desc = desc[len(title):].strip()
        desc = desc.lstrip("-").strip()
        return (title, desc)

    trad_tt = parse_tooltip(trad_label_el[0]) if trad_label_el else (None, None)
    eco_tt  = parse_tooltip(eco_label_el[0]) if eco_label_el else (None, None)

    # 输出两行：traditional 和 ecoAdjusted（更干净）
    rows.append({
        "metric": trad_label,
        "variant": "traditional",
        "value": trad_val,
        "unit": trad_unit,
        "grade": grade,
        "tooltip_title": trad_tt[0],
        "tooltip_desc": trad_tt[1],
    })
    rows.append({
        "metric": eco_label if eco_label else trad_label,   # 兜底
        "variant": "ecoAdjusted",
        "value": eco_val,
        "unit": eco_unit,
        "grade": grade,
        "tooltip_title": eco_tt[0],
        "tooltip_desc": eco_tt[1],
    })

df = pd.DataFrame(rows)

print(df)

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
df.to_json(OUT_JSON, orient="records", force_ascii=False, indent=2)

print("saved:", OUT_CSV)
print("saved:", OUT_JSON)