import re
import html as ihtml
import pandas as pd
from bs4 import BeautifulSoup
from lxml import html

path = r"E:\OneDrive - International Campus, Zhejiang University\EquiRating\结构化html\donk.html"
raw = open(path, "r", encoding="utf-8", errors="ignore").read()

# 1) 解析“保存的页面”（里面有大量 span.html-tag 这种高亮结构）
soup = BeautifulSoup(raw, "lxml")

# 2) 取出整页文本，然后反转义（&lt;div -> <div）
text = soup.get_text("\n")
real = ihtml.unescape(text)

# 3) 现在 real 里面应该包含真正的 <div class="role-stats-row ...">
root = html.fromstring(real)

# 4) 找 role-stats-row（这次应该 > 0）
rows = root.xpath('//*[contains(concat(" ", normalize-space(@class), " "), " role-stats-row ")]')
print("role-stats-row nodes after unescape:", len(rows))

out = []
for r in rows:
    cls = r.get("class", "")

    if "stats-side-combined" in cls:
        side = "Both"
    elif "stats-side-ct" in cls:
        side = "CT"
    elif "stats-side-t" in cls:
        side = "T"
    else:
        continue

    # metric 名称（title 文本）
    t = r.xpath('.//*[contains(concat(" ", normalize-space(@class), " "), " role-stats-title ")]')
    metric = t[0].text_content().strip() if t else None

    # value：优先 data-original-value；否则取 role-stats-data 文本（适配 rating 那种结构）
    val = r.get("data-original-value")
    if val is None:
        d = r.xpath('.//*[contains(concat(" ", normalize-space(@class), " "), " role-stats-data ")]')
        val = d[0].text_content().strip() if d else None

    if metric and val:
        out.append({"metric": metric, "side": side, "value": val})

df = pd.DataFrame(out)
wide = df.pivot_table(index="metric", columns="side", values="value", aggfunc="first").reset_index()

print("metrics:", wide.shape[0])
print(wide.head(30))

# 可选：保存
wide.to_csv(r"E:\OneDrive - International Campus, Zhejiang University\EquiRating\结构化html\donk_role_stats.csv", index=False, encoding="utf-8-sig")
print("saved csv")