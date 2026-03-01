import html as ihtml
from bs4 import BeautifulSoup
from lxml import html

PATH = r"E:\OneDrive - International Campus, Zhejiang University\EquiRating\结构化html\donk2.html"

raw = open(PATH, "r", encoding="utf-8", errors="ignore").read()

# 1️⃣ 解析“高亮代码页面”
soup = BeautifulSoup(raw, "lxml")

# 2️⃣ 只抓 html-tag 里的文本（这些才是真正的源码）
code_spans = soup.select("span.html-tag")

print("html-tag spans:", len(code_spans))

# 拼接源码
code_text = "\n".join(span.get_text() for span in code_spans)

# 3️⃣ 反转义 &lt; &gt; 等
real_html = ihtml.unescape(code_text)

print("real_html length:", len(real_html))
print("real_html head:", real_html[:200])

# 4️⃣ 再解析
root = html.fromstring(real_html)

# 5️⃣ 现在应该能找到 role-stats-row
rows = root.xpath('//*[contains(concat(" ", normalize-space(@class), " "), " role-stats-row ")]')
print("role-stats-row nodes:", len(rows))

# 6️⃣ 测试 right-bottom
rb = root.xpath('//*[contains(concat(" ", normalize-space(@class), " "), " player-summary-stat-box-right-bottom ")]')
print("right-bottom nodes:", len(rb))