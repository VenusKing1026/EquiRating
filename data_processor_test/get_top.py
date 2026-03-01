from bs4 import BeautifulSoup
import csv

# ==========================
# 1️⃣ 修改为你的 HTML 文件路径
# ==========================
HTML_FILE = r"E:\OneDrive - International Campus, Zhejiang University\EquiRating\data_processor_test\TOP20.html"

BASE_URL = "https://www.hltv.org"


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
            "url": full_url
        })

        order += 1

    return players


def save_to_csv(players, filename="players.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["order", "name", "url"])
        writer.writeheader()
        writer.writerows(players)

    print(f"✅ 成功解析 {len(players)} 名选手")
    print(f"📁 已保存为 {filename}")


def main():
    players = parse_html(HTML_FILE)

    if not players:
        print("❌ 未找到选手")
    else:
        save_to_csv(players)


if __name__ == "__main__":
    main()