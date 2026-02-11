import requests
from bs4 import BeautifulSoup

url = "https://www.lycorp.co.jp/ja/news/"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)..."}
res = requests.get(url, headers=headers)
res.encoding = res.apparent_encoding  # 文字化け防止

res.raise_for_status()

soup = BeautifulSoup(res.text, "html.parser")

# class="c-col" の li をすべて取得
items = soup.find_all("li", class_="c-col")

results = []

for item in items:
    # a タグ（記事リンク）
    a_tag = item.find("a", class_="c-article-panel-d2")
    if not a_tag:
        continue

    link = a_tag.get("href")

    # time タグ（日付）
    time_tag = item.find("time")
    date = time_tag.get_text(strip=True) if time_tag else None

    # p タグ（タイトル）
    p_tag = item.find("p")
    title = p_tag.get_text(strip=True) if p_tag else None

    results.append({
        "url": link,
        "date": date,
        "title": title
    })

# 結果を表示
for r in results:
    print(r)
