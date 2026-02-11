import requests
from bs4 import BeautifulSoup

# 1. サイトのHTMLを取得
url = "https://www.lycorp.co.jp/ja/news/"
response = requests.get(url)
response.encoding = response.apparent_encoding  # 文字化け防止

# 2. BeautifulSoupでパース
soup = BeautifulSoup(response.text, "html.parser")

# 3. ニュース項目（li class="c-col"）をすべて抽出
news_items = soup.find_all("li", class_="c-col")

# 4. 各項目から必要な情報を抜き出す
for item in news_items:
    # aタグのhref属性（リンク）を取得
    a_tag = item.find("a")
    link = a_tag.get("href") if a_tag else "N/A"

    # timeタグのテキスト（日付）を取得
    time_tag = item.find("time")
    date = time_tag.get_text(strip=True) if time_tag else "N/A"

    # pタグのテキスト（タイトル/内容）を取得
    p_tag = item.find("p")
    content = p_tag.get_text(strip=True) if p_tag else "N/A"

    # 結果を表示
    print(f"日付: {date}")
    print(f"リンク: {link}")
    print(f"内容: {content}")
    print("-" * 30)