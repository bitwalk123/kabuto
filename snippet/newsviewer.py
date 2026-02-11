import sys
import webbrowser
from abc import ABC, abstractmethod
from typing import Dict

import requests
from PySide6.QtCore import (
    Qt,
    QThread,
    Signal,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QHeaderView,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from bs4 import BeautifulSoup


class ParserBase(ABC):
    @abstractmethod
    def get_url(self):
        pass

    @abstractmethod
    def parse(self, soup):
        """BeautifulSoupのオブジェクトを受け取り、辞書のリストを返す"""
        pass


class Parser4005(ParserBase):
    """住友化学 (4005)"""

    def get_url(self):
        return "https://www.sumitomo-chem.co.jp/news/"

    def parse(self, soup):
        base_url = "https://www.sumitomo-chem.co.jp"
        # <ul class="m-list-news"> 内の <li> を探す
        news_list = soup.find("ul", class_="m-list-news")
        if not news_list:
            return []

        items = news_list.find_all("li")
        results = []
        for item in items:
            a_tag = item.find("a")
            if not a_tag:
                continue

            # 相対パスを絶対URLに変換
            relative_url = a_tag.get("href")
            full_url = base_url + relative_url if relative_url.startswith("/") else relative_url

            # 日付の取得
            date_el = item.find("p", class_="news-date")
            date_text = date_el.get_text(strip=True) if date_el else ""

            # タイトルの取得
            title_el = item.find("p", class_="news-ttl")
            title_text = title_el.get_text(strip=True) if title_el else ""

            results.append({
                "url": full_url,
                "date": date_text,
                "title": title_text
            })
        return results


class Parser4689(ParserBase):
    """LINEヤフー (4689)"""

    def get_url(self):
        return "https://www.lycorp.co.jp/ja/news/"

    def parse(self, soup):
        items = soup.find_all("li", class_="c-col")
        results = []
        for item in items:
            a_tag = item.find("a", class_="c-article-panel-d2")
            if a_tag:
                results.append({
                    "url": a_tag.get("href"),
                    "date": item.find("time").get_text(strip=True),
                    "title": item.find("p").get_text(strip=True)
                })
        return results


class Fetcher(QThread):
    finished = Signal(list)

    def __init__(self, parser: ParserBase):
        super().__init__()
        self.parser = parser

    def run(self):
        try:
            url = self.parser.get_url()
            res = requests.get(url, timeout=10)
            res.encoding = res.apparent_encoding
            res.raise_for_status()

            soup = BeautifulSoup(res.text, "html.parser")
            # 渡されたパーサーに解析を丸投げ
            results = self.parser.parse(soup)

            self.finished.emit(results)
        except Exception as e:
            print(f"Error: {e}")
            self.finished.emit([])


class NewsViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ニュース・ビューアー")
        self.resize(800, 600)
        self.worker = None

        self.parsers: Dict[str, ParserBase] = {
            "住友化学 (4005)": Parser4005(),
            "LINEヤフー (4689)": Parser4689(),
        }

        # ツールバー
        self.toolbar = toolbar = QToolBar()
        self.addToolBar(toolbar)

        # 銘柄選択用コンボボックス
        self.combo = combo = QComboBox()
        combo.addItems(self.parsers.keys())
        combo.currentTextChanged.connect(self.fetch_news)
        toolbar.addWidget(combo)

        # レイアウト
        self.base = base = QWidget()
        self.setCentralWidget(base)
        self.layout = layout = QVBoxLayout(base)

        # テーブルの設定
        self.table = table = QTableWidget(0, 2)
        table.setStyleSheet("QTableWidget {font-family: monospace;}")
        table.setHorizontalHeaderLabels(["日付", "タイトル"])
        # 行ヘッダー（垂直ヘッダー）を取得して、右寄せ＋垂直中央に設定
        table.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)  # 行選択
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)  # 編集禁止
        table.cellDoubleClicked.connect(self.on_cell_clicked)  # クリックイベント
        self.layout.addWidget(table)

        # 列の幅調整
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # 日付は内容に合わせる
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # タイトルは伸ばす

        # 更新ボタン
        self.btn_refresh = btn_refresh = QPushButton("ニュースを更新")
        btn_refresh.clicked.connect(self.fetch_news)
        self.layout.addWidget(btn_refresh)

        # 起動時に一度実行
        self.fetch_news()

    def fetch_news(self):
        # 選択されている銘柄名を取得
        selected_name = self.combo.currentText()
        # 対応するパーサーを取り出す
        parser = self.parsers[selected_name]
        # スレッドにそのパーサーを託す
        self.worker = worker = Fetcher(parser)
        worker.finished.connect(self.display_news)
        worker.start()

    def display_news(self, news_list):
        self.table.setRowCount(0)
        for news in news_list:
            row = self.table.rowCount()
            self.table.insertRow(row)

            # 日付アイテム
            date_item = QTableWidgetItem(news["date"])
            # タイトルアイテム（URLをデータとして持たせる）
            title_item = QTableWidgetItem(news["title"])
            title_item.setData(Qt.ItemDataRole.UserRole, news["url"])

            self.table.setItem(row, 0, date_item)
            self.table.setItem(row, 1, title_item)

        self.btn_refresh.setEnabled(True)

    def on_cell_clicked(self, row, column):
        # どの列をクリックしてもタイトル列(1)に保存したURLを取得
        url = self.table.item(row, 1).data(Qt.ItemDataRole.UserRole)
        if url:
            webbrowser.open(url)


def main():
    app = QApplication(sys.argv)
    win = NewsViewer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
