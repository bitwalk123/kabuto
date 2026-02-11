import sys
import webbrowser
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import (
    Qt,
    QThread,
    Signal,
)
import requests
from bs4 import BeautifulSoup


# --- スクレイピング処理を行うスレッド ---
class Fetcher(QThread):
    finished = Signal(list)

    def run(self):
        url = "https://www.lycorp.co.jp/ja/news/"
        try:
            res = requests.get(url, timeout=10)
            res.encoding = res.apparent_encoding
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            items = soup.find_all("li", class_="c-col")

            results = []
            for item in items:
                a_tag = item.find("a", class_="c-article-panel-d2")
                if not a_tag: continue

                results.append({
                    "url": a_tag.get("href"),
                    "date": item.find("time").get_text(strip=True) if item.find("time") else "",
                    "title": item.find("p").get_text(strip=True) if item.find("p") else ""
                })
            self.finished.emit(results)
        except Exception as e:
            print(f"Error: {e}")
            self.finished.emit([])


# --- メインウィンドウ ---
class NewsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LINEヤフー ニュースビューアー")
        self.resize(800, 500)

        # UIレイアウト
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # テーブルの設定
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["日付", "タイトル"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)  # 行選択
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)  # 編集禁止

        # 列の幅調整
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # 日付は内容に合わせる
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # タイトルは伸ばす

        # クリックイベント
        self.table.cellDoubleClicked.connect(self.on_cell_clicked)

        self.btn_refresh = QPushButton("ニュースを更新")
        self.btn_refresh.clicked.connect(self.fetch_news)

        self.layout.addWidget(self.table)
        self.layout.addWidget(self.btn_refresh)

        # 起動時に一度実行
        self.fetch_news()

    def fetch_news(self):
        self.btn_refresh.setEnabled(False)
        self.worker = Fetcher()
        self.worker.finished.connect(self.display_news)
        self.worker.start()

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
    window = NewsApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
