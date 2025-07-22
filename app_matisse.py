# MarketSPEED 2 RSS 用いた信用取引テスト
import logging
import sys

from PySide6.QtWidgets import QApplication

from funcs.logs import setup_logging
from widgets.containers import PanelTrading, Widget
from widgets.layouts import VBoxLayout


class Matisse(Widget):
    """
    MarketSPEED 2 RSS 用いた信用取引テスト
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # 信用取引テスト用 Excel ファイル
        self.excel_path = 'margin_transaction_test.xlsm'

        # GUI
        layout = VBoxLayout()
        self.setLayout(layout)
        panel = PanelTrading()
        panel.clickedBuy.connect(self.on_buy)
        panel.clickedRepay.connect(self.on_repay)
        panel.clickedSell.connect(self.on_sell)
        layout.addWidget(panel)

    def on_buy(self):
        """
        建玉の買建
        :return:
        """
        self.logger.info("「買建」ボタンがクリックされました。")

    def on_repay(self):
        """
        建玉の返済
        :return:
        """
        self.logger.info("「返済」ボタンがクリックされました。")

    def on_sell(self):
        """
        建玉の売建
        :return:
        """
        self.logger.info("「売建」ボタンがクリックされました。")


def main():
    app = QApplication(sys.argv)
    win = Matisse()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # ロギング設定を適用（ルートロガーを設定）
    main_logger = setup_logging()
    main()
