import datetime
import logging
import os
from pathlib import Path

from PySide6.QtCore import Signal, QThread
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QToolBar, QDialog

from funcs.ios import get_sheets_in_excel
from funcs.tse import get_ticker_name_list
from modules.uploader import UploadWorker
from structs.res import AppRes
from widgets.buttons import CheckBox
from widgets.containers import PadH
from widgets.dialogs import DlgTickFileSel, DlgCodeSel
from widgets.labels import Label, LCDTime


class ToolBar(QToolBar):
    """
    Kabuto 本体のツールバー
    """
    clickedAbout = Signal()
    clickedPlay = Signal()
    clickedStop = Signal()
    clickedTransaction = Signal()
    selectedExcelFile = Signal(str, list)

    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res

        self.thread = None
        self.worker = None

        # デバッグ（レビュー）モード時のみ
        if res.debug:
            # Excel ファイルを開く
            self.excel = action_open = QAction(
                QIcon(os.path.join(res.dir_image, "excel.png")),
                "Excel ファイルを開く",
                self
            )
            action_open.triggered.connect(self.on_select_excel)
            self.addAction(action_open)

            self.addSeparator()

            # タイマー開始
            self.action_play = action_play = QAction(
                QIcon(os.path.join(res.dir_image, "play.png")),
                "タイマー開始",
                self
            )
            action_play.setEnabled(False)
            action_play.triggered.connect(self.on_play)
            self.addAction(action_play)

            # タイマー停止
            self.action_stop = action_stop = QAction(
                QIcon(os.path.join(res.dir_image, "stop.png")),
                "タイマー停止",
                self
            )
            action_stop.triggered.connect(self.on_stop)
            action_stop.setEnabled(False)
            self.addAction(action_stop)

            # 設定ファイルのアップロード
            action_upload = QAction(
                QIcon(os.path.join(res.dir_image, "upload.png")),
                "設定ファイルのアップロード",
                self
            )
            action_upload.triggered.connect(self.on_upload)
            self.addAction(action_upload)

        # 取引履歴
        self.action_transaction = action_transaction = QAction(
            QIcon(os.path.join(res.dir_image, "transaction.png")),
            "取引履歴",
            self
        )
        action_transaction.setEnabled(False)
        action_transaction.triggered.connect(self.on_transaction)
        self.addAction(action_transaction)

        # システム設定
        self.action_setting = action_setting = QAction(
            QIcon(os.path.join(res.dir_image, "setting.png")),
            "システム設定",
            self
        )
        action_setting.triggered.connect(self.on_setting)
        self.addAction(action_setting)

        # このアプリについて
        self.action_about = action_about = QAction(
            QIcon(os.path.join(res.dir_image, "about.png")),
            "このアプリについて",
            self
        )
        action_about.triggered.connect(self.on_about)
        self.addAction(action_about)

        self.addSeparator()

        # バックアップ稼働かの識別用
        self.check_alt = check_alt = CheckBox("控")
        self.addWidget(check_alt)

        pad = PadH()
        self.addWidget(pad)

        lab_time = Label("システム時刻 ")
        self.addWidget(lab_time)

        self.lcd_time = lcd_time = LCDTime()
        self.addWidget(lcd_time)

    def closeEvent(self, event):
        thread = getattr(self, "thread", None)
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait()
        super().closeEvent(event)

    def isAlt(self) -> bool:
        # バックアップ用に稼働しているかどうか
        return self.check_alt.isChecked()

    def on_about(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「このアプリについて」ボタンがクリックされたことを通知
        self.clickedAbout.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_play(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「タイマー開始」ボタンがクリックされたことを通知
        self.clickedPlay.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Play / Stop ボタンの状態変更
        self.switch_playstop(False)

    def on_setting(self):
        print("未実装です。")

    def on_select_excel(self):
        """
        ティックデータを保持した Excel ファイルの選択
        :return:
        """
        # ティックデータ（Excel ファイル）の選択ダイアログ
        dlg_file = DlgTickFileSel(self.res)
        if dlg_file.exec():
            path_excel = dlg_file.selectedFiles()[0]
        else:
            return

        # Excel アイコンを Disable に
        self.excel.setDisabled(True)

        # 対象の Excel ファイルのシート一覧
        list_code = get_sheets_in_excel(path_excel)
        # 銘柄コードに対応する銘柄名の取得
        dict_name = get_ticker_name_list(list_code)
        # 「銘柄名 (銘柄コード)」の文字列リスト
        list_ticker = [f"{dict_name[code]} ({code})" for code in dict_name.keys()]
        # デフォルトの銘柄コードの要素のインデックス
        idx_default = list_code.index(self.res.code_default)
        # シミュレーション対象の銘柄を選択するダイアログ
        dlg_code = DlgCodeSel(self.res, list_ticker, idx_default)
        if dlg_code.exec() == QDialog.DialogCode.Accepted:
            list_code_selected = [list_code[r] for r in dlg_code.getSelected()]
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 Excel ファイルが選択されたことの通知
            self.selectedExcelFile.emit(path_excel, list_code_selected)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_stop(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「タイマー停止」ボタンがクリックされたことを通知
        self.clickedStop.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.action_stop.setDisabled(True)

    def on_transaction(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「取引履歴」ボタンがクリックされたことを通知
        self.clickedTransaction.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_upload(self):
        """
        現在の JSON ファイルを HTTP サーバーへアップロード
        :return:
        """
        local_conf = Path(self.res.dir_conf)
        files = local_conf.glob("*.json")
        # スレッド処理
        self.thread = thread = QThread()
        self.worker = worker = UploadWorker(self.res, files)
        worker.moveToThread(thread)
        # スレッドが開始されたら処理開始
        thread.started.connect(worker.run)
        # 処理が終わったら削除
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        # 終了後の処理
        worker.finished.connect(self.upload_completed)
        # スレッドの開始
        thread.start()

    def set_transaction(self):
        """
        取引履歴の表示ボタンを Enable にする
        :return:
        """
        self.action_transaction.setEnabled(True)

    def switch_playstop(self, state: bool):
        self.action_play.setEnabled(state)
        self.action_stop.setDisabled(state)

    def updateTime(self, ts: float):
        dt = datetime.datetime.fromtimestamp(ts)
        self.lcd_time.display(f"{dt.hour:02}:{dt.minute:02}:{dt.second:02}")

    def upload_completed(self):
        """
        アップロード終了メッセージ
        :return:
        """
        self.logger.info(f"{__name__}: アップロードが完了しました。")
