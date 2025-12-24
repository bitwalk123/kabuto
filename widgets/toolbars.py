import datetime
import os

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QStyle,
    QToolBar,
)

from funcs.ios import (
    get_sheets_in_excel,
    save_setting,
)
from funcs.setting import get_default_setting, load_setting
from funcs.tse import get_ticker_name_list
from structs.app_enum import AppMode
from structs.res import AppRes
from widgets.buttons import ButtonGroup, RadioButton
from widgets.combos import ComboBox
from widgets.containers import FrameSunken, PadH
from widgets.dialogs import DlgCodeSel, DlgParam, DlgTickFileSel
from widgets.labels import LCDTime, Label
from widgets.layouts import HBoxLayout


class ToolBar(QToolBar):
    clickedAbout = Signal()
    clickedPlay = Signal()
    clickedStop = Signal()
    clickedTransaction = Signal()
    selectedExcelFile = Signal(str, list)

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        self.code_default = "7011"  # デフォルトの銘柄コード

        # デバッグ（レビュー）モード時のみ
        if res.debug:
            # Excel ファイルを開く
            action_open = QAction(
                QIcon(os.path.join(res.dir_image, "excel.png")),
                "Excel ファイルを開く",
                self
            )
            action_open.triggered.connect(self.on_select_excel)
            self.addAction(action_open)

            self.addSeparator()

            # タイマー開始
            action_play = QAction(
                QIcon(os.path.join(res.dir_image, "play.png")),
                "タイマー開始",
                self
            )
            action_play.triggered.connect(self.on_play)
            self.addAction(action_play)

            # タイマー停止
            action_stop = QAction(
                QIcon(os.path.join(res.dir_image, "stop.png")),
                "タイマー停止",
                self
            )
            action_stop.triggered.connect(self.on_stop)
            self.addAction(action_stop)

        # 取引履歴
        self.action_transaction = action_transaction = QAction(
            QIcon(os.path.join(res.dir_image, "transaction.png")),
            "取引履歴",
            self
        )
        action_transaction.setEnabled(False)
        action_transaction.triggered.connect(self.on_transaction)
        self.addAction(action_transaction)

        # このアプリについて
        self.action_about = action_about = QAction(
            QIcon(os.path.join(res.dir_image, "about.png")),
            "このアプリについて",
            self
        )
        action_about.triggered.connect(self.on_about)
        self.addAction(action_about)

        pad = PadH()
        self.addWidget(pad)

        lab_time = Label("システム時刻 ")
        self.addWidget(lab_time)

        self.lcd_time = lcd_time = LCDTime()
        self.addWidget(lcd_time)

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

    def on_select_excel(self):
        # ティックデータ（Excel ファイル）の選択ダイアログ
        dlg_file = DlgTickFileSel(self.res)
        if dlg_file.exec():
            path_excel = dlg_file.selectedFiles()[0]
        else:
            return

        # 対象の Excel ファイルのシート一覧
        list_code = get_sheets_in_excel(path_excel)
        # 銘柄コードに対応する銘柄名の取得
        dict_ticker = get_ticker_name_list(list_code)
        # 「銘柄名 (銘柄コード)」の文字列リスト
        list_ticker = [f"{dict_ticker[code]} ({code})" for code in dict_ticker.keys()]
        # デフォルトの銘柄コードの要素のインデックス
        idx_default = list_code.index(self.code_default)
        # シミュレーション対象の銘柄を選択するダイアログ
        dlg_code = DlgCodeSel(list_ticker, idx_default)
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

    def on_transaction(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「取引履歴」ボタンがクリックされたことを通知
        self.clickedTransaction.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_transaction(self):
        """
        取引履歴の表示ボタンを Enable にする
        :return:
        """
        self.action_transaction.setEnabled(True)

    def updateTime(self, ts: float):
        dt = datetime.datetime.fromtimestamp(ts)
        self.lcd_time.display(f"{dt.hour:02}:{dt.minute:02}:{dt.second:02}")


class ToolBarProphet(QToolBar):
    """
    Prophet 用ツールバー
    """
    clickedDebug = Signal()
    clickedPlay = Signal()
    clickedUpdate = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self.dir_collection = self.res.dir_collection

        action_start = QAction(
            QIcon(os.path.join(res.dir_image, "play.png")),
            "処理開始",
            self
        )
        action_start.triggered.connect(self.on_start)
        self.addAction(action_start)

        self.addSeparator()

        lab_tick = Label("ティックデータ")
        lab_tick.setStyleSheet("QLabel {padding: 0 5px 0 5px;}")
        self.addWidget(lab_tick)

        self.combo_tick = combo_tick = ComboBox()
        combo_tick.setToolTip("ティックデータ一覧")
        combo_tick.addItems(self.getListTicks())
        combo_tick.currentTextChanged.connect(self.on_file_changed)
        self.addWidget(combo_tick)

        self.addSeparator()

        lab_code = Label("銘柄コード")
        lab_code.setStyleSheet("QLabel {padding: 0 5px 0 5px;}")
        self.addWidget(lab_code)

        self.combo_code = combo_code = ComboBox()
        combo_code.setToolTip("銘柄コード一覧")
        self.addWidget(combo_code)

        action_setting = QAction(
            QIcon(os.path.join(res.dir_image, "setting.png")),
            "パラメータ設定",
            self
        )
        action_setting.triggered.connect(self.on_setting)
        self.addAction(action_setting)

        self.addSeparator()

        frame = FrameSunken()
        frame.setStyleSheet("""
            QFrame {
                padding-left: 0.5em;
                padding-right: 0.5em;
            }
        """)
        self.addWidget(frame)
        hbox = HBoxLayout()
        hbox.setSpacing(5)
        frame.setLayout(hbox)

        rb_single = RadioButton("single")
        rb_single.toggle()
        hbox.addWidget(rb_single)

        rb_all = RadioButton("all")
        hbox.addWidget(rb_all)

        rb_doe = RadioButton("doe")
        hbox.addWidget(rb_doe)

        self.rb_group = rb_group = ButtonGroup()
        rb_group.addButton(rb_single)
        rb_group.addButton(rb_all)
        rb_group.addButton(rb_doe)

        self.addSeparator()

        pad = PadH()
        self.addWidget(pad)

        action_debug = QAction(
            QIcon(os.path.join(res.dir_image, 'debug.png')),
            "デバッグ用",
            self
        )
        action_debug.triggered.connect(self.on_debug)
        self.addAction(action_debug)

        # GUI が確定された後に処理
        QTimer.singleShot(0, self.on_file_changed)

    def get_code(self) -> str:
        return self.combo_code.currentText()

    def get_list_code(self):
        """
        銘柄コード一覧の取得
        :return:
        """
        excel = self.combo_tick.currentText()
        path_excel = os.path.join(self.res.dir_collection, excel)
        list_code = get_sheets_in_excel(path_excel)
        self.combo_code.clear()
        self.combo_code.addItems(list_code)

    def getInfo(self) -> dict:
        """
        選択されている情報を辞書にして返す
        :return:
        """
        dict_info = dict()

        # ティックデータ
        excel = self.combo_tick.currentText()
        path_excel = os.path.join(self.dir_collection, excel)
        dict_info["path_excel"] = path_excel

        # 銘柄コード
        code = self.get_code()
        dict_info["code"] = code

        # 銘柄コード別設定ファイルの取得
        dict_info["param"] = load_setting(self.res, code)

        # 処理モード single/all/doe
        rb = self.rb_group.checkedButton()
        mode = rb.text()
        if mode == "single":
            dict_info["mode"] = AppMode.SINGLE
        elif mode == "all":
            dict_info["mode"] = AppMode.ALL
        elif mode == "doe":
            dict_info["mode"] = AppMode.DOE
        else:
            raise TypeError(f"Unknown mode: {mode}")

        return dict_info

    def getListTicks(self, reverse: bool = True) -> list[str]:
        """
        ティックデータ一覧の取得
        :return:
        """
        list_tick = sorted(os.listdir(self.dir_collection), reverse=reverse)
        return list_tick

    def on_debug(self):
        self.clickedDebug.emit()

    def on_file_changed(self, *args):
        self.get_list_code()

    def on_setting(self):
        code = self.get_code()

        # 銘柄コード別設定ファイルの取得
        dict_setting = load_setting(self.res, code)
        dlg = DlgParam(self.res, code, dict_setting)
        if dlg.exec():
            print('OK ボタンがクリックされました。')
            dict_param = dlg.getParam()
            save_setting(self.res, code, dict_param)
            print(dict_param)
        else:
            print('Cancel ボタンがクリックされました。')

    def on_start(self):
        self.clickedPlay.emit()

    def on_update(self):
        self.clickedUpdate.emit()


class ToolBarTransaction(QToolBar):
    transdataSelected = Signal(str)
    saveClicked = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        action_save = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "取引履歴を保存する",
            self
        )
        action_save.triggered.connect(self.on_save)
        self.addAction(action_save)

        action_open = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
            "Excel ファイル（取引履歴）を開く",
            self
        )
        action_open.triggered.connect(self.on_select_excel)
        self.addAction(action_open)

    def on_save(self):
        # ----------------------------------------------
        # 🧿 「取引履歴を保存する」ボタンがクリックされたことを通知
        self.saveClicked.emit()
        # ----------------------------------------------

    def on_select_excel(self):
        excel_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            self.res.dir_transaction,
            "Excel File (*.xlsx)"
        )
        if excel_path == "":
            return
        else:
            # ----------------------------------
            # 🧿 Excel ファイルが選択されたことの通知
            self.transdataSelected.emit(excel_path)
            # ----------------------------------


class ToolBarVein(QToolBar):
    def __init__(self, res: AppRes):
        super().__init__()
        self.setFixedHeight(32)
        self.res = res

        hpad = PadH()
        self.addWidget(hpad)

        lab_time = Label("システム時刻 ")
        self.addWidget(lab_time)

        self.lcd_time = lcd_time = LCDTime()
        self.addWidget(lcd_time)

    def updateTime(self, ts: float):
        dt = datetime.datetime.fromtimestamp(ts)
        self.lcd_time.display(f"{dt.hour:02}:{dt.minute:02}:{dt.second:02}")
