import pandas as pd
from PySide6.QtCore import (
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtWidgets import (
    QDockWidget,
    QProgressBar,
    QStyle,
)

from funcs.plugin import load_plugins
from models_profit.abstract import ProfitSimulatorABS
from structs.res import AppRes
from tools.profit_sim_widgets import TimeRange
from tools.profit_sim_worker import PluginWorker
from widgets.buttons import Button
from widgets.combos import ComboBoxModel
from widgets.containers import Widget
from widgets.labels import LabelLeft2
from widgets.layouts import VBoxLayout, HBoxLayout
from widgets.textedits import PlainTextEdit


class ProfitSimulatorDock(QDockWidget):
    requestSelectedData = Signal(pd.Timestamp, pd.Timestamp)

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        self.thread = None
        self.worker = None

        self.code = "0000"
        self.df = pd.DataFrame()

        # プラグインの一覧を取得
        plugins: dict[str, type] = load_plugins(
            path_plugin=res.dir_model_profit,
            package_name=res.dir_model_profit,
            plugin_base_class=ProfitSimulatorABS
        )

        base = Widget()
        self.setWidget(base)
        layout = VBoxLayout()
        base.setLayout(layout)

        self.time_range = time_range = TimeRange(res)
        time_range.notifyTimeRangeFixed.connect(self.on_timerange_fixed)
        layout.addWidget(time_range)

        layout_combo = HBoxLayout()
        layout.addLayout(layout_combo)

        title = LabelLeft2("モデル名")
        layout_combo.addWidget(title)

        self.combo = combo = ComboBoxModel()
        for key in sorted(plugins.keys()):
            cls = plugins[key]
            combo.addItem(key, cls)
        layout_combo.addWidget(combo)

        but_play = Button()
        icon_play = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        but_play.setIcon(icon_play)
        but_play.clicked.connect(self.on_play)
        layout_combo.addWidget(but_play)

        self.pte = pte = PlainTextEdit()
        pte.setReadOnly(True)
        layout.addWidget(pte)

        progbar = QProgressBar()
        layout.addWidget(progbar)

        # ウィジェット配置後にコンボボックスの値を読み取る
        QTimer.singleShot(0, self.read_combo_value)

    def clearTimeRange(self):
        self.time_range.clearTimeRange()

    def on_finished(self, dict_result: dict):
        df_transaction = dict_result["transaction"]
        pnl = df_transaction["損益"].sum()
        print(df_transaction)
        print(f"損益 : {pnl} 円/株")

    def on_play(self):
        cls = self.combo.currentData()
        plugin = cls(self.code, self.df)

        # プラグイン用スレッド
        self.thread = QThread()
        self.worker = PluginWorker(plugin)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_play_old(self):
        # コンボボックスに表示されている名前に対応するクラス
        cls = self.combo.currentData()
        # クラスのインスタンス化
        obj = cls(self.code, self.df)
        # インスタンスの実行
        dict_result = obj.run()

        # 取引結果
        df_transaction = dict_result["transaction"]
        pnl = df_transaction["損益"].sum()
        print(df_transaction)
        print(f"損益 : {pnl} 円/株")

    def on_timerange_fixed(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        self.requestSelectedData.emit(dt1, dt2)

    def read_combo_value(self):
        cls = self.combo.currentData()
        desc = cls.DESC
        self.pte.setPlainText(desc)

    def setDataFrameSelected(self, code: str, df: pd.DataFrame):
        self.code = code
        self.df = df
        print("銘柄コード", code)
        print(df)

    def setTimeRange(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        self.time_range.setTimeRange(dt1, dt2)
