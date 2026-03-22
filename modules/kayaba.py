import datetime
import glob
import logging
import os
from collections import defaultdict
from typing import Any, Literal, TypeAlias

import pandas as pd
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from funcs.commons import (
    check_doe_factor_match,
    get_datestr_from_collections,
    get_dt_from_collections,
)
from funcs.excel import is_sheet_exists, load_excel
from funcs.plot import plot_price_vwap, plot_rsi, plot_momentum, plot_profit, plot_drawdown, plot_verticals
from funcs.setting import load_setting, update_setting
from funcs.tide import get_intraday_timestamp
from modules.agent_cli import AgentCLI
from modules.kabuto import Kabuto
from modules.posman import PositionManager
from structs.app_enum import ActionType, PositionType
from structs.res import AppRes

# 型エイリアスの定義（クラスの外に配置）
TradeAction: TypeAlias = Literal["doBuy", "doSell", "doRepay"]
TradeKey: TypeAlias = tuple[ActionType, PositionType]


class Kayaba:
    __app_name__ = "Kayaba"
    __version__ = Kabuto.__version__
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    # --- 状態遷移表 ---
    ACTION_DISPATCH: dict[TradeKey, TradeAction] = {
        (ActionType.BUY, PositionType.NONE): "doBuy",  # 建玉がなければ買建
        (ActionType.BUY, PositionType.SHORT): "doRepay",  # 売建（ショート）であれば（買って）返済
        (ActionType.SELL, PositionType.NONE): "doSell",  # 建玉がなければ売建
        (ActionType.SELL, PositionType.LONG): "doRepay",  # 買建（ロング）であれば（売って）返済
        # HOLD は何もしないので載せない
    }

    def __init__(self, name_doe: str, code: str, dt_start: datetime.datetime) -> None:
        self.logger = logging.getLogger(__name__)
        self.name_doe = name_doe
        self.code = code
        self.dt_start = dt_start
        self.res = res = AppRes()

        path_csv = os.path.join(res.dir_doe, f"{name_doe}.csv")
        self.df_doe = pd.read_csv(path_csv)

        # 最新のパラメータへ更新
        update_setting(res, code)
        # パラメータの読み込み
        self.dict_setting: dict[str, Any] = load_setting(res, code)

        # 実験表と設定ファイルを比較、因子が一致していることが前提
        if not check_doe_factor_match(self.df_doe, self.dict_setting):
            self.logger.error("DOE の因子と設定ファイルの因子が一致しませんでした。")
            raise

        print("下記の条件で DOE を実施します。")
        print(self.df_doe)

        # ポジション・マネージャ（使い回す）
        self.posman = PositionManager()

    def run(self) -> None:
        self.logger.info(self.name_doe)

        # DOE 結果
        dict_results = defaultdict(list)

        # ティックファイル
        path_glob = os.path.join(self.res.dir_collection, f"*.xlsx")
        list_excel = sorted(glob.glob(path_glob))
        for path_excel in list_excel:
            # 日付型
            dt_date = get_dt_from_collections(path_excel)
            if dt_date < self.dt_start:
                continue
            if not is_sheet_exists(path_excel, self.code):
                continue

            # 日付文字列
            str_date = get_datestr_from_collections(path_excel)

            # ザラ場の開始時間などのタイムスタンプ
            dict_ts = get_intraday_timestamp(path_excel)
            # print(dict_ts)

            # Excel 名
            print(f"\n{path_excel}")

            # Excel シートの読み込み
            dict_sheet = load_excel(path_excel)
            # 対象シートのデータフレーム
            df: pd.DataFrame = dict_sheet[self.code]

            for r in range(len(self.df_doe)):
                # DOE 条件のパラメータ・セット
                dict_setting_doe: dict[str, Any] = {}
                for key in self.dict_setting.keys():
                    dict_setting_doe[key] = self.df_doe[key].iloc[r]

                # シミュレーション用エージェント（パラメータ毎に生成し直す）
                agent = AgentCLI(self.code, dict_setting_doe)
                agent.setAutoPilot(True)  # エージェントの自動売買をオン
                agent.resetEnv()  # エージェントのリセット

                # シミュレーション
                n, total = self.simulation(agent, df, dict_ts)
                self.logger.info(
                    f"{r:d}: {str_date}: 売買回数: {n} 回, 損益: {total:.0f} 円"
                )

                # テクニカル・データ（出力先）
                path_csv = os.path.join(
                    self.res.dir_temp, f"{str_date}_{self.code}_{r:d}_technicals.csv"
                )
                self.saveTechnicals(agent, path_csv)

                # 結果の保持
                dict_results["date"].append(dt_date)
                dict_results["run"].append(r)
                for key in dict_setting_doe.keys():
                    dict_results[key].append(dict_setting_doe[key])
                dict_results["trade"].append(n)
                dict_results["total"].append(total)

        # print(f"総収益: {grand_total} 円, {days} 日")
        print(pd.DataFrame(dict_results))

    def simulation(self, agent: AgentCLI, df: pd.DataFrame, dict_ts: dict[str, float]) -> tuple[int, Any]:
        # ポジション・マネージャのリセット
        self.posman.reset()
        # ポジション・マネージャの初期化
        self.posman.initPosition([self.code])

        # ティックデータのループ
        ts, price = (0.0, 0.0)
        for row in df.itertuples():
            ts = row.Time
            if dict_ts["end_2h"] < ts:
                break
            price = row.Price
            volume = row.Volume

            action, position = agent.addData(ts, price, volume)
            action_enum = ActionType(action)
            if action_enum != ActionType.HOLD:
                method_name = self.ACTION_DISPATCH.get((action_enum, position))
                if method_name is None:
                    self.logger.error(f"trade rule violation! action={action_enum}, pos={position}")
                    break
                getattr(self, method_name)(ts, price)

        # 強制返済
        self.forceRepay(ts, price)

        # 取引明細
        df_transaction = self.posman.getTransactionResult()
        return len(df_transaction), df_transaction["損益"].sum()

    def doBuy(self, ts, price) -> None:
        self.posman.openPosition(self.code, ts, price, ActionType.BUY)

    def doSell(self, ts, price) -> None:
        self.posman.openPosition(self.code, ts, price, ActionType.SELL)

    def doRepay(self, ts, price) -> None:
        self.posman.closePosition(self.code, ts, price)

    def forceRepay(self, ts, price) -> None:
        if self.posman.hasPosition(self.code):
            self.doRepay(ts, price)

    @staticmethod
    def saveTechnicals(agent: AgentCLI, path_csv: str) -> None:
        # テクニカル・データの保存
        agent.saveTechnicals(path_csv)

    def draw_chart(
            self,
            title: str,
            df: pd.DataFrame,
            dict_setting: dict[str, Any],
            dict_ts: dict[str, Any],
            name_img: str,
    ):
        IMAGE_WIDTH = 680
        IMAGE_HEIGHT = 700

        # Matplotlib の共通設定
        fm.fontManager.addfont(self.res.path_monospace)

        # FontPropertiesオブジェクト生成（名前の取得のため）
        font_prop = fm.FontProperties(fname=self.res.path_monospace)
        font_prop.get_name()

        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["font.size"] = 9

        fig = Figure(figsize=(IMAGE_WIDTH / 100., IMAGE_HEIGHT / 100.))
        # キャンバスを表示
        canvas = FigureCanvas(fig)

        n = 5
        ax = dict()
        gs = fig.add_gridspec(
            n, 1, wspace=0.0, hspace=0.0,
            height_ratios=[1.5 if i == 0 else 1 for i in range(n)],
        )
        for i, axis in enumerate(gs.subplots(sharex="col")):
            ax[i] = axis
            ax[i].grid(axis="y")

        # 1. 株価と VWAP
        plot_price_vwap(self.ax[0], df, title, dict_ts)

        # 2. モメンタム
        plot_rsi(self.ax[1], df, dict_setting)

        # 3. モメンタム
        plot_momentum(self.ax[2], df, dict_setting)

        # 4. 含み益
        plot_profit(self.ax[3], df, dict_setting)

        # 5. ドローダウン
        plot_drawdown(self.ax[4], df, dict_setting)

        # --- クロス・シグナル、その他縦線系 ---
        plot_verticals(self.n, self.ax, df, dict_setting, dict_ts)

        # タイト・レイアウト
        self.fig.tight_layout()

        # 再描画
        canvas.draw()

        # 保存だけ実行
        self.fig.savefig(name_img, dpi=100)
        print(f"{name_img} に保存しました。")
