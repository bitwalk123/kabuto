import datetime
import glob
import logging
import os
from collections import defaultdict
from typing import Any, Literal, TypeAlias

import pandas as pd

from funcs.commons import (
    check_doe_factor_match,
    get_datestr_from_collections,
    get_dt_from_collections,
)
from funcs.excel import is_sheet_exists, load_excel
from funcs.plot import draw_review_chart
from funcs.setting import load_setting, update_setting
from funcs.tide import get_intraday_timestamp
from funcs.tse import get_ticker_name_list
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
        # 銘柄名の取得
        self.name: str = get_ticker_name_list([code])[code]

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
        self.dir_base = os.path.join(res.dir_doe, name_doe, code)
        os.makedirs(self.dir_base, exist_ok=True)

        # ポジション・マネージャ（使い回す）
        self.posman = PositionManager()

    def run(self) -> None:
        self.logger.info(self.name_doe)

        # ティックファイル
        path_glob = os.path.join(self.res.dir_collection, f"*.xlsx")
        list_excel = sorted(glob.glob(path_glob), reverse=True)
        for path_excel in list_excel:
            # 日付型
            dt_date = get_dt_from_collections(path_excel)
            if dt_date < self.dt_start:
                # 最初の日付より前であればループをスキップ
                continue
            if not is_sheet_exists(path_excel, self.code):
                # Excel ブックに対象銘柄のシートが存在しなければループをスキップ
                continue

            # 日付文字列
            str_date = get_datestr_from_collections(path_excel)
            # 日付フォルダ
            dir_date = os.path.join(self.dir_base, str_date)
            if os.path.isdir(dir_date):
                # 存在していればループをスキップ
                continue
            # 存在していなければフォルダ生成
            os.mkdir(dir_date)

            # ザラ場の開始時間などのタイムスタンプ
            dict_ts = get_intraday_timestamp(path_excel)
            # print(dict_ts)

            # Excel 名
            print(f"\n{path_excel}")

            # Excel シートの読み込み
            dict_sheet = load_excel(path_excel)
            # 対象シートのデータフレーム
            df: pd.DataFrame = dict_sheet[self.code]

            # DOE 結果
            dict_results = defaultdict(list)

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
                    dir_date, f"{self.code}_{r:03d}_technicals.csv"
                )
                df_technicals = agent.saveTechnicals(path_csv)

                # テクニカル・データのレビュー・チャート
                title = f"{dt_date.date()}: {self.name} ({self.code}) - #{r:03d}"
                path_img = os.path.join(
                    dir_date, f"{self.code}_{r:03d}_technicals.png"
                )
                draw_review_chart(
                    self.res,
                    title,
                    df_technicals,
                    dict_setting_doe,
                    dict_ts,
                    path_img,
                )

                # 結果の保持
                dict_results["date"].append(dt_date)
                dict_results["run"].append(r)
                for key in dict_setting_doe.keys():
                    dict_results[key].append(dict_setting_doe[key])
                dict_results["trade"].append(n)
                dict_results["total"].append(total)

            df_results = pd.DataFrame(dict_results)
            print(df_results)

            # DOE 結果の保存
            df_results.to_csv(
                os.path.join(dir_date, f"{self.code}_result.csv"),
                index=False
            )

            # HTML の出力
            if self.name_doe == "doe-001":
                df_results_extract = df_results[["date", "run", "DD_PROFIT", "DD_RATIO", "trade", "total"]]
                (
                    df_results_extract.style
                    .set_table_attributes('class="simple" style="font-family: monospace; font-size: small;"')
                    .set_properties(**{'text-align': 'right'})
                    .format(precision=2, thousands=",")
                    .to_html(
                        os.path.join(dir_date, f"{self.code}_result.html"),
                        index=False
                    )
                )

    def simulation(
            self,
            agent: AgentCLI,
            df: pd.DataFrame,
            dict_ts: dict[str, Any]
    ) -> tuple[int, float]:
        self.posman.reset()  # ポジション・マネージャのリセット
        self.posman.initPosition([self.code])  # ポジション・マネージャの初期化

        # ティックデータのループ
        ts, price = (0.0, 0.0)
        for row in df.itertuples():
            ts = row.Time

            if dict_ts["end_entry"] < ts:
                # 指定時間以降はエントリをしない。
                if agent.isAutoPilot():
                    agent.setAutoPilot(False)
                    self.logger.info(f"エントリを無効にしました。")

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
