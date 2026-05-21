from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from funcs.conv import position_to_onehot
from modules.env_data import EnvData
from modules.observatory import ObservationManager
from modules.posman import PositionManager
from structs.app_enum import ActionType, PositionType


class TradingEnv(gym.Env):
    """
    取引用環境クラス
    """

    def __init__(self, code: str, dict_setting: dict[str, Any]):
        super().__init__()
        self.CODE: str = code  # 銘柄コード
        self.dict_setting = dict_setting  # パラメータ辞書

        # step メソッドで渡される状態辞書
        self.states: dict = {}

        # データクラスのインスタンスを定義
        if self.dict_setting is None:
            self.s = EnvData()
        else:
            self.s = EnvData(**self.dict_setting)

        # 観測値管理クラス
        self.obs_man = ObservationManager(self.s)

        # ポジション・マネージャ
        self.posman = posman = PositionManager()
        posman.initPosition([self.CODE])

        # ====== 行動空間 action_space の定義 ======
        n_action_space = len(ActionType)
        self.action_space = spaces.Discrete(n_action_space)

        # ====== 観測（特徴量）空間 observation_space の定義 ======
        """
        【観測値】- VecNormalize Wrapper を使用する前提
        [market] - VecNormalize Wrapper で標準化
        1. MA1（短周期移動平均）
        2. MA2（長周期移動平均）
        3. Momentum（モメンタム）
        4. Profit（含み損益）
        5. ProfitMax（最大含み損益）
        6. n_trade（約定回数）
        7. count_negative（含み損の継続カウンタ）
        8. 約定コスト
        9. dd_ratio（ドローダウン率）
        [cross] - 符号が重要であるため標準化しない (-1, 1)
        1. DiffMA（乖離率 : (MA1 - MA2) / MA2）
        2. DiffVWAP（乖離率 : (MA1 - VWAP) / VWAP）
        3. RSI
        [position] - 標準化不要
        1. SHORT
        2. NONE
        3. LONG
        4. MA Golden Cross
        5. MA Dead Cross
        """
        self.observation_space = spaces.Dict({
            "market": spaces.Box(
                low=np.array([
                    -np.float32('inf'),  # 1. MA1（短周期移動平均）
                    -np.float32('inf'),  # 2. MA2（長周期移動平均）
                    -np.float32('inf'),  # 3. Momentum（モメンタム）
                    -np.float32('inf'),  # 4. Profit（含み損益）
                    -np.float32('inf'),  # 5. ProfitMax（最大含み損益）
                    np.float32(0),  # 6. n_trade（約定回数）
                    np.float32(0),  # 7. count_negative（含み損の継続カウンタ）
                    -np.float32('inf'),  # 8. 約定コスト
                    np.float32(0),  # 9. dd_ratio（ドローダウン率）
                ]),
                high=np.array([
                    np.float32('inf'),  # 1. MA1（短周期移動平均）
                    np.float32('inf'),  # 2. MA2（長周期移動平均）
                    np.float32('inf'),  # 3. Momentum（モメンタム）
                    np.float32('inf'),  # 4. Profit（含み損益）
                    np.float32('inf'),  # 5. ProfitMax（最大含み損益）
                    np.float32(1),  # 6. n_trade（約定回数）
                    np.float32(1),  # 7. count_negative（含み損の継続カウンタ）
                    np.float32(-self.s.COST_CONTRACT),  # 8. 約定コスト
                    np.float32('inf'),  # 9. dd_ratio（ドローダウン率）
                ]),
                shape=(9,),
                dtype=np.float32
            ),
            "cross": spaces.Box(
                low=np.array([
                    np.float32(-5),  # 1. DiffMA（乖離率 : (MA1 - MA2) / MA2）
                    np.float32(-5),  # 2. DiffVWAP（乖離率 : (MA1 - VWAP) / VWAP）
                    np.float32(0),  # 3. RSI
                ]),
                high=np.array([
                    np.float32(5),  # 1. DiffMA（乖離率 : (MA1 - MA2) / MA2）
                    np.float32(5),  # 2. DiffVWAP（乖離率 : (MA1 - VWAP) / VWAP）
                    np.float32(1),  # 3. RSI
                ]),
                shape=(3,),
                dtype=np.float32
            ),
            "signal": spaces.MultiBinary(4),  # signal
            "position": spaces.MultiBinary(3),  # one-hot
        })

    def action_masks(self) -> np.ndarray:
        """
        行動マスク
        【マスク】
        - ウォーミングアップ期間 → 強制 HOLD
        - ナンピン取引の禁止

        :return: mask
        """
        return self.s.get_masks()

    def forceRepay(self) -> None:
        """
        建玉の強制返済
        :return:
        """
        if self.posman.hasPosition(self.CODE):
            self.position_close_force()

    def getCurrentPosition(self) -> PositionType:
        """
        現在のポジションを返す
        :return:
        """
        return self.s.position

    def getParams(self) -> dict[str, Any]:
        """
        調整可能？なパラメータを辞書で返す
        :return:
        """
        # return self.provider.getSetting()
        return {}

    def getTimestamp(self) -> float:
        return self.s.ts

    def getTransaction(self) -> pd.DataFrame:
        return self.posman.getTransactionResult()

    def getObservation(self, ts: float, price: float, volume: float) -> tuple[dict, dict]:
        """
        観測値を取得（リアルタイム用）
        ティックデータから観測値を算出（デバッグ用）
        :param ts:
        :param price:
        :param volume:
        :return:
        """
        # 観測値
        # return self.s.get_obs(), self.s.get_technicals()
        return self.s.set_data(self.obs_man.update(ts, price, volume))

    def getObsList(self) -> list:
        # return self.obs_man.getObsList()
        return []

    def init_status(self) -> None:
        """
        初期化処理
        :return:
        """
        # データクラスのインスタンスを再定義
        if self.dict_setting is None:
            self.s = EnvData()
        else:
            self.s = EnvData(**self.dict_setting)

        # パラメータの標準出力
        self.s.print_param()

        # ポジション・マネージャのリセットと初期化
        self.posman.reset()
        self.posman.initPosition([self.CODE])

    def position_open(self, action_type: ActionType) -> None:
        """
        ポジションのオープン
        :param action_type:
        :return:
        """
        if "reason" in self.states:
            note = self.states["reason"]
        else:
            note = ""

        self.s.position = self.posman.openPosition(
            self.CODE, self.s.ts, self.s.price, action_type, note
        )
        self.s.n_trade += 1  # 取引回数の更新
        self.s.reset_profit_pre()  # 一つ前の含み益のリセット
        # 【報酬・ペナルティ】
        # r = 0.0
        # r += self.s.add_contract_cost()  # 約定コスト
        self.s.reset_count_post_contract()  # 約定後の経過カウンタのリセット
        # return r

    def position_close(self, note="") -> None:
        """
        ポジションのクローズ
        :param note:
        :return:
        """
        if "reason" in self.states and note == "":
            note = self.states["reason"]

        # ポジション管理
        self.s.position = self.posman.closePosition(
            self.CODE, self.s.ts, self.s.price, note=note
        )
        self.s.n_trade += 1  # 取引回数の更新
        self.s.reset_profit_pre()  # 一つ前の含み益のリセット
        self.s.reset_profit_max()  # 最大含み益のリセット
        # 【報酬】
        # r = 0.0
        # r += self.s.add_contract_cost()  # 約定コスト
        self.s.reset_count_post_contract()  # 約定後の経過カウンタのリセット
        # r += self.s.profit  # 含み損益分そっくり報酬
        self.s.reset_count_negative()
        # return r

    def position_close_force(self, note="強制返済") -> None:
        """
        ポジション・クローズ（強制）
        :param note:
        :return:
        """
        self.position_close(note)

    def reset(self, seed=None, options=None):
        """
        リセット
        :param seed:
        :param options:
        :return:
        """
        # 環境の初期化（常に寄り付きから開始）
        self.init_status()

        # ====== 観測値（状態） ======
        market = np.array(
            [
                1,  # 1. MA1（短周期移動平均）
                1,  # 2. MA2（長周期移動平均）
                0,  # 3. Momentum（モメンタム）
                0,  # 4. Profit（含み損益）
                0,  # 5. ProfitMax（最大含み損益）
                0,  # 6. n_trade（約定回数）
                0,  # 7. count_negative（含み損の継続カウンタ）
                0,  # 8. 約定コスト
                0,  # 9. dd_ratio（ドローダウン率）
            ],
            dtype=np.float32
        )
        cross = np.array([0, 0, 0], dtype=np.float32)
        signal = np.array([False, False, False, False], dtype=np.float32)
        position = position_to_onehot(self.s.position)
        obs = {"market": market, "cross": cross, "signal": signal, "position": position}
        print(obs)

        info = {}  # Additional debug info
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Gymnasium標準のstepメソッド（学習用）

        Note:
            このメソッドは過去データを使った学習・バックテスト用です。
            リアルタイム推論では step_realtime() を使用してください。

        Returns:
            observation: 次の観測値
            reward: 報酬
            terminated: エピソード終了フラグ
            truncated: エピソード打ち切りフラグ
            info: 追加情報
        """
        raise NotImplementedError(
            "標準のstepメソッドは未実装です。"
            "リアルタイム推論には step_realtime() を使用してください。"
            "学習用には TrainingEnv クラスの実装を検討してください。"
        )

    def step_realtime(self, action: int, states: dict) -> tuple[float, bool, bool, dict[str, Any]]:
        """
        アクションによるステップ処理（リアルタイム用）

        Note:
            このメソッドは観測値を返しません。
            観測値は事前に getObservation() で取得済みという前提です。

        Args:
            action: 実行するアクション
            states: アクションの状態を表す辞書

        Returns:
            reward: 報酬
            terminated: エピソード終了フラグ（目標達成など）
            truncated: エピソード打ち切りフラグ（取引上限など）
            info: 追加情報（pnl_total, done_reasonなど）
        """
        # アクションの理由
        if states is None:
            self.states = {}
        else:
            self.states = states

        info: dict[str, Any] = {}

        # アクションに対する報酬
        # reward = self.reward_man.evalReward(action)
        reward = 0

        # ステップ終了判定
        terminated = False
        truncated = False

        """
        # 取引回数上限チェック
        if self.provider.N_TRADE_MAX <= self.provider.getNTrade():
            reward += self.reward_man.forceRepay()
            truncated = True
            info["done_reason"] = "terminated:max_trades"
        """

        # 収益情報
        # info["pnl_total"] = self.provider.getPnLTotal()

        # 一つ前の特徴量の更新
        self.s.update_feature_pre()
        # ステップ（データフレームの行）更新
        self.s.inc_row()

        return reward, terminated, truncated, info

    # === スレッド外部からのコマンド ===
    def openPosition(self, action_type: ActionType):
        self.position_open(action_type)

    def closePosition(self):
        self.position_close()
