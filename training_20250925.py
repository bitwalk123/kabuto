import datetime
import gymnasium as gym
import numpy as np
import pandas as pd
import talib as ta
from enum import Enum
import os
# Stable-Baselines3からのインポート
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


# ==============================================================================
# ユーザー提供のクラス定義 (変更なし)
# ==============================================================================

class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    REPAY = 3


class PositionType(Enum):
    NONE = 0
    LONG = 1
    SHORT = 2


class TransactionManager:
    # ナンピンをしない（建玉を１単位しか持たない）売買管理クラス
    def __init__(self):
        # modified on 20250922
        self.reward_sell_buy = 0.1  # 約定ボーナスまたはペナルティ（買建、売建）
        self.penalty_repay = -0.05  # 約定ボーナスまたはペナルティ（返済）
        self.reward_pnl_scale = 0.3  # 含み損益のスケール（含み損益✕係数）
        self.reward_hold = 0.001  # 建玉を保持する報酬
        self.penalty_none = -0.001  # 建玉を持たないペナルティ
        self.penalty_rule = -1.0  # 売買ルール違反

        # 売買ルール違反カウンター
        self.penalty_count = 0  # 売買ルール違反ペナルティを繰り返すとカウントを加算

        self.action_pre = ActionType.HOLD
        self.position = PositionType.NONE
        self.price_entry = 0.0
        self.pnl_total = 0.0

        self.dict_transaction = self._init_transaction()
        self.code: str = '7011'
        self.unit: int = 1

    @staticmethod
    def _init_transaction() -> dict:
        return {
            "注文日時": [],
            "銘柄コード": [],
            "売買": [],
            "約定単価": [],
            "約定数量": [],
            "損益": [],
        }

    def _add_transaction(
            self,
            t: float,
            transaction: str,
            price: float,
            profit: float = np.nan,
    ):
        self.dict_transaction["注文日時"].append(self._get_datetime(t))
        self.dict_transaction["銘柄コード"].append(self.code)
        self.dict_transaction["売買"].append(transaction)
        self.dict_transaction["約定単価"].append(price)
        self.dict_transaction["約定数量"].append(self.unit)
        self.dict_transaction["損益"].append(profit)

    @staticmethod
    def _get_datetime(t: float) -> str:
        return str(datetime.datetime.fromtimestamp(int(t)))

    def clearAll(self):
        """
        初期状態に設定
        :return:
        """
        self.resetPosition()
        self.action_pre = ActionType.HOLD
        self.pnl_total = 0.0
        self.dict_transaction = self._init_transaction()

    def resetPosition(self):
        """
        ポジション（建玉）をリセット
        :return:
        """
        self.position = PositionType.NONE
        self.price_entry = 0.0

    def setAction(self, action: ActionType, t: float, price: float) -> float:
        reward = 0.0
        if action == ActionType.HOLD:
            # ■■■ HOLD: 何もしない
            # 建玉があれば含み損益から報酬を付与、無ければ少しばかりの保持ボーナス
            reward += self._calc_reward_pnl(price)
            # 売買ルールを遵守した処理だったのでペナルティカウントをリセット
            self.penalty_count = 0
        elif action == ActionType.BUY:
            # ■■■ BUY: 信用買い
            if self.position == PositionType.NONE:
                # === 建玉がない場合 ===
                # 買建 (LONG)
                self.position = PositionType.LONG
                self.price_entry = price
                # print(get_datetime(t), "買建", price)
                self._add_transaction(t, "買建", price)
                # 約定ボーナス付与（買建）
                reward += self.reward_sell_buy
                # 売買ルールを遵守した処理だったのでペナルティカウントをリセット
                self.penalty_count = 0
            else:
                # ○○○ 建玉がある場合 ○○○
                # 建玉があるので、含み損益から報酬を付与
                reward += self._calc_reward_pnl(price)
                # ただし、建玉があるのに更に買建 (BUY) しようとしたので売買ルール違反ペナルティも付与
                self.penalty_count += 1
                reward += self.penalty_rule * self.penalty_count

        elif action == ActionType.SELL:
            # ■■■ SELL: 信用空売り
            if self.position == PositionType.NONE:
                # === 建玉がない場合 ===
                # 売建 (SHORT)
                self.position = PositionType.SHORT
                self.price_entry = price
                # print(get_datetime(t), "売建", price)
                self._add_transaction(t, "売建", price)
                # 約定ボーナス付与（売建）
                reward += self.reward_sell_buy
                # 売買ルールを遵守した処理だったのでペナルティカウントをリセット
                self.penalty_count = 0
            else:
                # ○○○ 建玉がある場合 ○○○
                # 建玉があるので、含み損益から報酬を付与
                reward += self._calc_reward_pnl(price)
                # ただし、建玉があるのに更に売建しようとしたので売買ルール違反ペナルティも付与
                self.penalty_count += 1
                reward += self.penalty_rule * self.penalty_count

        elif action == ActionType.REPAY:
            # ■■■ REPAY: 建玉返済
            if self.position != PositionType.NONE:
                # ○○○ 建玉がある場合 ○○○
                if self.position == PositionType.LONG:
                    # 実現損益（売埋）
                    profit = price - self.price_entry
                    # print(get_datetime(t), "売埋", price, profit)
                    self._add_transaction(t, "売埋", price, profit)
                else:
                    # 実現損益（買埋）
                    profit = self.price_entry - price
                    # print(get_datetime(t), "買埋", price, profit)
                    self._add_transaction(t, "買埋", price, profit)
                # ポジション状態をリセット
                self.resetPosition()
                # 総収益を更新
                self.pnl_total += profit
                # 報酬に収益を追加
                reward += profit
                # 約定ペナルティ付与（返済）
                # reward += self.penalty_repay
                # 売買ルールを遵守した処理だったのでペナルティカウントをリセット
                self.penalty_count = 0
            else:
                # === 建玉がない場合 ===
                # 建玉がないのに建玉を返済しようとしたので売買ルール違反ペナルティを付与
                self.penalty_count += 1
                reward += self.penalty_rule * self.penalty_count
        else:
            raise ValueError(f"{action} is not defined!")

        self.action_pre = action
        return reward

    def _calc_reward_pnl(self, price: float) -> float:
        """
        含み損益に self.reward_pnl_scale を乗じた報酬を算出
        ポジションが無い場合は微小なペナルティを付与
        :param price:
        :return:
        """
        if self.position == PositionType.NONE:
            # PositionType.NONE に対して僅かなペナルティ
            return self.penalty_none
        else:
            reward = 0.0
            if self.position == PositionType.LONG:
                # 含み損益（買建）× 少数スケール
                reward += (price - self.price_entry) * self.reward_pnl_scale
            elif self.position == PositionType.SHORT:
                # 含み損益（売建）× 少数スケール
                reward += (self.price_entry - price) * self.reward_pnl_scale
            reward += self.reward_hold
            return reward


class TradingEnv(gym.Env):
    # 環境クラス
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        # dfはTime, Price, Volume列を持つ
        self.df = df.reset_index(drop=True)
        # ウォームアップ期間
        self.period = 60
        # 特徴量の列名のリストが返る
        self.cols_features = self._add_features(self.period)
        # 現在の行位置
        self.current_step = 0
        # 売買管理クラス
        self.transman = TransactionManager()
        # obs: len(self.cols_features) + one-hot(3)
        n_features = len(self.cols_features) + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        # ActionType.HOLD(0), .BUY(1), .SELL(2), .REPAY(3)
        self.action_space = gym.spaces.Discrete(len(ActionType))

    def _add_features(self, period: int) -> list:
        """
        特徴量の追加
        :param period:
        :return:
        """
        list_features = list()

        # 調整用係数
        factor_ticker = 10  # 調整因子（銘柄別）
        unit = 100  # 最小取引単位

        # 最初の株価（株価比率の算出用）
        price_start = self.df["Price"].iloc[0]

        # 1. 株価比率
        colname = "PriceRatio"
        self.df[colname] = self.df["Price"] / price_start
        list_features.append(colname)

        # 2. 累計出来高差分 / 最小取引単位
        colname = "dVol"
        # ログスケールは特徴量のスケールを調整するのに有効
        self.df[colname] = np.log1p(self.df["Volume"].diff() / unit) / factor_ticker
        list_features.append(colname)

        # 3. 組み込みのテクニカル分析（例: 期間60のSMAを追加）
        # colname_sma = "SMA60"
        # self.df[colname_sma] = ta.SMA(self.df["Price"], timeperiod=period) / price_start
        # list_features.append(colname_sma)
        # NOTE: talibを使用する場合、talibもインストールが必要です。ここではシンプルな特徴量に留めます。

        return list_features

    def _get_action_mask(self) -> np.ndarray:
        """
        行動マスク
        :return:
        """
        if self.current_step < self.period:
            # ウォーミングアップ期間
            return np.array([1, 0, 0, 0], dtype=np.int8)  # 強制HOLD
        if self.transman.position == PositionType.NONE:
            # 建玉なし: HOLD, BUY, SELL が可能 (REPAYは不可)
            return np.array([1, 1, 1, 0], dtype=np.int8)
        else:
            # 建玉あり: HOLD, REPAY が可能 (BUY, SELLは不可: ナンピンを許可しないため)
            return np.array([1, 0, 0, 1], dtype=np.int8)

    def _get_observation(self):
        if self.current_step >= len(self.df) or self.current_step < self.period:
            # データ終了またはウォーミングアップ期間は、特徴量をゼロ埋め
            features = [0.0] * len(self.cols_features)
        else:
            features = self.df.iloc[self.current_step][self.cols_features]

        obs = np.array(features, dtype=np.float32)

        # PositionType → one-hotエンコーディング [NONE, LONG, SHORT]
        # NONE=0, LONG=1, SHORT=2
        pos_onehot = np.eye(3)[self.transman.position.value].astype(np.float32)
        obs = np.concatenate([obs, pos_onehot])

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.period  # ウォーミングアップ期間終了からスタートとして扱う
        self.transman.clearAll()
        obs = self._get_observation()
        # 観測値と行動マスクを返す
        return obs, {"action_mask": self._get_action_mask()}

    def step(self, n_action: int):
        done = False

        if self.current_step >= len(self.df):
            # データ終了の場合、強制的にエピソードを終了
            done = True
            action = ActionType.HOLD
            reward = 0.0
            t = self.df.at[self.current_step - 1, "Time"]
            price = self.df.at[self.current_step - 1, "Price"]
        else:
            action = ActionType(n_action)
            t = self.df.at[self.current_step, "Time"]
            price = self.df.at[self.current_step, "Price"]
            reward = self.transman.setAction(action, t, price)

        obs = self._get_observation()

        if self.current_step >= len(self.df) - 1:
            # データ終端に到達した場合
            done = True
            # エピソード終了時に建玉が残っていたら強制返済（ペナルティは付与しない例）
            if self.transman.position != PositionType.NONE:
                self.transman.setAction(ActionType.REPAY, t, price)
                # ただし、最後の強制返済の報酬/ペナルティはエピソード全体の報酬には含めない設計にしても良い

        self.current_step += 1

        # info 辞書に総PnLと行動マスク
        info = {
            "pnl_total": self.transman.pnl_total,
            "action_mask": self._get_action_mask()
        }

        # NOTE: gymnasiumでは `truncated` を追加。データ終端が `terminated=True`、その他の早期終了条件が `truncated=True`
        terminated = done  # データ終端は終了(terminated)
        truncated = False  # 早期終了条件はなしと仮定

        return obs, reward, terminated, truncated, info


# ==============================================================================
# データ準備
# ==============================================================================

def load_data(file_path):
    """
    指定されたExcelファイルを読み込むか、ファイルが見つからない場合はダミーデータを生成する。
    """
    if os.path.exists(file_path):
        print(f"Excelファイル [{file_path}] を読み込みます。")
        try:
            # header=0 で1行目をヘッダーとして読み込み
            df = pd.read_excel(file_path, engine='openpyxl')
            # 必要な列のみを抽出
            df = df[['Time', 'Price', 'Volume']]
            return df
        except Exception as e:
            print(f"ファイルの読み込み中にエラーが発生しました: {e}")
            print("代わりにダミーデータを生成します。")
            return create_dummy_data()
    else:
        print(f"ファイル [{file_path}] が見つかりません。代わりにダミーデータを生成します。")
        return create_dummy_data()


def create_dummy_data(n_steps=5000):
    """
    tick_20250819.xlsx の形式を模倣したダミーデータ（1秒間隔）を生成
    """
    np.random.seed(42)
    print("--- ダミーデータ生成開始 ---")

    # 連続したタイムスタンプ（1秒間隔）
    start_time = datetime.datetime.now().timestamp()
    time_stamps = [start_time + i for i in range(n_steps)]

    # 株価のシミュレーション（ランダムウォーク）
    initial_price = 1000.0
    # 価格変動をシミュレート。ノイズにトレンドを少し加える
    trend = np.linspace(0, 10, n_steps)
    price_changes = np.random.randn(n_steps) * 0.1 + trend * 0.001
    prices = initial_price + np.cumsum(price_changes)

    # 出来高のシミュレーション
    volumes = np.abs(np.random.normal(loc=100, scale=30, size=n_steps)).astype(int)

    df = pd.DataFrame({
        "Time": time_stamps,
        "Price": prices,
        "Volume": volumes
    })
    print(f"--- ダミーデータ生成完了（{len(df)}ステップ） ---")
    return df


# ==============================================================================
# PPOエージェントの学習と評価
# ==============================================================================

if __name__ == '__main__':
    # 1. データ準備
    data_file = "excel/tick_20250819.xlsx"
    df_data = load_data(data_file)


    # 2. 環境のファクトリ関数
    # Stable-Baselines3の make_vec_env に渡すために関数化
    def make_env():
        # 環境リセット時にデータが変更されないよう、新しいデータフレームのコピーを渡す
        return TradingEnv(df=df_data.copy())


    # 3. ベクトル化環境の作成 (DummyVecEnvを使用)
    # n_envs=4 で4つの環境を並列実行（効率的な学習のため）
    n_envs = 4
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=0)

    # 4. PPOモデルの構築
    # MlpPolicy (多層パーセプトロンポリシー) を使用
    # n_steps はバッチサイズ。環境のステップ数に合わせて調整
    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=512, seed=42)

    # 5. 学習の実行
    total_timesteps = 50000  # 合計50,000ステップ学習
    print(f"\n--- PPO学習開始（合計 {total_timesteps} ステップ）---")
    # PPOは学習中に探索と利用のバランスを調整
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("--- 学習完了 ---")

    # 6. モデルの保存（オプション）
    # model.save("ppo_trading_model")
    # print("\nモデルを ppo_trading_model.zip に保存しました。")

    """
    # 7. 学習済みモデルのテスト
    print("\n--- 学習済みモデルの最終評価 ---")
    eval_env = make_env()
    obs, info = eval_env.reset()
    total_pnl = 0.0
    terminated = False
    truncated = False

    # 評価モードでは `deterministic=True` で行動を選択（探索を行わず、最も確率の高い行動を選択）
    while not (terminated or truncated):
        # 標準のPPOは行動マスクを自動で使わないが、環境内ではウォームアップ期間のHOLDは機能
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(
            action[0] if isinstance(action, np.ndarray) else action)
        total_pnl = info["pnl_total"]

    print(f"最終ステップ: {eval_env.current_step}")
    print(f"最終的な総損益 (Total PnL): {total_pnl:.2f}")

    # 8. 全ての取引履歴をDataFrameとして表示
    df_transactions = pd.DataFrame(eval_env.transman.dict_transaction)
    print("\n--- 取引履歴 ---")
    print(df_transactions.tail(10))  # 最新の10件を表示
    """

    # 7. 学習済みモデルのテスト (修正案)

    # テスト用のティックデータ（例：別の日のデータ）を読み込む
    # ※ここでは例として学習に使ったのと同じファイル名を使用しますが、実際は別のテスト用ファイルを使用することを推奨します。
    test_excel_file = "excel/tick_20250819.xlsx"  # ここを別のテストファイルに変更
    test_df_data = pd.read_excel(test_excel_file)
    print(f"\n--- 学習済みモデルの最終評価（テストデータ {test_excel_file}）---")

    # テスト環境を作成し、データを設定
    # 'raw_data'は環境クラスに渡すデータを想定
    #test_env = StockTradingEnv(df=test_df_data, initial_balance=1000000)
    # PPOモデルが VecEnv を使っている場合、テスト環境も VecEnv でラップする必要があります
    #test_vec_env = make_vec_env(lambda: test_env, n_envs=1)
    # 修正後のテスト環境作成部分 (TradingEnv を使用する場合)
    #test_env = TradingEnv(df=test_df_data)
    #test_vec_env = make_vec_env(lambda: TradingEnv(df=test_df_data), n_envs=1)
    # make_vec_env の中で環境作成用の関数を定義
    #test_vec_env = make_vec_env(lambda: TradingEnv(df=test_df_data), n_envs=1)

    # 環境をリセットし、最初の観測を取得
    #print(test_vec_env.reset())
    #obs, info = test_vec_env.reset()
    #terminated = False
    #truncated = False
    #total_reward = 0
    print("\n--- 学習済みモデルの最終評価 ---")
    eval_env = make_env()
    obs, info = eval_env.reset()
    total_pnl = 0.0
    terminated = False
    truncated = False


    while not terminated and not truncated:
        # モデルを使って行動を予測 (action は 1要素の NumPy 配列または 0次元配列)
        action, _states = model.predict(obs, deterministic=True)

        # action を環境に渡す前に、安全にスカラー値（Python int/float）に変換
        if isinstance(action, np.ndarray):
            # 0次元配列または要素数1の配列からスカラー値を取り出す
            # action.item() は 0次元配列または要素数1の配列からPythonスカラー値を取り出せるため、最も安全
            action_value = action.item()
        else:
            # action が既にスカラー値である場合
            action_value = action

        # 環境を1ステップ進める
        # action_value をそのまま渡します
        #obs, reward, terminated, truncated, info = eval_env.step([action_value])
        obs, reward, terminated, truncated, info = eval_env.step(action_value)

        total_pnl += reward[0]

        # 必要に応じて、ここでエージェントの行動（売買）や残高などを記録・表示

    print(f"--- テスト結果 ---")
    print(f"最終的な累積報酬（利益）: {total_pnl:.2f}")

    # 環境内の 'info' や最終残高などの評価指標を出力することも可能です
    final_info = info[0]  # info は VecEnv から返されるため、0番目の要素を取得
    # 例: print(f"最終残高: {final_info.get('current_balance', 'N/A')}")