import datetime
import numpy as np
import pandas as pd
import os
# Stable-Baselines3からのインポート
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from modules.trading_env import TradingEnv


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
    # vec_env = make_vec_env(make_env, n_envs=n_envs, seed=0)
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
    model.save("ppo_trading_model")
    print("\nモデルを ppo_trading_model.zip に保存しました。")

    # 7. 学習済みモデルのテスト
    print("\n--- 学習済みモデルの最終評価 ---")
    eval_env = make_env()
    obs, info = eval_env.reset()
    total_reward = 0.0
    total_pnl = 0.0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # モデルを使って行動を予測 (action は 1要素の NumPy 配列または 0次元配列)
        action, _states = model.predict(obs, deterministic=True)

        # action を環境に渡す前に、安全にスカラー値（Python int/float）に変換
        action_value = action.item()

        # 環境を1ステップ進める
        # action_value をそのまま渡します
        obs, reward, terminated, truncated, info = eval_env.step(action_value)

        # モデル報酬
        total_reward += reward

        # 必要に応じて、ここでエージェントの行動（売買）や残高などを記録・表示

    # モデル報酬（総額）
    print(f"モデル報酬（総額）: {total_reward:.2f}")

    print("取引明細")
    print(pd.DataFrame(eval_env.transman.dict_transaction))
    if "pnl_total" in info.keys():
        print(f"--- テスト結果 ---")
        print(f"最終的な累積報酬（利益）: {info['pnl_total']:.2f}")
