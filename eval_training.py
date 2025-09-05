import pandas as pd

from modules.ppo_agent_20250905_2 import PPOAgent
from modules.trading_env_20250905_2 import TradingEnv

if __name__ == "__main__":
    file_excel = "excel/tick_20250819.xlsx"
    # 過去ティックデータの読み込み
    df = pd.read_excel(file_excel)
    # 期待フォーマット: Time, Price, Volume

    # タイムスタンプを datetime に変換
    df["Time"] = pd.to_datetime(df["Time"])

    # 必要に応じてソート & 欠損チェック
    df = df.sort_values("Time").reset_index(drop=True)

    # 環境を初期化
    env = TradingEnv(df)

    # PPO エージェントをロード
    agent = PPOAgent(env)

    # 学習（1営業日分を1エピソードとする）
    agent.train(num_episodes=1)

    # 結果を確認
    print("1日分の学習が完了しました")
