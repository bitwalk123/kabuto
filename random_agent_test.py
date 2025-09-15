import pathlib

import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from funcs.conv import conv_date_str
from funcs.ios import get_excel_sheet
from funcs.tse import get_ticker_name_list
from modules.trading_env_20250914 import TradingEnv

if __name__ == "__main__":
    # ===== # 過去のティックデータ =====
    date_str = "20250819"
    path_excel = f"excel/tick_{date_str}.xlsx"
    code = "7011"
    name = get_ticker_name_list([code])[code]
    print(code, name)
    excel_file = str(pathlib.Path(path_excel).resolve())
    df = get_excel_sheet(excel_file, code)
    print(df)

    # ===== 環境初期化 =====
    env = TradingEnv(df)

    # ===== ランダムエージェント =====
    obs, info = env.reset()
    done = False
    rewards = []
    pnls = []
    steps = 0

    while not done:
        # 定義されたアクション空間からランダムにアクションを選択
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        pnls.append(info["pnl_total"])
        steps += 1

    print(
        f"Episode finished: steps={steps}, "
        f"total_reward={np.sum(rewards):.2f}, "
        f"final_pnl={pnls[-1]:.2f}"
    )

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 可視化
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['font.size'] = 14

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # 報酬分布
    ax[0].hist(rewards, bins=int(max(rewards) - min(rewards) + 0.5), alpha=0.7)
    ax[0].set_title("Reward distribution")
    ax[0].set_xlabel("Reward")
    ax[0].set_ylabel("Frequency")
    ax[0].set_yscale("log")
    ax[0].grid()

    # PnL 推移
    ax[1].plot(pnls, alpha=0.5)
    ax[1].set_title("PnL over steps")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Cumulative PnL")
    ax[1].grid()

    plt.suptitle(f"{name} ({code}) on {conv_date_str(date_str)}")
    plt.tight_layout()
    plt.show()
